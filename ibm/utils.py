"""
utils.py — Shared data loading, preprocessing, graph building,
           training loop, metrics, and visualisation for IBM.

Each model file imports from here.
"""

import gc
import copy
import os
import random
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    log_loss, brier_score_loss,
    ConfusionMatrixDisplay, roc_curve, auc,
)

from config import IBMFraudConfig

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_COSINE_AVAILABLE = True
except ImportError:
    SKLEARN_COSINE_AVAILABLE = False


# ============================================================================
#  UTILITIES
# ============================================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def ensure_outcome_dir(cfg):
    path = os.path.abspath(cfg.OUTCOME_DIR)
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
#  DATA LOADING
# ============================================================================
def load_and_preprocess(path: str, cfg: IBMFraudConfig = None) -> pd.DataFrame:
    if cfg is None: cfg = IBMFraudConfig()
    df = pd.read_parquet(path) if path.lower().endswith((".parquet", ".pq")) else pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded: {len(df):,} transactions")

    # Normalise label
    y = df[cfg.TARGET_COL].astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    mapped = y.map(mapping)
    if mapped.isna().any():
        raise ValueError(f"Unsupported label values: {y[mapped.isna()].unique().tolist()}")
    df[cfg.TARGET_COL] = mapped.astype("int8")

    # Amount
    df["Amount"] = pd.to_numeric(
        df["Amount"].astype(str).str.replace(r"[£$€,]", "", regex=True).str.strip(),
        errors="coerce").fillna(-1).astype("float32")

    # Time features
    time_parsed = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    bad = time_parsed.isna()
    if bad.any(): time_parsed[bad] = pd.to_datetime(df.loc[bad, "Time"], errors="coerce")
    date_str = (df["Year"].astype(str).str.zfill(4) + "-" +
                df["Month"].astype(str).str.zfill(2) + "-" +
                df["Day"].astype(str).str.zfill(2))
    base_date = pd.to_datetime(date_str, errors="coerce")
    h = time_parsed.dt.hour.fillna(0).astype(int)
    m = time_parsed.dt.minute.fillna(0).astype(int)
    df["transaction_dt"] = base_date + pd.to_timedelta(h, unit="h") + pd.to_timedelta(m, unit="m")
    df["hour"]         = time_parsed.dt.hour.fillna(-1).astype("int16")
    df["minute"]       = time_parsed.dt.minute.fillna(-1).astype("int16")
    df["day_of_week"]  = df["transaction_dt"].dt.dayofweek.fillna(-1).astype("int8")
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype("int8")
    df["is_work_hour"] = ((df["hour"] >= 9) & (df["hour"] < 18)).astype("int8")
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"].clip(lower=0) / 24.0).astype("float32")
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"].clip(lower=0) / 24.0).astype("float32")

    # Downsample
    if cfg.DOWNSAMPLE:
        fraud_df = df[df[cfg.TARGET_COL] == 1]
        nonfraud_df = df[df[cfg.TARGET_COL] == 0]
        n_target = len(fraud_df) * cfg.DOWNSAMPLE_RATIO
        if len(nonfraud_df) > n_target:
            nonfraud_df = nonfraud_df.sample(n=n_target, random_state=cfg.SEED)
            df = pd.concat([fraud_df, nonfraud_df], ignore_index=True
                           ).sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)
            print(f"Downsampled: {len(fraud_df):,} fraud + {n_target:,} non-fraud")
        del fraud_df, nonfraud_df; gc.collect()

    fill_map = {"Use Chip": "Unknown", "Merchant City": "UNK",
                "Merchant State": "UNK", "Errors?": "None"}
    for col, val in fill_map.items():
        if col in df.columns: df[col] = df[col].fillna(val).astype(str)
    for col in ["Zip", "MCC", "Year", "Month", "Day", "Card", "Merchant Name", "User"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    df = df.reset_index(drop=True)
    print(f"After preprocessing: {len(df):,} rows | Fraud rate: {df[cfg.TARGET_COL].mean():.4f}")
    gc.collect()
    return df


# ============================================================================
#  TRAIN-FITTED FEATURES  (no leakage)
# ============================================================================
def add_user_spending_features(train_df, other_df=None, user_col="User", amount_col="Amount"):
    train_df   = train_df.copy()
    user_avg   = train_df.groupby(user_col)[amount_col].mean()
    global_avg = float(train_df[amount_col].mean())
    def apply_fn(df):
        df = df.copy()
        df["user_avg_amount"]     = df[user_col].map(user_avg).fillna(global_avg).astype("float32")
        denom = df["user_avg_amount"].replace(0, 1.0)
        df["amount_over_user_avg"]  = (df[amount_col] / denom).astype("float32")
        df["amount_minus_user_avg"] = (df[amount_col] - df["user_avg_amount"]).astype("float32")
        return df
    train_df = apply_fn(train_df)
    if other_df is None: return train_df, user_avg, global_avg
    return train_df, apply_fn(other_df), user_avg, global_avg


def fit_factor_maps(train_df, factor_cols):
    maps, cardinalities = {}, {}
    for col in factor_cols:
        vals = train_df[col].fillna("UNK").astype(str)
        uniques = pd.Index(vals.unique())
        maps[col] = {v: i for i, v in enumerate(uniques)}
        cardinalities[col] = len(uniques)
    return maps, cardinalities

def apply_factor_maps(df, factor_maps):
    df = df.copy()
    for col, mapping in factor_maps.items():
        df[col] = df[col].fillna("UNK").astype(str).map(mapping).fillna(-1).astype("int32")
    return df


class FoldPreprocessor:
    def __init__(self, cfg: IBMFraudConfig):
        self.cfg = cfg
        self.factor_maps = {}; self.factor_cardinalities = {}
        self.high_card_cols = []; self.low_card_cols = []
        self.dense_feature_cols = []
        self.means = self.stds = None

    def fit(self, train_df):
        self.high_card_cols = [c for c in self.cfg.HIGH_CARD_COLS if c in train_df.columns]
        self.low_card_cols  = [c for c in self.cfg.LOW_CARD_COLS  if c in train_df.columns]
        self.factor_maps, self.factor_cardinalities = fit_factor_maps(
            train_df, self.high_card_cols + self.low_card_cols)
        tmp = apply_factor_maps(train_df, self.factor_maps)
        self.dense_feature_cols = [c for c in self.cfg.DENSE_FEATURE_COLS if c in tmp.columns]
        X = tmp[self.dense_feature_cols].astype("float32").to_numpy(copy=True)
        self.means = X.mean(axis=0, dtype=np.float64).astype("float32")
        self.stds  = X.std(axis=0,  dtype=np.float64).astype("float32")
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, df):
        df      = apply_factor_maps(df.copy(), self.factor_maps)
        X_dense = ((df[self.dense_feature_cols].astype("float32").to_numpy(copy=True)
                    - self.means) / self.stds).astype("float32")
        X_high  = (df[self.high_card_cols].astype("int64").to_numpy(copy=True)
                   if self.high_card_cols else np.zeros((len(df), 0), dtype=np.int64))
        X_low   = (df[self.low_card_cols].astype("int64").to_numpy(copy=True)
                   if self.low_card_cols  else np.zeros((len(df), 0), dtype=np.int64))
        y = df[self.cfg.TARGET_COL].to_numpy(dtype=np.int64)
        return X_dense, X_high, X_low, y


# ============================================================================
#  THRESHOLD / METRICS
# ============================================================================
def choose_threshold(y_true, y_prob, cfg):
    if cfg.THRESHOLD_OVERRIDE is not None: return float(cfg.THRESHOLD_OVERRIDE)
    if cfg.USE_COST_THRESHOLD:
        best_thr, best_cost = 0.5, float("inf")
        for thr in np.linspace(0, 1, 1001):
            yh = (y_prob >= thr).astype(int)
            cost = cfg.COST_FP*np.sum((yh==1)&(y_true==0)) + cfg.COST_FN*np.sum((yh==0)&(y_true==1))
            if cost < best_cost: best_cost, best_thr = cost, float(thr)
        return best_thr
    if cfg.TARGET_RECALL is not None:
        p, r, t = precision_recall_curve(y_true, y_prob)
        valid = np.where(r[:-1] >= cfg.TARGET_RECALL)[0]
        if not len(valid): return float(t[np.argmax(r[:-1])]) if len(t) else 0.5
        f1 = 2*p*r/(p+r+1e-12); return float(t[valid[np.nanargmax(f1[valid])]])
    p, r, t = precision_recall_curve(y_true, y_prob)
    if not len(t): return 0.5
    f1 = 2*p*r/(p+r+1e-12); return float(t[int(np.nanargmax(f1[:-1]))])

def eval_from_probs(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        acc=( y_pred==y_true).mean(),
        f1=f1_score(y_true, y_pred, zero_division=0),
        prec=precision_score(y_true, y_pred, zero_division=0),
        rec=recall_score(y_true, y_pred, zero_division=0),
        auc=roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else 0.5,
        ap=average_precision_score(y_true, y_prob) if len(np.unique(y_true))==2 else 0.0,
        cm=confusion_matrix(y_true, y_pred),
        logloss=log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7)),
        brier=brier_score_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7)),
    )

def print_metrics(tag, m):
    print(f"  {tag:>8} | F1 {m['f1']:.3f} | P {m['prec']:.3f} | R {m['rec']:.3f} | "
          f"AUC {m['auc']:.3f} | AP {m['ap']:.3f} | LL {m['logloss']:.4f} | Brier {m['brier']:.4f}")


# ============================================================================
#  DATA SPLITTING
# ============================================================================
def split_groups_holdout(df, key, target_col, train_ratio=0.8, bins=10, seed=42):
    rng = np.random.RandomState(seed)
    grp = df.groupby(key)[target_col].mean().rename("prev").reset_index()
    try:    grp["bin"] = pd.qcut(grp["prev"], q=bins, labels=False, duplicates="drop")
    except: grp["bin"] = 0
    dev_groups, test_groups = [], []
    for _, gbin in grp.groupby("bin"):
        groups = gbin[key].values.copy(); rng.shuffle(groups)
        n = int(round(train_ratio * len(groups)))
        dev_groups.extend(groups[:n]); test_groups.extend(groups[n:])
    return (df[df[key].isin(set(dev_groups))].reset_index(drop=True),
            df[df[key].isin(set(test_groups))].reset_index(drop=True))

def build_group_stratified_folds(df_dev, group_key, target_col, n_splits=5, bins=10, seed=42):
    grp = df_dev.groupby(group_key)[target_col].mean().rename("prev").reset_index()
    try:    grp["bin"] = pd.qcut(grp["prev"], q=bins, labels=False, duplicates="drop")
    except: grp["bin"] = 0
    groups = grp[group_key].values; ybins = grp["bin"].fillna(0).astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, va_idx in skf.split(groups, ybins):
        g_tr, g_va = set(groups[tr_idx]), set(groups[va_idx])
        folds.append((df_dev[df_dev[group_key].isin(g_tr)].reset_index(drop=True),
                      df_dev[df_dev[group_key].isin(g_va)].reset_index(drop=True)))
    return folds


# ============================================================================
#  GRAPH BUILDERS
# ============================================================================
def _make_sort_key(df, sort_key_spec):
    key = np.zeros(len(df), dtype=np.int64)
    for col, mult in (sort_key_spec or {}).items():
        if col in df.columns:
            key += pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64).values * mult
    return key

def _build_multi_relation_edges(df_raw, cfg):
    edge_set = set()
    sort_key = _make_sort_key(df_raw, cfg.SORT_KEY_COLS)
    node_ids = np.arange(len(df_raw), dtype=np.int64)
    for spec in (cfg.MULTI_REL_SPECS or []):
        col, k, max_gs = spec["col"], spec.get("k",1), spec.get("max_group_size", None)
        if col not in df_raw.columns: continue
        temp = (pd.DataFrame({"rel": df_raw[col].to_numpy(), "sk": sort_key, "nid": node_ids})
                .sort_values(["rel","sk"], kind="mergesort").reset_index(drop=True))
        nids, rels = temp["nid"].to_numpy(), temp["rel"].to_numpy()
        breaks = np.where(rels[:-1]!=rels[1:])[0]+1
        starts = np.concatenate([[0],breaks]); ends = np.concatenate([breaks,[len(rels)]])
        for s, e in zip(starts, ends):
            gs = e-s
            if gs<=1: continue
            gnodes = (nids[np.linspace(s,e-1,num=max_gs,dtype=int)]
                      if (max_gs and gs>max_gs) else nids[s:e])
            m = len(gnodes)
            for step in range(1, k+1):
                if m<=step: break
                for a,b in zip(gnodes[:-step],gnodes[step:]):
                    if a!=b: edge_set.add((int(a),int(b))); edge_set.add((int(b),int(a)))
    if cfg.ADD_GLOBAL_TIME_EDGES and cfg.GLOBAL_TIME_K>0:
        ordered = node_ids[np.argsort(sort_key, kind="mergesort")]
        for step in range(1, cfg.GLOBAL_TIME_K+1):
            if len(ordered)<=step: break
            for a,b in zip(ordered[:-step],ordered[step:]):
                if a!=b: edge_set.add((int(a),int(b))); edge_set.add((int(b),int(a)))
    return edge_set

def _build_faiss_edges(X_dense, X_high, X_low, cfg):
    if not FAISS_AVAILABLE: raise ImportError("pip install faiss-cpu")
    features = np.ascontiguousarray(np.hstack([X_dense.astype("float32"),
                                               X_high.astype("float32"),
                                               X_low.astype("float32")]))
    faiss.normalize_L2(features)
    index = faiss.IndexHNSWFlat(features.shape[1], cfg.FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = cfg.FAISS_EF_SEARCH; index.add(features)
    _, neighbors = index.search(features, cfg.FAISS_K+1)
    edge_set = set()
    for i in range(len(features)):
        for j in range(1, cfg.FAISS_K+1):
            nbr = int(neighbors[i,j])
            if 0<=nbr<len(features) and nbr!=i:
                edge_set.add((i,nbr)); edge_set.add((nbr,i))
    return edge_set

def _build_intra_group_edges(df_raw, X_dense, X_high, X_low, cfg):
    edge_set = set()
    sort_key = _make_sort_key(df_raw, cfg.SORT_KEY_COLS)
    all_feat = np.hstack([X_dense.astype("float32"), X_high.astype("float32"), X_low.astype("float32")])
    for _, indices in df_raw.groupby(cfg.INTRA_GROUP_KEY).groups.items():
        indices = list(indices)
        if len(indices)>cfg.INTRA_MAX_GROUP_SIZE:
            indices = np.random.choice(indices, cfg.INTRA_MAX_GROUP_SIZE, replace=False).tolist()
        if len(indices)<=1: continue
        si = sorted(indices, key=lambda i: sort_key[i])
        for i in range(len(si)):
            for j in range(max(0, i-cfg.INTRA_K_TEMPORAL), i):
                a,b=si[i],si[j]; edge_set.add((a,b)); edge_set.add((b,a))
        if SKLEARN_COSINE_AVAILABLE and len(indices)>cfg.INTRA_K_SIMILAR:
            sim = cosine_similarity(all_feat[indices])
            for i in range(len(indices)):
                scores = sim[i].copy(); scores[i]=-1.0
                for j in np.argsort(scores)[-cfg.INTRA_K_SIMILAR:]:
                    if scores[j]>cfg.INTRA_SIM_THRESHOLD:
                        a,b=indices[i],indices[j]; edge_set.add((a,b)); edge_set.add((b,a))
        for sub_col in (cfg.INTRA_SUB_RELATION_COLS or []):
            if sub_col not in df_raw.columns: continue
            for sub_idx in df_raw.iloc[indices].groupby(sub_col).groups.values():
                sub_list = sorted(list(sub_idx), key=lambda i: sort_key[i])
                for i in range(len(sub_list)-1):
                    a,b=sub_list[i],sub_list[i+1]; edge_set.add((a,b)); edge_set.add((b,a))
    return edge_set

def build_graph_edges(strategy, df_raw, X_dense, X_high, X_low, cfg, verbose=False):
    n = len(df_raw)
    if strategy=="multi_relation":   edge_set = _build_multi_relation_edges(df_raw, cfg)
    elif strategy=="hybrid":
        edge_set = _build_multi_relation_edges(df_raw, cfg)
        edge_set.update(_build_faiss_edges(X_dense, X_high, X_low, cfg))
    elif strategy=="intra_group":    edge_set = _build_intra_group_edges(df_raw, X_dense, X_high, X_low, cfg)
    else: raise ValueError(f"Unknown strategy: '{strategy}'")
    if cfg.SELF_LOOPS:
        for i in range(n): edge_set.add((i,i))
    if not edge_set:
        for i in range(n): edge_set.add((i,i))
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    if verbose:
        deg = np.bincount(edge_index[0].cpu().numpy(), minlength=n)
        print(f"  [{strategy}] nodes={n:,} edges={edge_index.size(1):,} "
              f"deg(min/med/mean/max)=({deg.min()},{np.median(deg):.1f},{deg.mean():.1f},{deg.max()})")
    del edge_set; gc.collect()
    return edge_index


# ============================================================================
#  SAFE INFERENCE
# ============================================================================
@torch.no_grad()
def safe_inference(model, x_d, x_h, x_l, edge_index, device, use_amp=True):
    model.eval()
    try:
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(x_d, x_h, x_l, edge_index)
        return torch.sigmoid(logits).cpu().numpy()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("  [WARN] GPU OOM — falling back to CPU")
            cleanup(); model_cpu = model.cpu()
            logits = model_cpu(x_d.cpu(), x_h.cpu(), x_l.cpu(),
                               edge_index.cpu() if edge_index is not None else None)
            model.to(device); return torch.sigmoid(logits).numpy()
        raise


# ============================================================================
#  TRAINING LOOP
# ============================================================================
def train_loop(model, x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr,
               x_d_va, x_h_va, x_l_va, y_va_np, edge_va,
               cfg, device, verbose=False):
    pos = max(1, int(y_tr_t.sum().item())); neg = max(1, int(len(y_tr_t)-pos))
    pos_weight = torch.tensor([neg/pos], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt  = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sch  = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-5)
    use_amp = cfg.USE_AMP and device.type=="cuda"
    scaler  = GradScaler("cuda", enabled=use_amp)
    best_ap, best_state, no_improve = -1.0, None, 0

    def val_ap():
        probs = safe_inference(model, x_d_va, x_h_va, x_l_va, edge_va, device, use_amp)
        return average_precision_score(y_va_np, probs) if len(np.unique(y_va_np))==2 else 0.0

    for ep in range(1, cfg.MAX_EPOCHS+1):
        model.train(); opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            loss = crit(model(x_d_tr, x_h_tr, x_l_tr, edge_tr), y_tr_t)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step(ep+1e-8)
        if ep % cfg.EVAL_EVERY == 0:
            ap = val_ap()
            if ap > best_ap+1e-6: best_ap, best_state, no_improve = ap, copy.deepcopy(model.state_dict()), 0
            else: no_improve += 1
            if no_improve >= cfg.PATIENCE_CHECKS:
                if verbose: print(f"  Early stop at epoch {ep}"); break
    if best_state is not None: model.load_state_dict(best_state)
    return model


# ============================================================================
#  TEST ENSEMBLE
# ============================================================================
def run_test_ensemble(all_out, df_test, cfg, graph_strategy, device):
    test_probs = []; is_mlp = (graph_strategy=="mlp_baseline")
    use_amp = cfg.USE_AMP and device.type=="cuda"
    for fold_i, out in enumerate(all_out):
        print(f"  Test fold {fold_i+1}/{len(all_out)}...")
        prep = out["preprocessor"]
        df_te = df_test.copy()
        df_te["user_avg_amount"]     = df_te["User"].map(out["user_avg_map"]).fillna(out["global_avg"]).astype("float32")
        denom = df_te["user_avg_amount"].replace(0, 1.0)
        df_te["amount_over_user_avg"]  = (df_te["Amount"]/denom).astype("float32")
        df_te["amount_minus_user_avg"] = (df_te["Amount"]-df_te["user_avg_amount"]).astype("float32")
        X_d, X_h, X_l, _ = prep.transform(df_te)
        edge_te = None if is_mlp else build_graph_edges(
            graph_strategy, df_test, X_d, X_h, X_l, cfg).to(device)
        x_d = torch.tensor(X_d, dtype=torch.float32).to(device)
        x_h = torch.tensor(X_h, dtype=torch.long).to(device)
        x_l = torch.tensor(X_l, dtype=torch.long).to(device)
        model = out["model"].to(device)
        probs = safe_inference(model, x_d, x_h, x_l, edge_te, device, use_amp)
        test_probs.append(probs); out["model"] = model.cpu()
        del x_d, x_h, x_l; cleanup()
        if edge_te is not None: del edge_te
    p_ens = np.mean(np.vstack(test_probs), axis=0)
    y_test = df_test[cfg.TARGET_COL].astype(int).to_numpy()
    thr_global = choose_threshold(
        np.concatenate([o["y_va"] for o in all_out]),
        np.concatenate([o["p_va"] for o in all_out]), cfg)
    return p_ens, y_test, thr_global


# ============================================================================
#  VISUALISATION
# ============================================================================
def save_results_plots(result, cfg):
    out_dir = ensure_outcome_dir(cfg)
    y = result["y_test"]; p = result["p_test_ens"]
    m = result["test_metrics"]
    strat = result.get("graph_strategy", "unknown")
    arch  = result.get("model_arch", "unknown")
    thr   = result["thr_global"]
    tag   = f"{arch}_{strat}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"IBM — {arch.upper()} | {strat} | F1={m['f1']:.3f} AUC={m['auc']:.3f}", fontsize=13)

    ConfusionMatrixDisplay(m["cm"]).plot(ax=axes[0,0], cmap="Blues")
    axes[0,0].set_title(f"Confusion Matrix (thr={thr:.3f})")

    fpr, tpr, _ = roc_curve(y, p)
    axes[0,1].plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}", color="steelblue")
    axes[0,1].plot([0,1],[0,1],"--",alpha=0.4,color="grey")
    axes[0,1].set(xlabel="FPR",ylabel="TPR",title="ROC Curve"); axes[0,1].legend()

    pr, rc, _ = precision_recall_curve(y, p)
    axes[1,0].plot(rc, pr, color="darkorange")
    axes[1,0].set(xlabel="Recall", ylabel="Precision", title=f"PR Curve (AP={m['ap']:.3f})")

    thrs = np.linspace(0, 1, 200)
    axes[1,1].plot(thrs, [f1_score(y,(p>=t).astype(int),zero_division=0) for t in thrs], label="F1", color="green")
    axes[1,1].plot(thrs, [precision_score(y,(p>=t).astype(int),zero_division=0) for t in thrs], label="Precision", color="steelblue")
    axes[1,1].plot(thrs, [recall_score(y,(p>=t).astype(int),zero_division=0) for t in thrs], label="Recall", color="darkorange")
    axes[1,1].axvline(thr, linestyle="--", color="red", alpha=0.6, label=f"thr={thr:.3f}")
    axes[1,1].set(xlabel="Threshold", ylabel="Score", title="Threshold Tuning"); axes[1,1].legend()

    plt.tight_layout()
    fpath = os.path.join(out_dir, f"ibm_{tag}_diagnostics.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fpath}")

def save_strategy_comparison(results, cfg):
    out_dir = ensure_outcome_dir(cfg)
    metrics = ["f1","prec","rec","auc","ap"]; labels = list(results.keys())
    x = np.arange(len(metrics)); w = 0.8/max(1,len(labels))
    fig, ax = plt.subplots(figsize=(max(14,len(labels)*2), 5))
    colors = plt.cm.tab10(np.linspace(0,1,len(labels)))
    for i,(key,col) in enumerate(zip(labels,colors)):
        ax.bar(x+i*w, [results[key]["test_metrics"][m] for m in metrics], w, label=key, color=col)
    ax.set_xticks(x+w*(len(labels)-1)/2); ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0,1.05); ax.set_ylabel("Score")
    ax.set_title("IBM — Strategy × Architecture Comparison")
    ax.legend(fontsize=7, ncol=max(1,len(labels)//4)); plt.tight_layout()
    fpath = os.path.join(out_dir, "ibm_strategy_comparison.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fpath}")

def save_results_csv(results, cfg):
    out_dir = ensure_outcome_dir(cfg)
    rows = []
    for key, res in results.items():
        m = res["test_metrics"]
        rows.append({"run": key,
                     "graph_strategy": res.get("graph_strategy",""),
                     "model_arch": res.get("model_arch",""),
                     "threshold": round(res["thr_global"],4),
                     "f1": round(m["f1"],4), "prec": round(m["prec"],4),
                     "rec": round(m["rec"],4), "auc": round(m["auc"],4),
                     "ap": round(m["ap"],4), "logloss": round(m["logloss"],5),
                     "brier": round(m["brier"],5)})
    fpath = os.path.join(out_dir, "ibm_results_summary.csv")
    pd.DataFrame(rows).to_csv(fpath, index=False); print(f"  Saved: {fpath}")

def print_final_comparison(results):
    print("\n"+"="*82)
    print(f"  {'Run':<32} | F1    | P     | R     | AUC   | AP    | Thr"); print("-"*82)
    for key, res in results.items():
        m, thr = res["test_metrics"], res["thr_global"]
        print(f"  {key:<32} | {m['f1']:.3f} | {m['prec']:.3f} | {m['rec']:.3f} | "
              f"{m['auc']:.3f} | {m['ap']:.3f} | {thr:.3f}")
    print("="*82)
