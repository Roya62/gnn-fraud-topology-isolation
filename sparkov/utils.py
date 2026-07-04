"""
utils.py — Shared data loading, preprocessing, graph building,
           training loop, metrics, and visualisation for Sparkov.

Each model file (gatv2_model.py, gcn_model.py, graphsage_model.py,
mlp_baseline.py) imports from here, keeping all shared logic in one place.
"""

import gc
import copy
import os
import random
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for servers
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    log_loss, brier_score_loss,
    ConfusionMatrixDisplay, roc_curve, auc,
)

from config import CardFraudConfig

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
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_outcome_dir(cfg: CardFraudConfig) -> str:
    """Create the outcome directory if it does not exist. Returns the path."""
    path = os.path.abspath(cfg.OUTCOME_DIR)
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
#  DATA LOADING
# ============================================================================
def load_and_preprocess(
    train_path: str = "fraudTrain.csv",
    test_path: str  = "fraudTest.csv",
    cfg: CardFraudConfig = None,
) -> pd.DataFrame:
    """Load, engineer features, downsample non-fraud. Returns clean DataFrame."""
    if cfg is None:
        cfg = CardFraudConfig()

    df1 = pd.read_csv(test_path)
    df2 = pd.read_csv(train_path)
    df  = pd.concat([df1, df2], ignore_index=True).dropna()
    del df1, df2; gc.collect()
    print(f"Combined: {len(df):,} transactions")

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = ((df["trans_date_trans_time"] - df["dob"]).dt.days // 365).astype("int16")
    df["hour"]        = df["trans_date_trans_time"].dt.hour.astype("int8")
    df["weekday"]     = df["trans_date_trans_time"].dt.weekday.astype("int8")
    df["trans_month"] = df["trans_date_trans_time"].dt.month.astype("int8")
    df["is_fraud"]    = df["is_fraud"].astype("int8")
    df["sin_hour"]    = np.sin(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["cos_hour"]    = np.cos(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype("int8")

    nonfraud_idx = df.index[df["is_fraud"] == 0]
    n_drop = min(len(nonfraud_idx), cfg.DOWNSAMPLE_NONFRAUD)
    drop_idx = np.random.RandomState(cfg.SEED).choice(nonfraud_idx, size=n_drop, replace=False)
    df = df.drop(drop_idx).reset_index(drop=True)
    print(f"After downsample: {len(df):,} rows (fraud rate: {df['is_fraud'].mean():.4f})")

    cols_to_drop = ["Unnamed: 0", "first", "last", "street", "trans_num", "dob"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    df["timestamp"] = np.arange(len(df), dtype=np.int64)

    df["trans_time_diff"] = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .diff().dt.total_seconds().div(60).fillna(0).astype("float32"))

    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        df["distance"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2 +
            (df["long"] - df["merch_long"]) ** 2).astype("float32")

    if "cc_num" in df.columns and "amt" in df.columns:
        cc_avg = df.groupby("cc_num")["amt"].transform("mean")
        df["amt_to_avg"] = (df["amt"] / (cc_avg + 1e-10)).astype("float32")

    df["cc_num"] = pd.to_numeric(df["cc_num"], errors="coerce").fillna(-1)
    df = df.drop(columns=["trans_date_trans_time"], errors="ignore")
    print(f"Final shape: {df.shape}")
    gc.collect()
    return df


# ============================================================================
#  THRESHOLD SELECTION
# ============================================================================
def choose_threshold(y_true, y_prob, cfg: CardFraudConfig) -> float:
    if cfg.THRESHOLD_OVERRIDE is not None:
        return float(cfg.THRESHOLD_OVERRIDE)
    if cfg.USE_COST_THRESHOLD:
        best_thr, best_cost = 0.5, float("inf")
        for thr in np.linspace(0, 1, 1001):
            yh   = (y_prob >= thr).astype(int)
            cost = cfg.COST_FP * np.sum((yh==1)&(y_true==0)) + cfg.COST_FN * np.sum((yh==0)&(y_true==1))
            if cost < best_cost: best_cost, best_thr = cost, float(thr)
        return best_thr
    if cfg.TARGET_RECALL is not None:
        p, r, t = precision_recall_curve(y_true, y_prob)
        valid = np.where(r[:-1] >= cfg.TARGET_RECALL)[0]
        if len(valid) == 0: return float(t[np.argmax(r[:-1])]) if len(t) else 0.5
        f1 = 2 * p * r / (p + r + 1e-12)
        return float(t[valid[np.nanargmax(f1[valid])]])
    # Default: max-F1
    p, r, t = precision_recall_curve(y_true, y_prob)
    if len(t) == 0: return 0.5
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[int(np.nanargmax(f1[:-1]))])


# ============================================================================
#  METRICS
# ============================================================================
def eval_from_probs(y_true, y_prob, thr) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        acc    = (y_pred == y_true).mean(),
        f1     = f1_score(y_true, y_pred, zero_division=0),
        prec   = precision_score(y_true, y_pred, zero_division=0),
        rec    = recall_score(y_true, y_pred, zero_division=0),
        auc    = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.5,
        ap     = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.0,
        cm     = confusion_matrix(y_true, y_pred),
        logloss= log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
        brier  = brier_score_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
    )


def print_metrics(tag: str, m: dict):
    print(f"  {tag:>8} | F1 {m['f1']:.3f} | P {m['prec']:.3f} | R {m['rec']:.3f} | "
          f"AUC {m['auc']:.3f} | AP {m['ap']:.3f} | LL {m['logloss']:.4f} | Brier {m['brier']:.4f}")


# ============================================================================
#  FOLD PREPROCESSOR
# ============================================================================
class FoldPreprocessor:
    """LabelEncoder + StandardScaler fitted on training fold only."""

    def __init__(self, cat_cols: List[str], num_cols: List[str]):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.cardinalities: Dict[str, int] = {}

    def fit(self, df_train: pd.DataFrame) -> "FoldPreprocessor":
        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(df_train[col].astype(str))
            self.label_encoders[col] = le
            self.cardinalities[col]  = len(le.classes_)
        self.num_cols = [c for c in self.num_cols if c in df_train.columns]
        self.scaler.fit(df_train[self.num_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cat_cols:
            le, known = self.label_encoders[col], set(self.label_encoders[col].classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else len(le.classes_))
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df


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


def build_stratified_folds(df_dev, target_col, n_splits=5, group_key=None, bins=10, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if group_key and group_key in df_dev.columns:
        grp = df_dev.groupby(group_key)[target_col].mean().rename("prev").reset_index()
        try:    grp["bin"] = pd.qcut(grp["prev"], q=bins, labels=False, duplicates="drop")
        except: grp["bin"] = 0
        groups = grp[group_key].values
        ybins  = grp["bin"].fillna(0).astype(int).values
        folds  = []
        for tr_idx, va_idx in skf.split(groups, ybins):
            g_tr, g_va = set(groups[tr_idx]), set(groups[va_idx])
            folds.append((df_dev[df_dev[group_key].isin(g_tr)].reset_index(drop=True),
                          df_dev[df_dev[group_key].isin(g_va)].reset_index(drop=True)))
    else:
        folds = []
        for tr_idx, va_idx in skf.split(df_dev, df_dev[target_col].values):
            folds.append((df_dev.iloc[tr_idx].reset_index(drop=True),
                          df_dev.iloc[va_idx].reset_index(drop=True)))
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
        col, k, max_gs = spec["col"], spec.get("k", 1), spec.get("max_group_size", None)
        if col not in df_raw.columns: continue
        temp = (pd.DataFrame({"rel": df_raw[col].to_numpy(), "sk": sort_key, "nid": node_ids})
                .sort_values(["rel", "sk"], kind="mergesort").reset_index(drop=True))
        nids, rels = temp["nid"].to_numpy(), temp["rel"].to_numpy()
        breaks = np.where(rels[:-1] != rels[1:])[0] + 1
        starts = np.concatenate([[0], breaks]); ends = np.concatenate([breaks, [len(rels)]])
        for s, e in zip(starts, ends):
            gs = e - s
            if gs <= 1: continue
            gnodes = (nids[np.linspace(s, e-1, num=max_gs, dtype=int)]
                      if (max_gs and gs > max_gs) else nids[s:e])
            m = len(gnodes)
            for step in range(1, k + 1):
                if m <= step: break
                for a, b in zip(gnodes[:-step], gnodes[step:]):
                    if a != b: edge_set.add((int(a), int(b))); edge_set.add((int(b), int(a)))
    if cfg.ADD_GLOBAL_TIME_EDGES and cfg.GLOBAL_TIME_K > 0:
        ordered = node_ids[np.argsort(sort_key, kind="mergesort")]
        for step in range(1, cfg.GLOBAL_TIME_K + 1):
            if len(ordered) <= step: break
            for a, b in zip(ordered[:-step], ordered[step:]):
                if a != b: edge_set.add((int(a), int(b))); edge_set.add((int(b), int(a)))
    return edge_set


def _build_faiss_edges(cat_np, num_np, cfg):
    if not FAISS_AVAILABLE:
        raise ImportError("pip install faiss-cpu  (required for hybrid strategy)")
    features = np.ascontiguousarray(np.hstack([cat_np.astype("float32"), num_np.astype("float32")]))
    faiss.normalize_L2(features)
    index = faiss.IndexHNSWFlat(features.shape[1], cfg.FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = cfg.FAISS_EF_SEARCH; index.add(features)
    _, neighbors = index.search(features, cfg.FAISS_K + 1)
    edge_set = set()
    for i in range(len(features)):
        for j in range(1, cfg.FAISS_K + 1):
            nbr = int(neighbors[i, j])
            if 0 <= nbr < len(features) and nbr != i:
                edge_set.add((i, nbr)); edge_set.add((nbr, i))
    return edge_set


def _build_intra_group_edges(df_raw, cat_np, num_np, cfg):
    edge_set = set()
    sort_key = _make_sort_key(df_raw, cfg.SORT_KEY_COLS)
    all_feat = np.hstack([cat_np.astype("float32"), num_np.astype("float32")])
    for _, indices in df_raw.groupby(cfg.INTRA_GROUP_KEY).groups.items():
        indices = list(indices)
        if len(indices) > cfg.INTRA_MAX_GROUP_SIZE:
            indices = np.random.choice(indices, cfg.INTRA_MAX_GROUP_SIZE, replace=False).tolist()
        if len(indices) <= 1: continue
        si = sorted(indices, key=lambda i: sort_key[i])
        for i in range(len(si)):
            for j in range(max(0, i - cfg.INTRA_K_TEMPORAL), i):
                a, b = si[i], si[j]; edge_set.add((a,b)); edge_set.add((b,a))
        if SKLEARN_COSINE_AVAILABLE and len(indices) > cfg.INTRA_K_SIMILAR:
            sim = cosine_similarity(all_feat[indices])
            for i in range(len(indices)):
                scores = sim[i].copy(); scores[i] = -1.0
                for j in np.argsort(scores)[-cfg.INTRA_K_SIMILAR:]:
                    if scores[j] > cfg.INTRA_SIM_THRESHOLD:
                        a, b = indices[i], indices[j]; edge_set.add((a,b)); edge_set.add((b,a))
        for sub_col in (cfg.INTRA_SUB_RELATION_COLS or []):
            if sub_col not in df_raw.columns: continue
            for sub_idx in df_raw.iloc[indices].groupby(sub_col).groups.values():
                sub_list = sorted(list(sub_idx), key=lambda i: sort_key[i])
                for i in range(len(sub_list) - 1):
                    a, b = sub_list[i], sub_list[i+1]; edge_set.add((a,b)); edge_set.add((b,a))
    return edge_set


def build_graph_edges(strategy, df_raw, cat_np, num_np, cfg, verbose=False):
    n = len(df_raw)
    if strategy == "multi_relation":
        edge_set = _build_multi_relation_edges(df_raw, cfg)
    elif strategy == "hybrid":
        edge_set = _build_multi_relation_edges(df_raw, cfg)
        edge_set.update(_build_faiss_edges(cat_np, num_np, cfg))
    elif strategy == "intra_group":
        edge_set = _build_intra_group_edges(df_raw, cat_np, num_np, cfg)
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'")
    if cfg.SELF_LOOPS:
        for i in range(n): edge_set.add((i, i))
    if not edge_set:
        for i in range(n): edge_set.add((i, i))
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    if verbose:
        deg = np.bincount(edge_index[0].cpu().numpy(), minlength=n)
        print(f"  [{strategy}] nodes={n:,} edges={edge_index.size(1):,} "
              f"deg(min/med/mean/max)=({deg.min()},{np.median(deg):.1f},{deg.mean():.1f},{deg.max()})")
    return edge_index


# ============================================================================
#  SHARED TRAINING LOOP
# ============================================================================
def train_loop(model, cat_tr_d, num_tr_t, y_tr_t, edge_tr,
               cat_va_d, num_va_t, y_va_np, edge_va,
               cfg: CardFraudConfig, verbose: bool = False) -> nn.Module:
    pos = max(1, int(y_tr_t.sum().item()))
    neg = max(1, int(len(y_tr_t) - pos))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float, device=y_tr_t.device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt  = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sch  = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-5)
    best_ap, best_state, no_improve = -1.0, None, 0

    @torch.no_grad()
    def val_ap():
        model.eval()
        probs = torch.sigmoid(model(cat_va_d, num_va_t, edge_va)).cpu().numpy()
        return average_precision_score(y_va_np, probs) if len(np.unique(y_va_np)) == 2 else 0.0

    for ep in range(1, cfg.MAX_EPOCHS + 1):
        model.train(); opt.zero_grad()
        loss = crit(model(cat_tr_d, num_tr_t, edge_tr), y_tr_t)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(ep + 1e-8)
        if ep % cfg.EVAL_EVERY == 0:
            ap = val_ap()
            if ap > best_ap + 1e-6:
                best_ap, best_state, no_improve = ap, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
            if no_improve >= cfg.PATIENCE_CHECKS:
                if verbose: print(f"  Early stop at epoch {ep}")
                break
    if best_state is not None: model.load_state_dict(best_state)
    return model


# ============================================================================
#  TENSOR PREPARATION
# ============================================================================
def prepare_tensors(df_enc, cat_cols, num_cols, target_col, device):
    cat_np = df_enc[cat_cols].to_numpy(dtype=np.int64)
    num_np = df_enc[num_cols].to_numpy(dtype=np.float32)
    y_np   = df_enc[target_col].to_numpy(dtype=np.int64)
    cat_d  = {col: torch.tensor(cat_np[:, i], dtype=torch.long).to(device)
               for i, col in enumerate(cat_cols)}
    num_t  = torch.tensor(num_np, dtype=torch.float).to(device)
    return cat_np, num_np, y_np, cat_d, num_t


# ============================================================================
#  TEST ENSEMBLE
# ============================================================================
def run_test_ensemble(all_out, df_test, cfg, graph_strategy, device):
    """Average 5-fold predictions on the held-out test set."""
    test_probs = []
    is_mlp = (graph_strategy == "mlp_baseline")
    for fold_i, out in enumerate(all_out):
        print(f"  Test fold {fold_i + 1}/{len(all_out)}...")
        prep      = out["preprocessor"]
        df_te_enc = prep.transform(df_test)
        cat_te    = df_te_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
        num_te    = df_te_enc[prep.num_cols].to_numpy(dtype=np.float32)
        edge_te   = None if is_mlp else build_graph_edges(
            graph_strategy, df_test, cat_te, num_te, cfg).to(device)
        cat_te_d  = {col: torch.tensor(cat_te[:, i], dtype=torch.long).to(device)
                     for i, col in enumerate(cfg.CATEGORICAL_COLS)}
        num_te_t  = torch.tensor(num_te, dtype=torch.float).to(device)
        out["model"].eval()
        with torch.no_grad():
            probs = torch.sigmoid(out["model"](cat_te_d, num_te_t, edge_te)).cpu().numpy()
        test_probs.append(probs); cleanup()
    p_ens = np.mean(np.vstack(test_probs), axis=0)
    y_test = df_test[cfg.TARGET_COL].astype(int).to_numpy()
    thr_global = choose_threshold(
        np.concatenate([o["y_va"] for o in all_out]),
        np.concatenate([o["p_va"] for o in all_out]), cfg)
    return p_ens, y_test, thr_global


# ============================================================================
#  VISUALISATION  (all figures saved to outcomes/)
# ============================================================================
def save_results_plots(result: dict, cfg: CardFraudConfig):
    """
    Save four diagnostic plots for a single (strategy, model) run:
      1. Confusion matrix
      2. ROC curve
      3. Precision-Recall curve
      4. Threshold tuning curve
    Files are saved to cfg.OUTCOME_DIR as PNG.
    """
    out_dir = ensure_outcome_dir(cfg)
    y    = result["y_test"]
    p    = result["p_test_ens"]
    m    = result["test_metrics"]
    strat = result.get("graph_strategy", "unknown")
    arch  = result.get("model_arch", "unknown")
    thr   = result["thr_global"]
    tag   = f"{arch}_{strat}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Sparkov — {arch.upper()} | {strat} | F1={m['f1']:.3f} AUC={m['auc']:.3f}", fontsize=13)

    # 1. Confusion matrix
    ConfusionMatrixDisplay(m["cm"]).plot(ax=axes[0, 0], cmap="Blues")
    axes[0, 0].set_title(f"Confusion Matrix (thr={thr:.3f})")

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y, p)
    axes[0, 1].plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}", color="steelblue")
    axes[0, 1].plot([0, 1], [0, 1], "--", alpha=0.4, color="grey")
    axes[0, 1].set(xlabel="FPR", ylabel="TPR", title="ROC Curve"); axes[0, 1].legend()

    # 3. Precision-Recall curve
    pr, rc, _ = precision_recall_curve(y, p)
    axes[1, 0].plot(rc, pr, color="darkorange")
    axes[1, 0].set(xlabel="Recall", ylabel="Precision", title=f"PR Curve (AP={m['ap']:.3f})")

    # 4. Threshold tuning
    thrs = np.linspace(0, 1, 200)
    f1s  = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in thrs]
    prs  = [precision_score(y, (p >= t).astype(int), zero_division=0) for t in thrs]
    rcs  = [recall_score(y, (p >= t).astype(int), zero_division=0) for t in thrs]
    axes[1, 1].plot(thrs, f1s, label="F1",        color="green")
    axes[1, 1].plot(thrs, prs, label="Precision",  color="steelblue")
    axes[1, 1].plot(thrs, rcs, label="Recall",     color="darkorange")
    axes[1, 1].axvline(thr, linestyle="--", color="red", alpha=0.6, label=f"thr={thr:.3f}")
    axes[1, 1].set(xlabel="Threshold", ylabel="Score", title="Threshold Tuning"); axes[1, 1].legend()

    plt.tight_layout()
    fpath = os.path.join(out_dir, f"sparkov_{tag}_diagnostics.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def save_strategy_comparison(results: dict, cfg: CardFraudConfig):
    """
    Save a grouped bar chart comparing F1, Precision, Recall, AUC, AP
    across all (model, strategy) combinations in results.
    """
    out_dir = ensure_outcome_dir(cfg)
    metrics = ["f1", "prec", "rec", "auc", "ap"]
    labels  = list(results.keys())
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(labels))

    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 2), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i, (key, col) in enumerate(zip(labels, colors)):
        vals = [results[key]["test_metrics"][m] for m in metrics]
        ax.bar(x + i * w, vals, w, label=key, color=col)

    ax.set_xticks(x + w * (len(labels) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Sparkov — Strategy × Architecture Comparison")
    ax.legend(fontsize=7, ncol=max(1, len(labels) // 4))
    plt.tight_layout()
    fpath = os.path.join(out_dir, "sparkov_strategy_comparison.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


def save_results_csv(results: dict, cfg: CardFraudConfig):
    """Save a CSV summary of all test metrics to outcomes/."""
    out_dir = ensure_outcome_dir(cfg)
    rows = []
    for key, res in results.items():
        m = res["test_metrics"]
        rows.append({
            "run": key,
            "graph_strategy": res.get("graph_strategy", ""),
            "model_arch": res.get("model_arch", ""),
            "threshold": round(res["thr_global"], 4),
            "f1":    round(m["f1"],    4),
            "prec":  round(m["prec"],  4),
            "rec":   round(m["rec"],   4),
            "auc":   round(m["auc"],   4),
            "ap":    round(m["ap"],    4),
            "logloss": round(m["logloss"], 5),
            "brier":   round(m["brier"],   5),
        })
    fpath = os.path.join(out_dir, "sparkov_results_summary.csv")
    pd.DataFrame(rows).to_csv(fpath, index=False)
    print(f"  Saved: {fpath}")


def print_final_comparison(results: dict):
    print("\n" + "=" * 82)
    print(f"  {'Run':<32} | F1    | P     | R     | AUC   | AP    | Thr")
    print("-" * 82)
    for key, res in results.items():
        m, thr = res["test_metrics"], res["thr_global"]
        print(f"  {key:<32} | {m['f1']:.3f} | {m['prec']:.3f} | {m['rec']:.3f} | "
              f"{m['auc']:.3f} | {m['ap']:.3f} | {thr:.3f}")
    print("=" * 82)
