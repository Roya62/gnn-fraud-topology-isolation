##############################################################################
# IBM CREDIT CARD FRAUD — GAT + GCN + GRAPHSAGE PIPELINE (MEMORY-OPTIMIZED)
# WITH MLP BASELINE FOR CONTROLLED ABLATION
#
# Dataset: IBM synthetic credit card transactions
#
# Graph strategies:
#   "multi_relation"  – temporal edges along Card, User, Merchant Name, MCC
#   "hybrid"          – multi_relation + FAISS k-NN feature-similarity edges
#   "intra_group"     – intra-Card temporal + similarity + Merchant Name chains
#   "mlp_baseline"    – no graph; identical features, loss, splits, threshold
#
# Model architectures:
#   "gatv2"           – GATConv, 2 layers, 2 heads, LayerNorm
#   "gcn"             – GCNConv, 2 layers, LayerNorm
#   "sage"            – SAGEConv, 2 layers, mean aggregation, LayerNorm
#
# This is NOT a fused GAT+GCN+SAGE hybrid model. It is a controlled comparison
# pipeline: each architecture is trained separately on the same graph strategies.
#
# Produces the results behind Tables 4, 5, and 6 of:
#   "Isolating Graph Topology from Model Architecture in GNN-Based Fraud
#    Detection: An Empirical Framework"
##############################################################################

# ========================== imports =========================================
import gc
import copy
import random
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

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
#  CONFIGURATION
# ============================================================================
class IBMFraudConfig:
    """All tuneable knobs for the IBM credit card pipeline."""

    # --- reproducibility ---
    SEED: int = 42

    # --- cross-validation ---
    N_SPLITS: int = 5
    TRAIN_RATIO: float = 0.8
    STRATIFY_BINS: int = 10

    # --- training ---
    MAX_EPOCHS: int = 200
    EVAL_EVERY: int = 5
    PATIENCE_CHECKS: int = 12
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    USE_AMP: bool = True

    # --- model (shared) ---
    DROPOUT: float = 0.30
    HIGH_CARD_EMB_DIM: int = 8
    LOW_CARD_EMB_DIM: int = 4

    # --- model (GAT-specific) ---
    HIDDEN_DIM: int = 32
    HEADS: int = 2

    # --- model (GCN-specific) ---
    GCN_HIDDEN_DIM: int = 64

    # --- model (GraphSAGE-specific) ---
    SAGE_HIDDEN_DIM: int = 64
    SAGE_AGGR: str = "mean"

    # --- model (MLP baseline) ---
    MLP_HIDDEN_DIMS: List[int] = [256, 128]

    # --- threshold policy ---
    THRESHOLD_OVERRIDE: Optional[float] = None
    USE_COST_THRESHOLD: bool = False
    COST_FN: float = 100.0
    COST_FP: float = 1.0
    TARGET_RECALL: Optional[float] = None

    # --- dataset schema ---
    TARGET_COL: str = "Is Fraud?"
    GROUP_KEY: str = "User"

    HIGH_CARD_COLS: List[str] = ["Merchant City", "Merchant State", "MCC"]
    LOW_CARD_COLS: List[str] = ["Use Chip", "Errors?"]

    DENSE_FEATURE_COLS: List[str] = [
        "Amount", "user_avg_amount", "amount_over_user_avg",
        "amount_minus_user_avg", "Zip", "day_of_week", "is_weekend",
        "is_work_hour", "hour", "minute", "hour_sin", "hour_cos",
        "Year", "Month", "Day",
    ]

    # --- temporal sort key ---
    SORT_KEY_COLS: Dict[str, int] = {
        "Year": 100_000_000,
        "Month": 1_000_000,
        "Day": 10_000,
        "hour": 100,
        "minute": 1,
    }

    # --- graph: multi-relation temporal ---
    MULTI_REL_SPECS: List[Dict[str, Any]] = [
        {"col": "Card",          "k": 1, "max_group_size": 500},
        {"col": "User",          "k": 1, "max_group_size": 500},
        {"col": "Merchant Name", "k": 1, "max_group_size": 500},
        {"col": "MCC",           "k": 1, "max_group_size": 1000},
    ]
    ADD_GLOBAL_TIME_EDGES: bool = False
    GLOBAL_TIME_K: int = 2
    SELF_LOOPS: bool = True

    # --- graph: hybrid ---
    FAISS_K: int = 4
    FAISS_HNSW_M: int = 32
    FAISS_EF_SEARCH: int = 64

    # --- graph: intra-group ---
    INTRA_GROUP_KEY: str = "Card"
    INTRA_MAX_GROUP_SIZE: int = 150
    INTRA_K_TEMPORAL: int = 2
    INTRA_K_SIMILAR: int = 4
    INTRA_SIM_THRESHOLD: float = 0.5
    INTRA_SUB_RELATION_COLS: List[str] = ["Merchant Name"]

    # --- downsampling ---
    DOWNSAMPLE: bool = True
    DOWNSAMPLE_RATIO: int = 10


# ============================================================================
#  SEED / DEVICE / CLEANUP
# ============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
#  DATA LOADING & GLOBAL PREPROCESSING
# ============================================================================
def normalize_label_column(df: pd.DataFrame, label_col: str = "Is Fraud?") -> pd.DataFrame:
    df = df.copy()
    y = df[label_col].astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    mapped = y.map(mapping)
    if mapped.isna().any():
        bad_vals = y[mapped.isna()].unique().tolist()
        raise ValueError(f"Unsupported values in '{label_col}': {bad_vals}")
    df[label_col] = mapped.astype("int8")
    return df


def clean_amount_column(amount_series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        amount_series.astype(str)
        .str.replace(r"[£$€,]", "", regex=True)
        .str.strip(),
        errors="coerce",
    )


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amount"] = clean_amount_column(df["Amount"]).fillna(-1).astype("float32")

    time_parsed = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    bad_time = time_parsed.isna()
    if bad_time.any():
        fallback = pd.to_datetime(df.loc[bad_time, "Time"], errors="coerce")
        time_parsed.loc[bad_time] = fallback

    date_str = (
        df["Year"].astype(str).str.zfill(4) + "-"
        + df["Month"].astype(str).str.zfill(2) + "-"
        + df["Day"].astype(str).str.zfill(2)
    )
    base_date = pd.to_datetime(date_str, errors="coerce")

    hour_tmp = time_parsed.dt.hour.fillna(0).astype(int)
    minute_tmp = time_parsed.dt.minute.fillna(0).astype(int)

    df["transaction_dt"] = (
        base_date
        + pd.to_timedelta(hour_tmp, unit="h")
        + pd.to_timedelta(minute_tmp, unit="m")
    )
    df["hour"] = time_parsed.dt.hour.fillna(-1).astype("int16")
    df["minute"] = time_parsed.dt.minute.fillna(-1).astype("int16")
    df["day_of_week"] = df["transaction_dt"].dt.dayofweek.fillna(-1).astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_work_hour"] = ((df["hour"] >= 9) & (df["hour"] < 18)).astype("int8")

    valid_hour = df["hour"].clip(lower=0)
    df["hour_sin"] = np.sin(2 * np.pi * valid_hour / 24.0).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * valid_hour / 24.0).astype("float32")
    return df


def preprocess_df(df: pd.DataFrame, cfg: IBMFraudConfig = None) -> pd.DataFrame:
    if cfg is None:
        cfg = IBMFraudConfig()

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    print(f"Input dataset: {len(df):,} transactions")

    df = normalize_label_column(df, label_col=cfg.TARGET_COL)
    df = add_datetime_features(df)

    if cfg.DOWNSAMPLE:
        fraud_df = df[df[cfg.TARGET_COL] == 1]
        nonfraud_df = df[df[cfg.TARGET_COL] == 0]
        n_fraud = len(fraud_df)
        n_nonfraud_target = n_fraud * cfg.DOWNSAMPLE_RATIO

        if len(nonfraud_df) > n_nonfraud_target:
            nonfraud_sample = nonfraud_df.sample(n=n_nonfraud_target, random_state=cfg.SEED)
            df = pd.concat([fraud_df, nonfraud_sample], ignore_index=True)
            df = df.sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)
            print(
                f"Downsampled: kept all {n_fraud:,} fraud + "
                f"{n_nonfraud_target:,} non-fraud (1:{cfg.DOWNSAMPLE_RATIO} ratio)"
            )
        else:
            print(f"No downsampling needed (only {len(nonfraud_df):,} non-fraud)")

        del fraud_df, nonfraud_df
        gc.collect()

    fill_map = {
        "Use Chip": "Unknown",
        "Merchant City": "UNK",
        "Merchant State": "UNK",
        "Errors?": "None",
    }
    for col, val in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(val).astype(str)

    for col in ["Zip", "MCC", "Year", "Month", "Day", "Card", "Merchant Name", "User"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    df = df.reset_index(drop=True)
    print(f"After preprocessing: {len(df):,} transactions")
    print(f"Fraud rate: {df[cfg.TARGET_COL].mean():.4f}")
    print(f"Shape: {df.shape}")
    gc.collect()
    return df


def load_and_preprocess(path: str, cfg: IBMFraudConfig = None) -> pd.DataFrame:
    if cfg is None:
        cfg = IBMFraudConfig()
    path_lower = str(path).lower()
    if path_lower.endswith(".parquet") or path_lower.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return preprocess_df(df, cfg)


# ============================================================================
#  TRAIN-FITTED PREPROCESSING (no leakage)
# ============================================================================
def add_user_spending_features(train_df, other_df=None, user_col="User", amount_col="Amount"):
    train_df = train_df.copy()
    user_avg = train_df.groupby(user_col)[amount_col].mean()
    global_avg = float(train_df[amount_col].mean())

    def apply_fn(df):
        df = df.copy()
        df["user_avg_amount"] = df[user_col].map(user_avg).fillna(global_avg).astype("float32")
        denom = df["user_avg_amount"].replace(0, 1.0)
        df["amount_over_user_avg"] = (df[amount_col] / denom).astype("float32")
        df["amount_minus_user_avg"] = (df[amount_col] - df["user_avg_amount"]).astype("float32")
        return df

    train_df = apply_fn(train_df)
    if other_df is None:
        return train_df, user_avg, global_avg
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
        df[col] = (
            df[col].fillna("UNK").astype(str)
            .map(mapping).fillna(-1).astype("int32")
        )
    return df


class FoldPreprocessor:
    def __init__(self, cfg: IBMFraudConfig):
        self.cfg = cfg
        self.factor_maps: Dict[str, Dict] = {}
        self.factor_cardinalities: Dict[str, int] = {}
        self.high_card_cols: List[str] = []
        self.low_card_cols: List[str] = []
        self.dense_feature_cols: List[str] = []
        self.means: np.ndarray = None
        self.stds: np.ndarray = None

    def fit(self, train_df: pd.DataFrame):
        self.high_card_cols = [c for c in self.cfg.HIGH_CARD_COLS if c in train_df.columns]
        self.low_card_cols = [c for c in self.cfg.LOW_CARD_COLS if c in train_df.columns]
        factor_cols = self.high_card_cols + self.low_card_cols
        self.factor_maps, self.factor_cardinalities = fit_factor_maps(train_df, factor_cols)

        train_tmp = apply_factor_maps(train_df, self.factor_maps)
        self.dense_feature_cols = [c for c in self.cfg.DENSE_FEATURE_COLS if c in train_tmp.columns]
        X = train_tmp[self.dense_feature_cols].astype("float32").to_numpy(copy=True)
        self.means = X.mean(axis=0, dtype=np.float64).astype("float32")
        self.stds = X.std(axis=0, dtype=np.float64).astype("float32")
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, df: pd.DataFrame):
        df = apply_factor_maps(df.copy(), self.factor_maps)
        X_dense = df[self.dense_feature_cols].astype("float32").to_numpy(copy=True)
        X_dense = ((X_dense - self.means) / self.stds).astype("float32")
        X_high = (
            df[self.high_card_cols].astype("int64").to_numpy(copy=True)
            if self.high_card_cols else np.zeros((len(df), 0), dtype=np.int64)
        )
        X_low = (
            df[self.low_card_cols].astype("int64").to_numpy(copy=True)
            if self.low_card_cols else np.zeros((len(df), 0), dtype=np.int64)
        )
        y = df[self.cfg.TARGET_COL].to_numpy(dtype=np.int64)
        return X_dense, X_high, X_low, y


# ============================================================================
#  THRESHOLD HELPERS
# ============================================================================
def pick_threshold_maxF1(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[int(np.nanargmax(f1[:-1]))])


def pick_threshold_cost(y_true, y_prob, cost_fn, cost_fp):
    best_thr, best_cost = 0.5, float("inf")
    y = y_true.astype(int)
    for thr in np.linspace(0, 1, 1001):
        yh = (y_prob >= thr).astype(int)
        cost = cost_fp * np.sum((yh == 1) & (y == 0)) + cost_fn * np.sum((yh == 0) & (y == 1))
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)
    return best_thr


def pick_threshold_recall(y_true, y_prob, target_recall):
    p, r, t = precision_recall_curve(y_true, y_prob)
    if len(t) == 0:
        return 0.5
    valid = np.where(r[:-1] >= target_recall)[0]
    if len(valid) == 0:
        return float(t[np.argmax(r[:-1])])
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[valid[np.nanargmax(f1[valid])]])


def choose_threshold(y_true, y_prob, cfg: IBMFraudConfig):
    if cfg.THRESHOLD_OVERRIDE is not None:
        return float(cfg.THRESHOLD_OVERRIDE)
    if cfg.USE_COST_THRESHOLD:
        return pick_threshold_cost(y_true, y_prob, cfg.COST_FN, cfg.COST_FP)
    if cfg.TARGET_RECALL is not None:
        return pick_threshold_recall(y_true, y_prob, cfg.TARGET_RECALL)
    return pick_threshold_maxF1(y_true, y_prob)


# ============================================================================
#  METRICS
# ============================================================================
def eval_from_probs(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        acc=(y_pred == y_true).mean(),
        f1=f1_score(y_true, y_pred, zero_division=0),
        prec=precision_score(y_true, y_pred, zero_division=0),
        rec=recall_score(y_true, y_pred, zero_division=0),
        auc=roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.5,
        ap=average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.0,
        cm=confusion_matrix(y_true, y_pred),
        logloss=log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
        brier=brier_score_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
    )


def print_metrics(tag, m, nd=3, ndloss=4):
    print(
        f"  {tag:>8} | "
        f"Acc {m['acc']:.{nd}f} | F1 {m['f1']:.{nd}f} | "
        f"P {m['prec']:.{nd}f} | R {m['rec']:.{nd}f} | "
        f"AUC {m['auc']:.{nd}f} | AP {m['ap']:.{nd}f} | "
        f"LL {m['logloss']:.{ndloss}f} | Brier {m['brier']:.{ndloss}f}"
    )
    print(f"           CM:\n{m['cm']}\n")


# ============================================================================
#  DATA SPLITTING (group-aware by User)
# ============================================================================
def split_groups_holdout(df, key, target_col, train_ratio=0.8, bins=10, seed=42):
    rng = np.random.RandomState(seed)
    grp = df.groupby(key)[target_col].mean().rename("prev").reset_index()
    try:
        grp["bin"] = pd.qcut(grp["prev"], q=bins, labels=False, duplicates="drop")
    except ValueError:
        grp["bin"] = 0

    dev_groups, test_groups = [], []
    for _, gbin in grp.groupby("bin"):
        groups = gbin[key].values.copy()
        rng.shuffle(groups)
        n_dev = int(round(train_ratio * len(groups)))
        dev_groups.extend(groups[:n_dev])
        test_groups.extend(groups[n_dev:])

    return (
        df[df[key].isin(set(dev_groups))].reset_index(drop=True),
        df[df[key].isin(set(test_groups))].reset_index(drop=True),
    )


def build_group_stratified_folds(df_dev, group_key, target_col, n_splits=5, bins=10, seed=42):
    grp = df_dev.groupby(group_key)[target_col].mean().rename("prev").reset_index()
    try:
        grp["bin"] = pd.qcut(grp["prev"], q=bins, labels=False, duplicates="drop")
    except ValueError:
        grp["bin"] = 0

    groups = grp[group_key].values
    ybins = grp["bin"].fillna(0).astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for tr_idx, va_idx in skf.split(groups, ybins):
        g_tr, g_va = set(groups[tr_idx]), set(groups[va_idx])
        folds.append((
            df_dev[df_dev[group_key].isin(g_tr)].reset_index(drop=True),
            df_dev[df_dev[group_key].isin(g_va)].reset_index(drop=True),
        ))
    return folds


# ============================================================================
#  GRAPH BUILDERS
# ============================================================================
def _make_sort_key(df: pd.DataFrame, sort_key_spec: Dict[str, int]) -> np.ndarray:
    key = np.zeros(len(df), dtype=np.int64)
    for col, mult in (sort_key_spec or {}).items():
        if col in df.columns:
            key += (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0).astype(np.int64).values * mult
            )
    return key


def _build_multi_relation_edges(df_raw, cfg: IBMFraudConfig) -> set:
    edge_set = set()
    sort_key = _make_sort_key(df_raw, cfg.SORT_KEY_COLS)
    n = len(df_raw)
    node_ids = np.arange(n, dtype=np.int64)

    for spec in (cfg.MULTI_REL_SPECS or []):
        col = spec["col"]
        k = spec.get("k", 1)
        max_gs = spec.get("max_group_size", None)
        if col not in df_raw.columns:
            continue

        temp = pd.DataFrame({
            "rel": df_raw[col].to_numpy(), "sk": sort_key, "nid": node_ids,
        }).sort_values(["rel", "sk"], kind="mergesort").reset_index(drop=True)

        nids = temp["nid"].to_numpy()
        rels = temp["rel"].to_numpy()

        breaks = np.where(rels[:-1] != rels[1:])[0] + 1
        starts = np.concatenate([[0], breaks])
        ends = np.concatenate([breaks, [len(rels)]])

        for s, e in zip(starts, ends):
            gs = e - s
            if gs <= 1:
                continue
            if max_gs and gs > max_gs:
                gnodes = nids[np.linspace(s, e - 1, num=max_gs, dtype=int)]
            else:
                gnodes = nids[s:e]

            m = len(gnodes)
            for step in range(1, k + 1):
                if m <= step:
                    break
                for a, b in zip(gnodes[:-step], gnodes[step:]):
                    if a != b:
                        edge_set.add((int(a), int(b)))
                        edge_set.add((int(b), int(a)))

    if cfg.ADD_GLOBAL_TIME_EDGES and cfg.GLOBAL_TIME_K > 0:
        ordered = node_ids[np.argsort(sort_key, kind="mergesort")]
        for step in range(1, cfg.GLOBAL_TIME_K + 1):
            if len(ordered) <= step:
                break
            for a, b in zip(ordered[:-step], ordered[step:]):
                if a != b:
                    edge_set.add((int(a), int(b)))
                    edge_set.add((int(b), int(a)))

    return edge_set


def _build_faiss_edges(X_dense, X_high, X_low, cfg: IBMFraudConfig) -> set:
    if not FAISS_AVAILABLE:
        raise ImportError("faiss required for 'hybrid' strategy. pip install faiss-cpu")
    features = np.ascontiguousarray(np.hstack([
        X_dense.astype("float32"), X_high.astype("float32"), X_low.astype("float32"),
    ]))
    faiss.normalize_L2(features)
    index = faiss.IndexHNSWFlat(features.shape[1], cfg.FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = cfg.FAISS_EF_SEARCH
    index.add(features)
    _, neighbors = index.search(features, cfg.FAISS_K + 1)

    edge_set = set()
    for i in range(len(features)):
        for j in range(1, cfg.FAISS_K + 1):
            nbr = int(neighbors[i, j])
            if 0 <= nbr < len(features) and nbr != i:
                edge_set.add((i, nbr))
                edge_set.add((nbr, i))
    return edge_set


def _build_hybrid_edges(df_raw, X_dense, X_high, X_low, cfg: IBMFraudConfig) -> set:
    edge_set = _build_multi_relation_edges(df_raw, cfg)
    edge_set.update(_build_faiss_edges(X_dense, X_high, X_low, cfg))
    return edge_set


def _build_intra_group_edges(df_raw, X_dense, X_high, X_low, cfg: IBMFraudConfig) -> set:
    edge_set = set()
    group_col = cfg.INTRA_GROUP_KEY
    if group_col not in df_raw.columns:
        raise ValueError(f"Intra-group key '{group_col}' not in dataframe")

    sort_key = _make_sort_key(df_raw, cfg.SORT_KEY_COLS)
    all_features = np.hstack([
        X_dense.astype("float32"), X_high.astype("float32"), X_low.astype("float32")
    ])

    for _, indices in df_raw.groupby(group_col).groups.items():
        indices = list(indices)
        if len(indices) > cfg.INTRA_MAX_GROUP_SIZE:
            indices = np.random.choice(indices, cfg.INTRA_MAX_GROUP_SIZE, replace=False).tolist()
        if len(indices) <= 1:
            continue

        sorted_idx = sorted(indices, key=lambda i: sort_key[i])

        for i in range(len(sorted_idx)):
            for j in range(max(0, i - cfg.INTRA_K_TEMPORAL), i):
                a, b = sorted_idx[i], sorted_idx[j]
                edge_set.add((a, b))
                edge_set.add((b, a))

        if SKLEARN_COSINE_AVAILABLE and len(indices) > cfg.INTRA_K_SIMILAR:
            sim = cosine_similarity(all_features[indices])
            for i in range(len(indices)):
                scores = sim[i].copy()
                scores[i] = -1.0
                for j in np.argsort(scores)[-cfg.INTRA_K_SIMILAR:]:
                    if scores[j] > cfg.INTRA_SIM_THRESHOLD:
                        a, b = indices[i], indices[j]
                        edge_set.add((a, b))
                        edge_set.add((b, a))

        for sub_col in (cfg.INTRA_SUB_RELATION_COLS or []):
            if sub_col not in df_raw.columns:
                continue
            for sub_idx in df_raw.iloc[indices].groupby(sub_col).groups.values():
                sub_list = sorted(list(sub_idx), key=lambda i: sort_key[i])
                for i in range(len(sub_list) - 1):
                    a, b = sub_list[i], sub_list[i + 1]
                    edge_set.add((a, b))
                    edge_set.add((b, a))

    return edge_set


def build_graph_edges(
    strategy: str,
    df_raw: pd.DataFrame,
    X_dense: np.ndarray,
    X_high: np.ndarray,
    X_low: np.ndarray,
    cfg: IBMFraudConfig,
    verbose: bool = False,
) -> torch.Tensor:
    n = len(df_raw)

    if strategy == "multi_relation":
        edge_set = _build_multi_relation_edges(df_raw, cfg)
    elif strategy == "hybrid":
        edge_set = _build_hybrid_edges(df_raw, X_dense, X_high, X_low, cfg)
    elif strategy == "intra_group":
        edge_set = _build_intra_group_edges(df_raw, X_dense, X_high, X_low, cfg)
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'")

    if cfg.SELF_LOOPS:
        for i in range(n):
            edge_set.add((i, i))
    if not edge_set:
        for i in range(n):
            edge_set.add((i, i))

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()

    if verbose:
        deg = np.bincount(edge_index[0].cpu().numpy(), minlength=n)
        print(
            f"  Graph [{strategy}] -> nodes={n:,}, edges={edge_index.size(1):,}, "
            f"deg(min/med/mean/max)=({deg.min()}, {np.median(deg):.1f}, "
            f"{deg.mean():.1f}, {deg.max()})"
        )

    del edge_set
    gc.collect()
    return edge_index


# ============================================================================
#  MODELS
# ============================================================================
class _EmbeddingEncoder(nn.Module):
    """Shared input encoding: high-card embeddings + low-card embeddings + dense."""

    def __init__(
        self,
        dense_in_dim: int,
        high_card_cols: List[str],
        low_card_cols: List[str],
        factor_cardinalities: Dict[str, int],
        high_card_emb_dim: int = 8,
        low_card_emb_dim: int = 4,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.dropout = dropout
        self.high_card_cols = list(high_card_cols)
        self.low_card_cols = list(low_card_cols)

        self.high_emb_layers = nn.ModuleDict({
            col: nn.Embedding(int(factor_cardinalities[col]) + 2, high_card_emb_dim)
            for col in high_card_cols
        })
        self.low_emb_layers = nn.ModuleDict({
            col: nn.Embedding(int(factor_cardinalities[col]) + 2, low_card_emb_dim)
            for col in low_card_cols
        })

        self.out_dim = (
            dense_in_dim
            + len(high_card_cols) * high_card_emb_dim
            + len(low_card_cols) * low_card_emb_dim
        )

    def encode(self, x_dense, x_high, x_low) -> torch.Tensor:
        parts = [x_dense]
        if self.high_card_cols:
            parts.append(torch.cat([
                self.high_emb_layers[col](x_high[:, i] + 1)
                for i, col in enumerate(self.high_card_cols)
            ], dim=1))
        if self.low_card_cols:
            parts.append(torch.cat([
                self.low_emb_layers[col](x_low[:, i] + 1)
                for i, col in enumerate(self.low_card_cols)
            ], dim=1))
        return torch.cat(parts, dim=1)


class IBMGATNet(_EmbeddingEncoder):
    """GATConv-based fraud detector. Two GATConv layers, multi-head attention."""

    def __init__(
        self,
        dense_in_dim: int,
        high_card_cols: List[str],
        low_card_cols: List[str],
        factor_cardinalities: Dict[str, int],
        hidden: int = 32,
        heads: int = 2,
        dropout: float = 0.30,
        high_card_emb_dim: int = 8,
        low_card_emb_dim: int = 4,
    ):
        super().__init__(
            dense_in_dim, high_card_cols, low_card_cols,
            factor_cardinalities, high_card_emb_dim, low_card_emb_dim, dropout,
        )
        in_ch = self.out_dim

        self.gat1 = GATConv(in_ch, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden * heads)

        d = hidden * heads
        self.cls = nn.Sequential(
            nn.Linear(d, 128), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_dense, x_high, x_low, edge_index):
        x = self.encode(x_dense, x_high, x_low)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm1(self.gat1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm2(self.gat2(h, edge_index)))
        return self.cls(h).view(-1)


class IBMGCNNet(_EmbeddingEncoder):
    """GCNConv-based fraud detector. Degree-normalized neighborhood aggregation."""

    def __init__(
        self,
        dense_in_dim: int,
        high_card_cols: List[str],
        low_card_cols: List[str],
        factor_cardinalities: Dict[str, int],
        hidden: int = 64,
        dropout: float = 0.30,
        high_card_emb_dim: int = 8,
        low_card_emb_dim: int = 4,
    ):
        super().__init__(
            dense_in_dim, high_card_cols, low_card_cols,
            factor_cardinalities, high_card_emb_dim, low_card_emb_dim, dropout,
        )
        in_ch = self.out_dim

        self.gcn1 = GCNConv(in_ch, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.cls = nn.Sequential(
            nn.Linear(hidden, 128), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_dense, x_high, x_low, edge_index):
        x = self.encode(x_dense, x_high, x_low)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm1(self.gcn1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm2(self.gcn2(h, edge_index)))
        return self.cls(h).view(-1)


class IBMSAGENet(_EmbeddingEncoder):
    """GraphSAGE fraud detector. Mean neighborhood aggregation by default."""

    def __init__(
        self,
        dense_in_dim: int,
        high_card_cols: List[str],
        low_card_cols: List[str],
        factor_cardinalities: Dict[str, int],
        hidden: int = 64,
        aggr: str = "mean",
        dropout: float = 0.30,
        high_card_emb_dim: int = 8,
        low_card_emb_dim: int = 4,
    ):
        super().__init__(
            dense_in_dim, high_card_cols, low_card_cols,
            factor_cardinalities, high_card_emb_dim, low_card_emb_dim, dropout,
        )
        in_ch = self.out_dim

        self.sage1 = SAGEConv(in_ch, hidden, aggr=aggr)
        self.sage2 = SAGEConv(hidden, hidden, aggr=aggr)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.cls = nn.Sequential(
            nn.Linear(hidden, 128), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_dense, x_high, x_low, edge_index):
        x = self.encode(x_dense, x_high, x_low)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm1(self.sage1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm2(self.sage2(h, edge_index)))
        return self.cls(h).view(-1)


class MLPBaseline(_EmbeddingEncoder):
    """Pure MLP — identical feature encoding, no graph. edge_index ignored."""

    def __init__(
        self,
        dense_in_dim: int,
        high_card_cols: List[str],
        low_card_cols: List[str],
        factor_cardinalities: Dict[str, int],
        hidden_dims: List[int] = None,
        dropout: float = 0.30,
        high_card_emb_dim: int = 8,
        low_card_emb_dim: int = 4,
    ):
        super().__init__(
            dense_in_dim, high_card_cols, low_card_cols,
            factor_cardinalities, high_card_emb_dim, low_card_emb_dim, dropout,
        )
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: List[nn.Module] = []
        prev = self.out_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_dense, x_high, x_low, edge_index=None):
        return self.net(self.encode(x_dense, x_high, x_low)).view(-1)


def build_model(
    model_arch: str,
    dense_in_dim: int,
    prep: FoldPreprocessor,
    cfg: IBMFraudConfig,
) -> nn.Module:
    """Factory for gatv2, gcn, sage, or mlp-style models."""
    common = dict(
        dense_in_dim=dense_in_dim,
        high_card_cols=prep.high_card_cols,
        low_card_cols=prep.low_card_cols,
        factor_cardinalities=prep.factor_cardinalities,
        dropout=cfg.DROPOUT,
        high_card_emb_dim=cfg.HIGH_CARD_EMB_DIM,
        low_card_emb_dim=cfg.LOW_CARD_EMB_DIM,
    )

    if model_arch == "gatv2":
        return IBMGATNet(**common, hidden=cfg.HIDDEN_DIM, heads=cfg.HEADS)
    if model_arch == "gcn":
        return IBMGCNNet(**common, hidden=cfg.GCN_HIDDEN_DIM)
    if model_arch == "sage":
        return IBMSAGENet(**common, hidden=cfg.SAGE_HIDDEN_DIM, aggr=cfg.SAGE_AGGR)

    raise ValueError(f"Unknown model_arch: '{model_arch}'. Choose 'gatv2', 'gcn', or 'sage'.")


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
            print("  [WARN] GPU OOM during inference — falling back to CPU")
            cleanup()
            model_cpu = model.cpu()
            logits = model_cpu(
                x_d.cpu(), x_h.cpu(), x_l.cpu(),
                edge_index.cpu() if edge_index is not None else None,
            )
            model.to(device)
            return torch.sigmoid(logits).numpy()
        raise


# ============================================================================
#  SHARED TRAINING LOOP
# ============================================================================
def _train_loop(
    model: nn.Module,
    x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr,
    x_d_va, x_h_va, x_l_va, y_va_np, edge_va,
    cfg: IBMFraudConfig,
    device: torch.device,
    verbose: bool = False,
):
    pos = max(1, int(y_tr_t.sum().item()))
    neg = max(1, int(len(y_tr_t) - pos))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-5)

    use_amp = cfg.USE_AMP and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    best_ap, best_state, no_improve = -1.0, None, 0

    def val_ap():
        probs = safe_inference(model, x_d_va, x_h_va, x_l_va, edge_va, device, use_amp)
        return average_precision_score(y_va_np, probs) if len(np.unique(y_va_np)) == 2 else 0.0

    for ep in range(1, cfg.MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(x_d_tr, x_h_tr, x_l_tr, edge_tr)
            loss = crit(logits, y_tr_t)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step(ep + 1e-8)

        if ep % cfg.EVAL_EVERY == 0:
            ap = val_ap()
            if ap > best_ap + 1e-6:
                best_ap, best_state, no_improve = ap, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
            if no_improve >= cfg.PATIENCE_CHECKS:
                if verbose:
                    print(f"  Early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================================
#  TRAIN ONE FOLD — GNN
# ============================================================================
def train_one_fold(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    cfg: IBMFraudConfig,
    graph_strategy: str,
    model_arch: str = "gatv2",
    fold_idx: int = 0,
    device: torch.device = None,
):
    """Trains one fold of a GNN: GAT, GCN, or GraphSAGE."""
    if device is None:
        device = get_device()

    df_tr, df_va, user_avg_map, global_avg = add_user_spending_features(
        df_tr, df_va, user_col="User", amount_col="Amount"
    )

    prep = FoldPreprocessor(cfg)
    prep.fit(df_tr)

    df_tr_raw, df_va_raw = df_tr.copy(), df_va.copy()
    X_dense_tr, X_high_tr, X_low_tr, y_tr_np = prep.transform(df_tr)
    X_dense_va, X_high_va, X_low_va, y_va_np = prep.transform(df_va)

    verbose = (fold_idx == 0)
    if verbose:
        print(f"  Building TRAIN graph ({graph_strategy})...")
    edge_tr = build_graph_edges(graph_strategy, df_tr_raw, X_dense_tr, X_high_tr, X_low_tr, cfg, verbose)
    if verbose:
        print(f"  Building VAL graph ({graph_strategy})...")
    edge_va = build_graph_edges(graph_strategy, df_va_raw, X_dense_va, X_high_va, X_low_va, cfg, verbose)

    del df_tr_raw, df_va_raw
    gc.collect()

    x_d_tr = torch.tensor(X_dense_tr, dtype=torch.float32).to(device)
    x_h_tr = torch.tensor(X_high_tr, dtype=torch.long).to(device)
    x_l_tr = torch.tensor(X_low_tr, dtype=torch.long).to(device)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float32).to(device)
    edge_tr = edge_tr.to(device)

    x_d_va = torch.tensor(X_dense_va, dtype=torch.float32).to(device)
    x_h_va = torch.tensor(X_high_va, dtype=torch.long).to(device)
    x_l_va = torch.tensor(X_low_va, dtype=torch.long).to(device)
    edge_va = edge_va.to(device)

    del X_dense_tr, X_high_tr, X_low_tr, X_dense_va, X_high_va, X_low_va
    gc.collect()

    model = build_model(model_arch, x_d_tr.shape[1], prep, cfg).to(device)
    use_amp = cfg.USE_AMP and device.type == "cuda"

    model = _train_loop(
        model, x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr,
        x_d_va, x_h_va, x_l_va, y_va_np, edge_va,
        cfg, device, verbose=verbose,
    )

    model.eval()
    p_tr = safe_inference(model, x_d_tr, x_h_tr, x_l_tr, edge_tr, device, use_amp)
    p_va = safe_inference(model, x_d_va, x_h_va, x_l_va, edge_va, device, use_amp)
    thr = choose_threshold(y_va_np, p_va, cfg)

    del x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr
    del x_d_va, x_h_va, x_l_va, edge_va
    cleanup()

    return {
        "model": model.cpu(), "preprocessor": prep, "thr": thr,
        "user_avg_map": user_avg_map, "global_avg": global_avg,
        "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
        "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr),
    }


# ============================================================================
#  TRAIN ONE FOLD — MLP BASELINE
# ============================================================================
def train_one_fold_mlp(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    cfg: IBMFraudConfig,
    fold_idx: int = 0,
    device: torch.device = None,
):
    if device is None:
        device = get_device()

    df_tr, df_va, user_avg_map, global_avg = add_user_spending_features(
        df_tr, df_va, user_col="User", amount_col="Amount"
    )

    prep = FoldPreprocessor(cfg)
    prep.fit(df_tr)

    X_dense_tr, X_high_tr, X_low_tr, y_tr_np = prep.transform(df_tr)
    X_dense_va, X_high_va, X_low_va, y_va_np = prep.transform(df_va)

    x_d_tr = torch.tensor(X_dense_tr, dtype=torch.float32).to(device)
    x_h_tr = torch.tensor(X_high_tr, dtype=torch.long).to(device)
    x_l_tr = torch.tensor(X_low_tr, dtype=torch.long).to(device)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float32).to(device)

    x_d_va = torch.tensor(X_dense_va, dtype=torch.float32).to(device)
    x_h_va = torch.tensor(X_high_va, dtype=torch.long).to(device)
    x_l_va = torch.tensor(X_low_va, dtype=torch.long).to(device)

    del X_dense_tr, X_high_tr, X_low_tr, X_dense_va, X_high_va, X_low_va
    gc.collect()

    model = MLPBaseline(
        dense_in_dim=x_d_tr.shape[1],
        high_card_cols=prep.high_card_cols,
        low_card_cols=prep.low_card_cols,
        factor_cardinalities=prep.factor_cardinalities,
        hidden_dims=cfg.MLP_HIDDEN_DIMS,
        dropout=cfg.DROPOUT,
        high_card_emb_dim=cfg.HIGH_CARD_EMB_DIM,
        low_card_emb_dim=cfg.LOW_CARD_EMB_DIM,
    ).to(device)

    use_amp = cfg.USE_AMP and device.type == "cuda"
    model = _train_loop(
        model, x_d_tr, x_h_tr, x_l_tr, y_tr_t, None,
        x_d_va, x_h_va, x_l_va, y_va_np, None,
        cfg, device, verbose=(fold_idx == 0),
    )

    model.eval()
    p_tr = safe_inference(model, x_d_tr, x_h_tr, x_l_tr, None, device, use_amp)
    p_va = safe_inference(model, x_d_va, x_h_va, x_l_va, None, device, use_amp)
    thr = choose_threshold(y_va_np, p_va, cfg)

    del x_d_tr, x_h_tr, x_l_tr, y_tr_t
    del x_d_va, x_h_va, x_l_va
    cleanup()

    return {
        "model": model.cpu(), "preprocessor": prep, "thr": thr,
        "user_avg_map": user_avg_map, "global_avg": global_avg,
        "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
        "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr),
    }


# ============================================================================
#  TEST ENSEMBLE
# ============================================================================
def _run_test_ensemble(
    all_out: List[dict],
    df_test: pd.DataFrame,
    cfg: IBMFraudConfig,
    graph_strategy: str,
    device: torch.device,
) -> tuple:
    test_probs = []
    is_mlp = (graph_strategy == "mlp_baseline")
    use_amp = cfg.USE_AMP and device.type == "cuda"

    for fold_i, out in enumerate(all_out):
        print(f"  Evaluating fold {fold_i + 1} on test set...")
        prep = out["preprocessor"]
        user_avg = out["user_avg_map"]
        g_avg = out["global_avg"]

        df_te = df_test.copy()
        df_te["user_avg_amount"] = df_te["User"].map(user_avg).fillna(g_avg).astype("float32")
        denom = df_te["user_avg_amount"].replace(0, 1.0)
        df_te["amount_over_user_avg"] = (df_te["Amount"] / denom).astype("float32")
        df_te["amount_minus_user_avg"] = (df_te["Amount"] - df_te["user_avg_amount"]).astype("float32")

        X_d_te, X_h_te, X_l_te, _ = prep.transform(df_te)

        if is_mlp:
            edge_te = None
        else:
            edge_te = build_graph_edges(graph_strategy, df_test, X_d_te, X_h_te, X_l_te, cfg).to(device)

        x_d = torch.tensor(X_d_te, dtype=torch.float32).to(device)
        x_h = torch.tensor(X_h_te, dtype=torch.long).to(device)
        x_l = torch.tensor(X_l_te, dtype=torch.long).to(device)

        model = out["model"].to(device)
        probs = safe_inference(model, x_d, x_h, x_l, edge_te, device, use_amp)
        test_probs.append(probs)
        out["model"] = model.cpu()

        del x_d, x_h, x_l
        if edge_te is not None:
            del edge_te
        del X_d_te, X_h_te, X_l_te
        cleanup()

    p_ens = np.mean(np.vstack(test_probs), axis=0)
    y_test = df_test[cfg.TARGET_COL].astype(int).to_numpy()
    thr_global = choose_threshold(
        np.concatenate([o["y_va"] for o in all_out]),
        np.concatenate([o["p_va"] for o in all_out]),
        cfg,
    )
    return p_ens, y_test, thr_global


# ============================================================================
#  MAIN PIPELINE — GNN
# ============================================================================
def run_pipeline(
    df: pd.DataFrame,
    cfg: IBMFraudConfig,
    graph_strategy: str = "multi_relation",
    model_arch: str = "gatv2",
):
    """Full pipeline for one graph strategy and one model architecture."""
    set_seed(cfg.SEED)
    device = get_device()

    print(f"Device: {device}")
    print(f"Graph strategy: {graph_strategy}  |  Model arch: {model_arch}")
    print(f"Dataset: {df.shape},  fraud rate: {df[cfg.TARGET_COL].mean():.4f}\n")

    df_dev, df_test = split_groups_holdout(
        df, key=cfg.GROUP_KEY, target_col=cfg.TARGET_COL,
        train_ratio=cfg.TRAIN_RATIO, bins=cfg.STRATIFY_BINS, seed=cfg.SEED,
    )
    print(f"DEV  {df_dev.shape}  fraud={df_dev[cfg.TARGET_COL].mean():.4f}")
    print(f"TEST {df_test.shape}  fraud={df_test[cfg.TARGET_COL].mean():.4f}")

    folds = build_group_stratified_folds(
        df_dev, group_key=cfg.GROUP_KEY, target_col=cfg.TARGET_COL,
        n_splits=cfg.N_SPLITS, bins=cfg.STRATIFY_BINS, seed=cfg.SEED,
    )

    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"\n{'='*20} Fold {i}/{cfg.N_SPLITS} {'='*20}")
        out = train_one_fold(
            df_tr, df_va, cfg, graph_strategy,
            model_arch=model_arch, fold_idx=i - 1, device=device,
        )
        print(f"  Threshold: {out['thr']:.4f}")
        print_metrics("TRAIN", out["m_tr"])
        print_metrics("VAL", out["m_va"])
        all_out.append(out)
        cleanup()

    def mean_of(ms, k):
        return float(np.mean([m[k] for m in ms]))

    tr_m = [o["m_tr"] for o in all_out]
    va_m = [o["m_va"] for o in all_out]
    print(f"\n{'='*20} CV SUMMARY {'='*20}")
    for tag, ms in [("TRAIN", tr_m), ("VAL", va_m)]:
        print(
            f"  {tag:>5}  "
            f"ACC {mean_of(ms,'acc'):.3f} | F1 {mean_of(ms,'f1'):.3f} | "
            f"P {mean_of(ms,'prec'):.3f} | R {mean_of(ms,'rec'):.3f} | "
            f"AUC {mean_of(ms,'auc'):.3f} | AP {mean_of(ms,'ap'):.3f} | "
            f"LL {mean_of(ms,'logloss'):.4f} | Brier {mean_of(ms,'brier'):.4f}"
        )

    p_ens, y_test, thr_global = _run_test_ensemble(all_out, df_test, cfg, graph_strategy, device)
    m_test = eval_from_probs(y_test, p_ens, thr_global)

    print(f"\n{'='*20} TEST (ENSEMBLED) {'='*20}")
    print(f"  Global threshold: {thr_global:.4f}")
    print_metrics("TEST", m_test)

    return {
        "all_out": all_out, "df_dev": df_dev, "df_test": df_test,
        "thr_global": thr_global, "test_metrics": m_test,
        "p_test_ens": p_ens, "y_test": y_test,
        "graph_strategy": graph_strategy, "model_arch": model_arch,
    }


# ============================================================================
#  MAIN PIPELINE — MLP BASELINE
# ============================================================================
def run_pipeline_mlp(df: pd.DataFrame, cfg: IBMFraudConfig):
    set_seed(cfg.SEED)
    device = get_device()

    print(f"\n{'#'*60}")
    print("# STRATEGY: mlp_baseline  (no graph — controlled ablation)")
    print(f"{'#'*60}\n")
    print(f"Device: {device}")
    print(f"Dataset: {df.shape},  fraud rate: {df[cfg.TARGET_COL].mean():.4f}\n")

    df_dev, df_test = split_groups_holdout(
        df, key=cfg.GROUP_KEY, target_col=cfg.TARGET_COL,
        train_ratio=cfg.TRAIN_RATIO, bins=cfg.STRATIFY_BINS, seed=cfg.SEED,
    )
    print(f"DEV  {df_dev.shape}  fraud={df_dev[cfg.TARGET_COL].mean():.4f}")
    print(f"TEST {df_test.shape}  fraud={df_test[cfg.TARGET_COL].mean():.4f}")

    folds = build_group_stratified_folds(
        df_dev, group_key=cfg.GROUP_KEY, target_col=cfg.TARGET_COL,
        n_splits=cfg.N_SPLITS, bins=cfg.STRATIFY_BINS, seed=cfg.SEED,
    )

    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"\n{'='*20} Fold {i}/{cfg.N_SPLITS} {'='*20}")
        out = train_one_fold_mlp(df_tr, df_va, cfg, fold_idx=i - 1, device=device)
        print(f"  Threshold: {out['thr']:.4f}")
        print_metrics("TRAIN", out["m_tr"])
        print_metrics("VAL", out["m_va"])
        all_out.append(out)
        cleanup()

    def mean_of(ms, k):
        return float(np.mean([m[k] for m in ms]))

    tr_m = [o["m_tr"] for o in all_out]
    va_m = [o["m_va"] for o in all_out]
    print(f"\n{'='*20} CV SUMMARY {'='*20}")
    for tag, ms in [("TRAIN", tr_m), ("VAL", va_m)]:
        print(
            f"  {tag:>5}  "
            f"ACC {mean_of(ms,'acc'):.3f} | F1 {mean_of(ms,'f1'):.3f} | "
            f"P {mean_of(ms,'prec'):.3f} | R {mean_of(ms,'rec'):.3f} | "
            f"AUC {mean_of(ms,'auc'):.3f} | AP {mean_of(ms,'ap'):.3f} | "
            f"LL {mean_of(ms,'logloss'):.4f} | Brier {mean_of(ms,'brier'):.4f}"
        )

    p_ens, y_test, thr_global = _run_test_ensemble(all_out, df_test, cfg, "mlp_baseline", device)
    m_test = eval_from_probs(y_test, p_ens, thr_global)

    print(f"\n{'='*20} TEST (ENSEMBLED) {'='*20}")
    print(f"  Global threshold: {thr_global:.4f}")
    print_metrics("TEST", m_test)

    return {
        "all_out": all_out, "df_dev": df_dev, "df_test": df_test,
        "thr_global": thr_global, "test_metrics": m_test,
        "p_test_ens": p_ens, "y_test": y_test,
        "graph_strategy": "mlp_baseline", "model_arch": "mlp",
    }


# ============================================================================
#  VISUALIZATION
# ============================================================================
def plot_results(result: dict):
    y = result["y_test"]
    p = result["p_test_ens"]
    m = result["test_metrics"]
    strat = result.get("graph_strategy", "")
    arch = result.get("model_arch", "")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"IBM Credit Card Fraud — {strat} / {arch}", fontsize=14)

    ConfusionMatrixDisplay(m["cm"]).plot(ax=axes[0, 0], cmap="Blues")
    axes[0, 0].set_title(f"Confusion Matrix (thr={result['thr_global']:.2f})")

    fpr, tpr, _ = roc_curve(y, p)
    axes[0, 1].plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "--", alpha=0.5)
    axes[0, 1].set(xlabel="FPR", ylabel="TPR", title="ROC")
    axes[0, 1].legend()

    pr, rc, _ = precision_recall_curve(y, p)
    axes[1, 0].plot(rc, pr)
    axes[1, 0].set(xlabel="Recall", ylabel="Precision", title="PR Curve")

    thrs = np.linspace(0, 1, 100)
    axes[1, 1].plot(thrs, [eval_from_probs(y, p, t)["f1"] for t in thrs], label="F1")
    axes[1, 1].plot(thrs, [eval_from_probs(y, p, t)["prec"] for t in thrs], label="Precision")
    axes[1, 1].plot(thrs, [eval_from_probs(y, p, t)["rec"] for t in thrs], label="Recall")
    axes[1, 1].set(xlabel="Threshold", ylabel="Score", title="Threshold Tuning")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def compare_strategies(results: Dict[str, dict]):
    metrics = ["f1", "prec", "rec", "auc", "ap"]
    labels = list(results.keys())

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(labels))

    for i, key in enumerate(labels):
        vals = [results[key]["test_metrics"][m] for m in metrics]
        ax.bar(x + i * w, vals, w, label=key)

    ax.set_xticks(x + w * (len(labels) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("IBM — Strategy × Architecture Comparison")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def print_final_comparison(results: Dict[str, dict]):
    print("\n" + "=" * 88)
    print("FINAL COMPARISON — IBM CREDIT CARD FRAUD")
    print(f"{'Run key':<32} | {'F1':>5} | {'P':>5} | {'R':>5} | {'AUC':>5} | {'AP':>5} | {'Thr':>5}")
    print("-" * 88)
    for key, res in results.items():
        m = res["test_metrics"]
        thr = res["thr_global"]
        print(
            f"  {key:<30s} | "
            f"{m['f1']:.3f} | {m['prec']:.3f} | {m['rec']:.3f} | "
            f"{m['auc']:.3f} | {m['ap']:.3f} | {thr:.3f}"
        )
    print("=" * 88)

    gnn_strats = ["multi_relation", "hybrid", "intra_group"]
    archs = ["gatv2", "gcn", "sage"]

    print("\n--- Topology ranking consistency across architectures ---")
    rankings = {}
    for arch in archs:
        arch_f1 = {
            s: results[f"{s}_{arch}"]["test_metrics"]["f1"]
            for s in gnn_strats if f"{s}_{arch}" in results
        }
        if not arch_f1:
            continue
        ranked = sorted(arch_f1.items(), key=lambda x: -x[1])
        rankings[arch] = [s for s, _ in ranked]
        print(f"  {arch.upper():8s}  " + "  >  ".join(f"{s}({v:.3f})" for s, v in ranked))

    if len(rankings) >= 2:
        base_arch = next(iter(rankings))
        consistent = all(rank == rankings[base_arch] for rank in rankings.values())
        print(f"\n  Ranking consistent across available GNN architectures: {'YES' if consistent else 'NO'}")
        if consistent:
            print("  -> Topology ranking appears architecture-independent on this dataset.")
        else:
            print("  -> Rankings differ; discuss graph-topology x architecture interaction.")

    if "intra_group_gatv2" in results and "mlp_baseline" in results:
        ig_f1 = results["intra_group_gatv2"]["test_metrics"]["f1"]
        mlp_f1 = results["mlp_baseline"]["test_metrics"]["f1"]
        gap = ig_f1 - mlp_f1
        print(f"\n  [Ablation] intra_group (GAT) F1={ig_f1:.3f} vs MLP F1={mlp_f1:.3f} (gap={gap:+.3f})")


# ============================================================================
#  RUNNERS
# ============================================================================
def run_all_strategies(df: pd.DataFrame, cfg: IBMFraudConfig = None):
    """Runs 3 graph strategies x 3 architectures + MLP baseline."""
    if cfg is None:
        cfg = IBMFraudConfig()

    results = {}
    for strat in ["multi_relation", "hybrid", "intra_group"]:
        for arch in ["gatv2", "gcn", "sage"]:
            key = f"{strat}_{arch}"
            print(f"\n{'#'*60}")
            print(f"# STRATEGY: {strat}  |  ARCH: {arch}")
            print(f"{'#'*60}\n")
            results[key] = run_pipeline(df, cfg, graph_strategy=strat, model_arch=arch)
            plot_results(results[key])
            cleanup()

    results["mlp_baseline"] = run_pipeline_mlp(df, cfg)
    plot_results(results["mlp_baseline"])

    compare_strategies(results)
    print_final_comparison(results)
    return results


def run_sage_comparison(df: pd.DataFrame, cfg: IBMFraudConfig = None):
    """GraphSAGE arm only — all 3 graph strategies under GraphSAGE."""
    if cfg is None:
        cfg = IBMFraudConfig()

    results = {}
    for strat in ["multi_relation", "hybrid", "intra_group"]:
        key = f"{strat}_sage"
        print(f"\n{'#'*60}")
        print(f"# GRAPHSAGE — STRATEGY: {strat}")
        print(f"{'#'*60}\n")
        results[key] = run_pipeline(df, cfg, graph_strategy=strat, model_arch="sage")
        plot_results(results[key])
        cleanup()

    print("\n--- GraphSAGE topology ranking (IBM) ---")
    ranked = sorted(
        {s: results[f"{s}_sage"]["test_metrics"]["f1"]
         for s in ["multi_relation", "hybrid", "intra_group"]}.items(),
        key=lambda x: -x[1],
    )
    for s, v in ranked:
        print(f"  {s}: F1={v:.3f}")
    return results


def run_gcn_comparison(df: pd.DataFrame, cfg: IBMFraudConfig = None):
    """GCN arm only — all 3 graph strategies under GCN."""
    if cfg is None:
        cfg = IBMFraudConfig()

    results = {}
    for strat in ["multi_relation", "hybrid", "intra_group"]:
        key = f"{strat}_gcn"
        print(f"\n{'#'*60}")
        print(f"# GCN — STRATEGY: {strat}")
        print(f"{'#'*60}\n")
        results[key] = run_pipeline(df, cfg, graph_strategy=strat, model_arch="gcn")
        plot_results(results[key])
        cleanup()

    print("\n--- GCN topology ranking (IBM) ---")
    ranked = sorted(
        {s: results[f"{s}_gcn"]["test_metrics"]["f1"]
         for s in ["multi_relation", "hybrid", "intra_group"]}.items(),
        key=lambda x: -x[1],
    )
    for s, v in ranked:
        print(f"  {s}: F1={v:.3f}")
    return results


# ============================================================================
#  MAIN
# ============================================================================
if __name__ == "__main__":
    cfg = IBMFraudConfig()
    df = load_and_preprocess("data/ibm_transactions.parquet", cfg)

    # Option A — full matrix: 3 graph strategies x 3 architectures + MLP
    results = run_all_strategies(df, cfg)

    # Option B — GraphSAGE arm only
    # sage_results = run_sage_comparison(df, cfg)

    # Option C — GCN arm only
    # gcn_results = run_gcn_comparison(df, cfg)

    # Option D — single run
    # result = run_pipeline(df, cfg, graph_strategy="multi_relation", model_arch="sage")
    # plot_results(result)
