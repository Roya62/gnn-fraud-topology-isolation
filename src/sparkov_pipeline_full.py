##############################################################################
# CARD TRANSACTION FRAUD — STANDALONE GAT + GCN + GRAPHSAGE PIPELINE
# WITH MLP BASELINE FOR CONTROLLED ABLATION
#
# Dataset: fraudTrain.csv + fraudTest.csv (Kaggle-style credit card fraud)
#
# Graph strategies (pass via `graph_strategy`):
#   "multi_relation"  – temporal edges along cc_num, merchant, category, zip
#   "hybrid"          – multi_relation + FAISS k-NN feature-similarity edges
#   "intra_group"     – intra-cc_num temporal + similarity + merchant chains
#   "mlp_baseline"    – no graph; identical features, loss, splits, threshold
#
# Model architectures (pass via `model_arch`):
#   "gatv2"           – GATv2Conv (2 layers, 4 heads, LayerNorm)  [original]
#   "gcn"             – GCNConv   (2 layers, LayerNorm)
#   "sage"            – SAGEConv  (2 layers, mean aggregation, LayerNorm)
#
# Running both architectures over the same graph strategies isolates whether
# topology rankings (multi_relation > intra_group > hybrid) are preserved
# across architectures — directly addressing the reviewer concern that
# "topology rankings are findings about GATv2 + topology X, not topology alone."
#
# Fixed training (both architectures):
#   • BCEWithLogitsLoss with auto pos_weight
#   • AdamW + CosineAnnealingWarmRestarts
#   • Group-stratified 5-fold CV by cc_num
#   • Threshold tuning (max-F1 on pooled validation)
#   • 5-fold ensemble on held-out test
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

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
class CardFraudConfig:
    """All tuneable knobs for the card-transaction pipeline."""

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

    # --- model (shared) ---
    HIDDEN_DIM: int = 64
    DROPOUT: float = 0.30
    EMBEDDING_DIM: int = 8

    # --- model (GATv2-specific) ---
    HEADS: int = 4

    # --- model (GCN-specific) ---
    # GCN uses HIDDEN_DIM directly (no multi-head); two layers of HIDDEN_DIM each.
    # Output dim = HIDDEN_DIM (no concatenation unlike GATv2).
    GCN_HIDDEN_DIM: int = 256   # wider to roughly match GATv2 capacity (64*4=256)

    # --- model (GraphSAGE-specific) ---
    SAGE_HIDDEN_DIM: int = 256   # match GCN width / GATv2 effective output dim

    # --- model (MLP baseline) ---
    MLP_HIDDEN_DIMS: List[int] = [256, 128]

    # --- threshold policy ---
    THRESHOLD_OVERRIDE: Optional[float] = None
    USE_COST_THRESHOLD: bool = False
    COST_FN: float = 100.0
    COST_FP: float = 1.0
    TARGET_RECALL: Optional[float] = None

    # --- dataset schema ---
    TARGET_COL: str = "is_fraud"
    GROUP_KEY: str = "cc_num"

    CATEGORICAL_COLS: List[str] = [
        "merchant", "category", "state", "gender",
        "city", "zip", "job",
    ]
    NUMERICAL_COLS: List[str] = [
        "amt", "age", "city_pop",
        "lat", "long", "merch_lat", "merch_long",
        "hour", "weekday", "trans_month",
        "sin_hour", "cos_hour",
        "distance", "amt_to_avg", "business_hours",
        "trans_time_diff",
    ]

    # --- temporal sort key ---
    SORT_KEY_COLS: Dict[str, int] = {"timestamp": 1}

    # --- graph: multi-relation temporal ---
    MULTI_REL_SPECS: List[Dict[str, Any]] = [
        {"col": "cc_num",    "k": 2, "max_group_size": 2000},
        {"col": "merchant",  "k": 2, "max_group_size": 1000},
        {"col": "category",  "k": 1, "max_group_size": 3000},
        {"col": "zip",       "k": 1, "max_group_size": 2000},
    ]
    ADD_GLOBAL_TIME_EDGES: bool = True
    GLOBAL_TIME_K: int = 2
    SELF_LOOPS: bool = True

    # --- graph: hybrid (adds FAISS) ---
    FAISS_K: int = 6
    FAISS_HNSW_M: int = 32
    FAISS_EF_SEARCH: int = 64

    # --- graph: intra-group ---
    INTRA_GROUP_KEY: str = "cc_num"
    INTRA_MAX_GROUP_SIZE: int = 200
    INTRA_K_TEMPORAL: int = 3
    INTRA_K_SIMILAR: int = 5
    INTRA_SIM_THRESHOLD: float = 0.5
    INTRA_SUB_RELATION_COLS: List[str] = ["merchant"]

    # --- downsampling ---
    DOWNSAMPLE_NONFRAUD: int = 1_800_000


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
#  DATA LOADING & PREPROCESSING
# ============================================================================
def load_and_preprocess(
    file1: str = "fraudTrain.csv",
    file2: str = "fraudTest.csv",
    cfg: CardFraudConfig = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = CardFraudConfig()

    df1 = pd.read_csv('/content/fraudTest.csv')
    df2 = pd.read_csv('/content/fraudTrain.csv')
    df = pd.concat([df1, df2], ignore_index=True).dropna()
    del df1, df2
    gc.collect()
    print(f"Combined dataset: {len(df):,} transactions")

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = ((df["trans_date_trans_time"] - df["dob"]).dt.days // 365).astype("int16")

    df["hour"]        = df["trans_date_trans_time"].dt.hour.astype("int8")
    df["weekday"]     = df["trans_date_trans_time"].dt.weekday.astype("int8")
    df["trans_month"] = df["trans_date_trans_time"].dt.month.astype("int8")
    df["is_fraud"]    = df["is_fraud"].astype("int8")

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype("int8")

    nonfraud_idx = df.index[df["is_fraud"] == 0]
    n_drop = min(len(nonfraud_idx), cfg.DOWNSAMPLE_NONFRAUD)
    drop_idx = np.random.RandomState(cfg.SEED).choice(nonfraud_idx, size=n_drop, replace=False)
    df = df.drop(drop_idx).reset_index(drop=True)
    print(
        f"After downsampling: {len(df):,} transactions  "
        f"(fraud rate: {df['is_fraud'].mean():.4f})"
    )

    cols_to_drop = ["Unnamed: 0", "first", "last", "street", "trans_num", "dob"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    df["timestamp"] = np.arange(len(df), dtype=np.int64)

    df["trans_time_diff"] = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .diff().dt.total_seconds().div(60).fillna(0).astype("float32")
    )

    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        df["distance"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2
            + (df["long"] - df["merch_long"]) ** 2
        ).astype("float32")

    if "cc_num" in df.columns and "amt" in df.columns:
        cc_avg = df.groupby("cc_num")["amt"].transform("mean")
        df["amt_to_avg"] = (df["amt"] / (cc_avg + 1e-10)).astype("float32")

    df["cc_num"] = pd.to_numeric(df["cc_num"], errors="coerce").fillna(-1)

    if "trans_date_trans_time" in df.columns:
        df = df.drop(columns=["trans_date_trans_time"])

    print(f"Final shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
    gc.collect()
    return df


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
        yh   = (y_prob >= thr).astype(int)
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


def choose_threshold(y_true, y_prob, cfg: CardFraudConfig):
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
        acc =(y_pred == y_true).mean(),
        f1  =f1_score(y_true, y_pred, zero_division=0),
        prec=precision_score(y_true, y_pred, zero_division=0),
        rec =recall_score(y_true, y_pred, zero_division=0),
        auc =roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.5,
        ap  =average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.0,
        cm  =confusion_matrix(y_true, y_pred),
        logloss=log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
        brier  =brier_score_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
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
#  FOLD PREPROCESSOR
# ============================================================================
class FoldPreprocessor:
    def __init__(self, cat_cols: List[str], num_cols: List[str]):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.cardinalities: Dict[str, int] = {}

    def fit(self, df_train: pd.DataFrame):
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
            le    = self.label_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x, _k=known, _le=le: (
                    _le.transform([x])[0] if x in _k else len(_le.classes_)
                )
            )
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df


# ============================================================================
#  DATA SPLITTING
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


def build_stratified_folds(df_dev, target_col, n_splits=5, group_key=None, bins=10, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if group_key is not None and group_key in df_dev.columns:
        grp = df_dev.groupby(group_key)[target_col].mean().rename("prev").reset_index()
        try:
            grp["bin"] = pd.qcut(grp["prev"], q=bins, labels=False, duplicates="drop")
        except ValueError:
            grp["bin"] = 0
        groups = grp[group_key].values
        ybins  = grp["bin"].fillna(0).astype(int).values

        folds = []
        for tr_idx, va_idx in skf.split(groups, ybins):
            g_tr, g_va = set(groups[tr_idx]), set(groups[va_idx])
            folds.append((
                df_dev[df_dev[group_key].isin(g_tr)].reset_index(drop=True),
                df_dev[df_dev[group_key].isin(g_va)].reset_index(drop=True),
            ))
    else:
        y_all = df_dev[target_col].values
        folds = []
        for tr_idx, va_idx in skf.split(df_dev, y_all):
            folds.append((
                df_dev.iloc[tr_idx].reset_index(drop=True),
                df_dev.iloc[va_idx].reset_index(drop=True),
            ))
    return folds


# ============================================================================
#  GRAPH BUILDERS  (unchanged from original)
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


def _build_multi_relation_edges(df_raw, cfg: CardFraudConfig) -> set:
    edge_set = set()
    sort_key = _make_sort_key(df_raw, cfg.SORT_KEY_COLS)
    n        = len(df_raw)
    node_ids = np.arange(n, dtype=np.int64)

    for spec in (cfg.MULTI_REL_SPECS or []):
        col    = spec["col"]
        k      = spec.get("k", 1)
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
        ends   = np.concatenate([breaks, [len(rels)]])

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


def _build_faiss_edges(cat_np, num_np, cfg: CardFraudConfig) -> set:
    if not FAISS_AVAILABLE:
        raise ImportError("faiss required for 'hybrid' strategy. pip install faiss-cpu")
    features = np.ascontiguousarray(np.hstack([
        cat_np.astype("float32"), num_np.astype("float32"),
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


def _build_hybrid_edges(df_raw, cat_np, num_np, cfg: CardFraudConfig) -> set:
    edge_set = _build_multi_relation_edges(df_raw, cfg)
    edge_set.update(_build_faiss_edges(cat_np, num_np, cfg))
    return edge_set


def _build_intra_group_edges(df_raw, cat_np, num_np, cfg: CardFraudConfig) -> set:
    edge_set  = set()
    group_col = cfg.INTRA_GROUP_KEY
    if group_col not in df_raw.columns:
        raise ValueError(f"Intra-group key '{group_col}' not in dataframe")

    sort_key     = _make_sort_key(df_raw, cfg.SORT_KEY_COLS)
    all_features = np.hstack([cat_np.astype("float32"), num_np.astype("float32")])

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
                edge_set.add((a, b)); edge_set.add((b, a))

        if SKLEARN_COSINE_AVAILABLE and len(indices) > cfg.INTRA_K_SIMILAR:
            sim = cosine_similarity(all_features[indices])
            for i in range(len(indices)):
                scores = sim[i].copy(); scores[i] = -1.0
                for j in np.argsort(scores)[-cfg.INTRA_K_SIMILAR:]:
                    if scores[j] > cfg.INTRA_SIM_THRESHOLD:
                        a, b = indices[i], indices[j]
                        edge_set.add((a, b)); edge_set.add((b, a))

        for sub_col in (cfg.INTRA_SUB_RELATION_COLS or []):
            if sub_col not in df_raw.columns:
                continue
            for sub_idx in df_raw.iloc[indices].groupby(sub_col).groups.values():
                sub_list = sorted(list(sub_idx), key=lambda i: sort_key[i])
                for i in range(len(sub_list) - 1):
                    a, b = sub_list[i], sub_list[i + 1]
                    edge_set.add((a, b)); edge_set.add((b, a))

    return edge_set


def build_graph_edges(
    strategy: str,
    df_raw: pd.DataFrame,
    cat_np: np.ndarray,
    num_np: np.ndarray,
    cfg: CardFraudConfig,
    verbose: bool = False,
) -> torch.Tensor:
    n = len(df_raw)

    if strategy == "multi_relation":
        edge_set = _build_multi_relation_edges(df_raw, cfg)
    elif strategy == "hybrid":
        edge_set = _build_hybrid_edges(df_raw, cat_np, num_np, cfg)
    elif strategy == "intra_group":
        edge_set = _build_intra_group_edges(df_raw, cat_np, num_np, cfg)
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
    return edge_index


# ============================================================================
#  MODELS
# ============================================================================
class UnifiedGATNet(nn.Module):
    """
    GATv2 node classifier (original architecture, unchanged).
    Two GATv2Conv layers, 4 heads each → output dim = HIDDEN_DIM * HEADS = 256.
    """

    def __init__(
        self,
        cardinalities: Dict[str, int],
        cat_cols: List[str],
        num_input_dim: int,
        embedding_dim: int = 8,
        hidden: int = 64,
        heads: int = 4,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.dropout  = dropout

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinalities[col] + 1, embedding_dim)
            for col in cat_cols
        })

        in_channels = len(cat_cols) * embedding_dim + num_input_dim

        self.gat1  = GATv2Conv(in_channels,  hidden, heads=heads, dropout=dropout)
        self.gat2  = GATv2Conv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden * heads)

        d = hidden * heads
        self.cls = nn.Sequential(
            nn.Linear(d, 128), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_cat_dict, x_num, edge_index):
        cat_embs = [self.embeddings[col](x_cat_dict[col]) for col in self.cat_cols]
        x = torch.cat([torch.cat(cat_embs, dim=1), x_num], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        h = F.leaky_relu(self.norm1(self.gat1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm2(self.gat2(h, edge_index)))

        return self.cls(h).view(-1)


# ----------------------------------------------------------------------------
#  NEW: GCN model
# ----------------------------------------------------------------------------
class UnifiedGCNNet(nn.Module):
    """
    GCN node classifier — identical feature encoding to UnifiedGATNet,
    GCNConv replaces GATv2Conv.

    GCNConv uses symmetric normalisation (Kipf & Welling, 2017) with no
    attention mechanism, making it a structurally simpler aggregator than
    GATv2. Comparing topology rankings under both architectures tests whether
    the reported rankings (multi_relation > intra_group > hybrid) are a
    property of the graph topology or an artefact of GATv2's attention.

    Architecture:
        embeddings + numerics
            → GCNConv(in, GCN_HIDDEN_DIM) → LayerNorm → LeakyReLU → Dropout
            → GCNConv(GCN_HIDDEN_DIM, GCN_HIDDEN_DIM) → LayerNorm → LeakyReLU
            → MLP(GCN_HIDDEN_DIM → 128 → 1)

    GCN_HIDDEN_DIM defaults to 256 to roughly match GATv2's effective output
    dimension (hidden=64, heads=4 → 256-d node representations), keeping
    model capacity comparable across the two architectures.

    API: identical to UnifiedGATNet — accepts x_cat_dict, x_num, edge_index.
    """

    def __init__(
        self,
        cardinalities: Dict[str, int],
        cat_cols: List[str],
        num_input_dim: int,
        embedding_dim: int = 8,
        hidden: int = 256,        # GCN_HIDDEN_DIM
        dropout: float = 0.30,
    ):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.dropout  = dropout

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinalities[col] + 1, embedding_dim)
            for col in cat_cols
        })

        in_channels = len(cat_cols) * embedding_dim + num_input_dim

        self.gcn1  = GCNConv(in_channels, hidden)
        self.gcn2  = GCNConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.cls = nn.Sequential(
            nn.Linear(hidden, 128), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_cat_dict, x_num, edge_index):
        cat_embs = [self.embeddings[col](x_cat_dict[col]) for col in self.cat_cols]
        x = torch.cat([torch.cat(cat_embs, dim=1), x_num], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        h = F.leaky_relu(self.norm1(self.gcn1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm2(self.gcn2(h, edge_index)))

        return self.cls(h).view(-1)


class UnifiedSAGENet(nn.Module):
    """
    GraphSAGE node classifier — identical feature encoding to UnifiedGATNet
    and UnifiedGCNNet, but using SAGEConv mean aggregation.
    """

    def __init__(
        self,
        cardinalities: Dict[str, int],
        cat_cols: List[str],
        num_input_dim: int,
        embedding_dim: int = 8,
        hidden: int = 256,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.dropout  = dropout

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinalities[col] + 1, embedding_dim)
            for col in cat_cols
        })

        in_channels = len(cat_cols) * embedding_dim + num_input_dim

        self.sage1 = SAGEConv(in_channels, hidden, aggr="mean")
        self.sage2 = SAGEConv(hidden, hidden, aggr="mean")
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.cls = nn.Sequential(
            nn.Linear(hidden, 128), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_cat_dict, x_num, edge_index):
        cat_embs = [self.embeddings[col](x_cat_dict[col]) for col in self.cat_cols]
        x = torch.cat([torch.cat(cat_embs, dim=1), x_num], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        h = F.leaky_relu(self.norm1(self.sage1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm2(self.sage2(h, edge_index)))

        return self.cls(h).view(-1)


class MLPBaseline(nn.Module):
    """Pure MLP — identical feature encoding, no graph."""

    def __init__(
        self,
        cardinalities: Dict[str, int],
        cat_cols: List[str],
        num_input_dim: int,
        embedding_dim: int = 8,
        hidden_dims: List[int] = None,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.dropout  = dropout

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinalities[col] + 1, embedding_dim)
            for col in cat_cols
        })

        in_dim = len(cat_cols) * embedding_dim + num_input_dim
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cat_dict, x_num, edge_index=None):
        cat_embs = [self.embeddings[col](x_cat_dict[col]) for col in self.cat_cols]
        x = torch.cat([torch.cat(cat_embs, dim=1), x_num], dim=1)
        return self.net(x).view(-1)


def build_model(
    model_arch: str,
    cardinalities: Dict[str, int],
    cat_cols: List[str],
    num_input_dim: int,
    cfg: CardFraudConfig,
) -> nn.Module:
    """
    Factory that returns the correct model for a given architecture string.
    Keeps train_one_fold / train_one_fold_mlp architecture-agnostic.

    model_arch options:
        "gatv2"        → UnifiedGATNet   (original)
        "gcn"          → UnifiedGCNNet   (GCN ablation)
        "sage"         → UnifiedSAGENet  (GraphSAGE ablation)
    """
    if model_arch == "gatv2":
        return UnifiedGATNet(
            cardinalities=cardinalities,
            cat_cols=cat_cols,
            num_input_dim=num_input_dim,
            embedding_dim=cfg.EMBEDDING_DIM,
            hidden=cfg.HIDDEN_DIM,
            heads=cfg.HEADS,
            dropout=cfg.DROPOUT,
        )
    elif model_arch == "gcn":
        return UnifiedGCNNet(
            cardinalities=cardinalities,
            cat_cols=cat_cols,
            num_input_dim=num_input_dim,
            embedding_dim=cfg.EMBEDDING_DIM,
            hidden=cfg.GCN_HIDDEN_DIM,
            dropout=cfg.DROPOUT,
        )
    elif model_arch == "sage":
        return UnifiedSAGENet(
            cardinalities=cardinalities,
            cat_cols=cat_cols,
            num_input_dim=num_input_dim,
            embedding_dim=cfg.EMBEDDING_DIM,
            hidden=cfg.SAGE_HIDDEN_DIM,
            dropout=cfg.DROPOUT,
        )
    else:
        raise ValueError(
            f"Unknown model_arch: '{model_arch}'. Choose 'gatv2', 'gcn', or 'sage'."
        )


# ============================================================================
#  SHARED TRAINING LOOP  (unchanged)
# ============================================================================
def _train_loop(
    model: nn.Module,
    cat_tr_d, num_tr_t, y_tr_t, edge_tr,
    cat_va_d, num_va_t, y_va_np, edge_va,
    cfg: CardFraudConfig,
    verbose: bool = False,
):
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
        model.train()
        opt.zero_grad()
        loss = crit(model(cat_tr_d, num_tr_t, edge_tr), y_tr_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(ep + 1e-8)

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
#  TRAIN ONE FOLD — GNN  (now accepts model_arch parameter)
# ============================================================================
def train_one_fold(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    cfg: CardFraudConfig,
    graph_strategy: str,
    model_arch: str = "gatv2",     # <-- NEW parameter
    fold_idx: int = 0,
    device: torch.device = None,
):
    """
    Trains one fold of a GNN (GATv2 or GCN) with the given graph strategy.
    model_arch = "gatv2" reproduces original behaviour exactly.
    model_arch = "gcn"   uses UnifiedGCNNet on the same graph.
    """
    if device is None:
        device = get_device()

    prep = FoldPreprocessor(cfg.CATEGORICAL_COLS, cfg.NUMERICAL_COLS)
    prep.fit(df_tr)

    df_tr_raw, df_va_raw = df_tr.copy(), df_va.copy()
    df_tr_enc = prep.transform(df_tr)
    df_va_enc = prep.transform(df_va)

    cat_tr  = df_tr_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
    num_tr  = df_tr_enc[prep.num_cols].to_numpy(dtype=np.float32)
    y_tr_np = df_tr_enc[cfg.TARGET_COL].to_numpy(dtype=np.int64)

    cat_va  = df_va_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
    num_va  = df_va_enc[prep.num_cols].to_numpy(dtype=np.float32)
    y_va_np = df_va_enc[cfg.TARGET_COL].to_numpy(dtype=np.int64)

    verbose = fold_idx == 0
    if verbose:
        print(f"  Building TRAIN graph ({graph_strategy})...")
    edge_tr = build_graph_edges(graph_strategy, df_tr_raw, cat_tr, num_tr, cfg, verbose).to(device)
    if verbose:
        print(f"  Building VAL graph ({graph_strategy})...")
    edge_va = build_graph_edges(graph_strategy, df_va_raw, cat_va, num_va, cfg, verbose).to(device)

    cat_tr_d = {col: torch.tensor(cat_tr[:, i], dtype=torch.long).to(device)
                for i, col in enumerate(cfg.CATEGORICAL_COLS)}
    num_tr_t = torch.tensor(num_tr, dtype=torch.float).to(device)
    y_tr_t   = torch.tensor(y_tr_np, dtype=torch.float).to(device)

    cat_va_d = {col: torch.tensor(cat_va[:, i], dtype=torch.long).to(device)
                for i, col in enumerate(cfg.CATEGORICAL_COLS)}
    num_va_t = torch.tensor(num_va, dtype=torch.float).to(device)

    # Build correct architecture via factory
    model = build_model(
        model_arch, prep.cardinalities, cfg.CATEGORICAL_COLS,
        num_tr.shape[1], cfg,
    ).to(device)

    model = _train_loop(
        model,
        cat_tr_d, num_tr_t, y_tr_t, edge_tr,
        cat_va_d, num_va_t, y_va_np, edge_va,
        cfg, verbose=verbose,
    )

    model.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(model(cat_tr_d, num_tr_t, edge_tr)).cpu().numpy()
        p_va = torch.sigmoid(model(cat_va_d, num_va_t, edge_va)).cpu().numpy()

    thr = choose_threshold(y_va_np, p_va, cfg)
    cleanup()

    return {
        "model": model, "preprocessor": prep, "thr": thr,
        "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
        "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr),
    }


# ============================================================================
#  TRAIN ONE FOLD — MLP BASELINE  (unchanged)
# ============================================================================
def train_one_fold_mlp(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    cfg: CardFraudConfig,
    fold_idx: int = 0,
    device: torch.device = None,
):
    if device is None:
        device = get_device()

    prep = FoldPreprocessor(cfg.CATEGORICAL_COLS, cfg.NUMERICAL_COLS)
    prep.fit(df_tr)

    df_tr_enc = prep.transform(df_tr)
    df_va_enc = prep.transform(df_va)

    cat_tr  = df_tr_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
    num_tr  = df_tr_enc[prep.num_cols].to_numpy(dtype=np.float32)
    y_tr_np = df_tr_enc[cfg.TARGET_COL].to_numpy(dtype=np.int64)

    cat_va  = df_va_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
    num_va  = df_va_enc[prep.num_cols].to_numpy(dtype=np.float32)
    y_va_np = df_va_enc[cfg.TARGET_COL].to_numpy(dtype=np.int64)

    cat_tr_d = {col: torch.tensor(cat_tr[:, i], dtype=torch.long).to(device)
                for i, col in enumerate(cfg.CATEGORICAL_COLS)}
    num_tr_t = torch.tensor(num_tr, dtype=torch.float).to(device)
    y_tr_t   = torch.tensor(y_tr_np, dtype=torch.float).to(device)

    cat_va_d = {col: torch.tensor(cat_va[:, i], dtype=torch.long).to(device)
                for i, col in enumerate(cfg.CATEGORICAL_COLS)}
    num_va_t = torch.tensor(num_va, dtype=torch.float).to(device)

    model = MLPBaseline(
        cardinalities=prep.cardinalities,
        cat_cols=cfg.CATEGORICAL_COLS,
        num_input_dim=num_tr.shape[1],
        embedding_dim=cfg.EMBEDDING_DIM,
        hidden_dims=cfg.MLP_HIDDEN_DIMS,
        dropout=cfg.DROPOUT,
    ).to(device)

    model = _train_loop(
        model,
        cat_tr_d, num_tr_t, y_tr_t, None,
        cat_va_d, num_va_t, y_va_np, None,
        cfg, verbose=(fold_idx == 0),
    )

    model.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(model(cat_tr_d, num_tr_t, None)).cpu().numpy()
        p_va = torch.sigmoid(model(cat_va_d, num_va_t, None)).cpu().numpy()

    thr = choose_threshold(y_va_np, p_va, cfg)
    cleanup()

    return {
        "model": model, "preprocessor": prep, "thr": thr,
        "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
        "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr),
    }


# ============================================================================
#  SHARED TEST ENSEMBLE HELPER  (now passes model_arch through)
# ============================================================================
def _run_test_ensemble(
    all_out: List[dict],
    df_test: pd.DataFrame,
    cfg: CardFraudConfig,
    graph_strategy: str,
    device: torch.device,
) -> tuple:
    test_probs = []
    is_mlp     = (graph_strategy == "mlp_baseline")

    for fold_i, out in enumerate(all_out):
        print(f"  Evaluating fold {fold_i + 1} on test set...")
        prep      = out["preprocessor"]
        df_te_enc = prep.transform(df_test)
        cat_te    = df_te_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
        num_te    = df_te_enc[prep.num_cols].to_numpy(dtype=np.float32)

        if is_mlp:
            edge_te = None
        else:
            edge_te = build_graph_edges(
                graph_strategy, df_test, cat_te, num_te, cfg
            ).to(device)

        cat_te_d = {col: torch.tensor(cat_te[:, i], dtype=torch.long).to(device)
                    for i, col in enumerate(cfg.CATEGORICAL_COLS)}
        num_te_t = torch.tensor(num_te, dtype=torch.float).to(device)

        out["model"].eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                out["model"](cat_te_d, num_te_t, edge_te)
            ).cpu().numpy()
        test_probs.append(probs)

        if edge_te is not None:
            del edge_te
        del cat_te_d, num_te_t
        cleanup()

    p_ens      = np.mean(np.vstack(test_probs), axis=0)
    y_test     = df_test[cfg.TARGET_COL].astype(int).to_numpy()
    thr_global = choose_threshold(
        np.concatenate([o["y_va"] for o in all_out]),
        np.concatenate([o["p_va"] for o in all_out]),
        cfg,
    )
    return p_ens, y_test, thr_global


# ============================================================================
#  MAIN PIPELINE — GNN  (now accepts model_arch parameter)
# ============================================================================
def run_pipeline(
    df: pd.DataFrame,
    cfg: CardFraudConfig,
    graph_strategy: str = "multi_relation",
    model_arch: str = "gatv2",     # <-- NEW parameter
):
    """
    Full pipeline for one (graph_strategy, model_arch) combination.
    Pass model_arch="gcn" to run the GCN variant on any graph strategy.
    """
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

    folds = build_stratified_folds(
        df_dev, target_col=cfg.TARGET_COL, n_splits=cfg.N_SPLITS,
        group_key=cfg.GROUP_KEY, bins=cfg.STRATIFY_BINS, seed=cfg.SEED,
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
        print_metrics("VAL",   out["m_va"])
        all_out.append(out); cleanup()

    def mean_of(ms, k):
        return float(np.mean([m[k] for m in ms]))

    tr_m, va_m = [o["m_tr"] for o in all_out], [o["m_va"] for o in all_out]
    print(f"\n{'='*20} CV SUMMARY {'='*20}")
    for tag, ms in [("TRAIN", tr_m), ("VAL", va_m)]:
        print(
            f"  {tag:>5}  "
            f"ACC {mean_of(ms,'acc'):.3f} | F1 {mean_of(ms,'f1'):.3f} | "
            f"P {mean_of(ms,'prec'):.3f} | R {mean_of(ms,'rec'):.3f} | "
            f"AUC {mean_of(ms,'auc'):.3f} | AP {mean_of(ms,'ap'):.3f} | "
            f"LL {mean_of(ms,'logloss'):.4f} | Brier {mean_of(ms,'brier'):.4f}"
        )

    p_ens, y_test, thr_global = _run_test_ensemble(
        all_out, df_test, cfg, graph_strategy, device
    )
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
#  MAIN PIPELINE — MLP BASELINE  (unchanged)
# ============================================================================
def run_pipeline_mlp(df: pd.DataFrame, cfg: CardFraudConfig):
    set_seed(cfg.SEED)
    device = get_device()

    print(f"\n{'#'*60}")
    print(f"# STRATEGY: mlp_baseline  (no graph — controlled ablation)")
    print(f"{'#'*60}\n")
    print(f"Device: {device}")
    print(f"Dataset: {df.shape},  fraud rate: {df[cfg.TARGET_COL].mean():.4f}\n")

    df_dev, df_test = split_groups_holdout(
        df, key=cfg.GROUP_KEY, target_col=cfg.TARGET_COL,
        train_ratio=cfg.TRAIN_RATIO, bins=cfg.STRATIFY_BINS, seed=cfg.SEED,
    )
    print(f"DEV  {df_dev.shape}  fraud={df_dev[cfg.TARGET_COL].mean():.4f}")
    print(f"TEST {df_test.shape}  fraud={df_test[cfg.TARGET_COL].mean():.4f}")

    folds = build_stratified_folds(
        df_dev, target_col=cfg.TARGET_COL, n_splits=cfg.N_SPLITS,
        group_key=cfg.GROUP_KEY, bins=cfg.STRATIFY_BINS, seed=cfg.SEED,
    )

    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"\n{'='*20} Fold {i}/{cfg.N_SPLITS} {'='*20}")
        out = train_one_fold_mlp(df_tr, df_va, cfg, fold_idx=i - 1, device=device)
        print(f"  Threshold: {out['thr']:.4f}")
        print_metrics("TRAIN", out["m_tr"])
        print_metrics("VAL",   out["m_va"])
        all_out.append(out); cleanup()

    def mean_of(ms, k):
        return float(np.mean([m[k] for m in ms]))

    tr_m, va_m = [o["m_tr"] for o in all_out], [o["m_va"] for o in all_out]
    print(f"\n{'='*20} CV SUMMARY {'='*20}")
    for tag, ms in [("TRAIN", tr_m), ("VAL", va_m)]:
        print(
            f"  {tag:>5}  "
            f"ACC {mean_of(ms,'acc'):.3f} | F1 {mean_of(ms,'f1'):.3f} | "
            f"P {mean_of(ms,'prec'):.3f} | R {mean_of(ms,'rec'):.3f} | "
            f"AUC {mean_of(ms,'auc'):.3f} | AP {mean_of(ms,'ap'):.3f} | "
            f"LL {mean_of(ms,'logloss'):.4f} | Brier {mean_of(ms,'brier'):.4f}"
        )

    p_ens, y_test, thr_global = _run_test_ensemble(
        all_out, df_test, cfg, "mlp_baseline", device
    )
    m_test = eval_from_probs(y_test, p_ens, thr_global)

    print(f"\n{'='*20} TEST (ENSEMBLED) {'='*20}")
    print(f"  Global threshold: {thr_global:.4f}")
    print_metrics("TEST", m_test)

    return {
        "all_out": all_out, "df_dev": df_dev, "df_test": df_test,
        "thr_global": thr_global, "test_metrics": m_test,
        "p_test_ens": p_ens, "y_test": y_test, "graph_strategy": "mlp_baseline",
        "model_arch": "mlp",
    }


# ============================================================================
#  VISUALIZATION
# ============================================================================
def plot_results(result: dict):
    y     = result["y_test"]
    p     = result["p_test_ens"]
    m     = result["test_metrics"]
    strat = result.get("graph_strategy", "")
    arch  = result.get("model_arch", "")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Sparkov Card Fraud — {strat} / {arch}", fontsize=14)

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
    f1s  = [eval_from_probs(y, p, t)["f1"]   for t in thrs]
    prs  = [eval_from_probs(y, p, t)["prec"]  for t in thrs]
    rcs  = [eval_from_probs(y, p, t)["rec"]   for t in thrs]
    axes[1, 1].plot(thrs, f1s, label="F1")
    axes[1, 1].plot(thrs, prs, label="Precision")
    axes[1, 1].plot(thrs, rcs, label="Recall")
    axes[1, 1].set(xlabel="Threshold", ylabel="Score", title="Threshold Tuning")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def compare_strategies(results: Dict[str, dict]):
    """Bar chart comparing all (strategy, arch) combinations."""
    metrics = ["f1", "prec", "rec", "auc", "ap"]
    labels  = list(results.keys())

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(metrics))
    w = 0.8 / len(labels)

    for i, key in enumerate(labels):
        vals = [results[key]["test_metrics"][m] for m in metrics]
        ax.bar(x + i * w, vals, w, label=key)

    ax.set_xticks(x + w * (len(labels) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Strategy × Architecture Comparison (Sparkov)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def print_final_comparison(results: Dict[str, dict]):
    """Print comparison table and topology-ranking consistency check."""
    print("\n" + "=" * 88)
    print("FINAL COMPARISON — SPARKOV CARD FRAUD")
    print(
        f"{'Run key':<34} | {'F1':>5} | {'P':>5} | {'R':>5} | "
        f"{'AUC':>5} | {'AP':>5} | {'Thr':>5}"
    )
    print("-" * 88)
    for key, res in results.items():
        m = res["test_metrics"]
        thr = res["thr_global"]
        print(
            f"  {key:<32s} | "
            f"{m['f1']:.3f} | {m['prec']:.3f} | {m['rec']:.3f} | "
            f"{m['auc']:.3f} | {m['ap']:.3f} | {thr:.3f}"
        )
    print("=" * 88)

    gnn_strats = ["multi_relation", "hybrid", "intra_group"]
    archs = ["gatv2", "gcn", "sage"]

    print("\n--- Topology ranking consistency across architectures ---")
    rankings = {}
    for arch in archs:
        arch_results = {
            strat: results[f"{strat}_{arch}"]["test_metrics"]["f1"]
            for strat in gnn_strats
            if f"{strat}_{arch}" in results
        }
        if not arch_results:
            continue
        ranked = sorted(arch_results.items(), key=lambda x: -x[1])
        rankings[arch] = [strat for strat, _ in ranked]
        print(f"  {arch.upper():8s}  " + "  >  ".join(
            f"{strat}({score:.3f})" for strat, score in ranked
        ))

    if len(rankings) >= 2:
        first_rank = next(iter(rankings.values()))
        consistent = all(rank == first_rank for rank in rankings.values())
        print(
            "\n  Ranking consistent across available architectures: "
            + ("YES" if consistent else "NO")
        )
        if consistent:
            print(
                "  -> Topology rankings are architecture-independent across the tested GNNs.\n"
                "     The ordering is a property of graph construction quality, not one operator."
            )
        else:
            print(
                "  -> Rankings differ: topology effects may interact with architecture.\n"
                "     Discuss this divergence in the paper."
            )


# ============================================================================
#  RUN ALL STRATEGIES × ARCHITECTURES
# ============================================================================
def run_all_strategies(df: pd.DataFrame, cfg: CardFraudConfig = None):
    """
    Runs all 3 graph strategies under GATv2, GCN, and GraphSAGE, plus the MLP baseline.
    Results dict keys: "multi_relation_gatv2", "multi_relation_gcn", etc.

    For a faster reviewer-response run, use run_gcn_comparison() below to
    run only the GCN arm (GATv2 results already in the paper).
    """
    if cfg is None:
        cfg = CardFraudConfig()

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


def run_gcn_comparison(df: pd.DataFrame, cfg: CardFraudConfig = None):
    """
    Convenience runner: GCN arm only — all 3 strategies under GCN.
    Use this when GATv2 results are already in the paper and you only need
    the GCN column to add to the architecture comparison table.
    """
    if cfg is None:
        cfg = CardFraudConfig()

    results = {}
    for strat in ["multi_relation", "hybrid", "intra_group"]:
        key = f"{strat}_gcn"
        print(f"\n{'#'*60}")
        print(f"# GCN — STRATEGY: {strat}")
        print(f"{'#'*60}\n")
        results[key] = run_pipeline(df, cfg, graph_strategy=strat, model_arch="gcn")
        plot_results(results[key])
        cleanup()

    print("\n--- GCN topology ranking ---")
    ranked = sorted(
        {s: results[f"{s}_gcn"]["test_metrics"]["f1"]
         for s in ["multi_relation", "hybrid", "intra_group"]}.items(),
        key=lambda x: -x[1],
    )
    for s, v in ranked:
        print(f"  {s}: F1={v:.3f}")

    return results


def run_sage_comparison(df: pd.DataFrame, cfg: CardFraudConfig = None):
    """Runs the GraphSAGE arm only — all 3 graph strategies under GraphSAGE."""
    if cfg is None:
        cfg = CardFraudConfig()

    results = {}
    for strat in ["multi_relation", "hybrid", "intra_group"]:
        key = f"{strat}_sage"
        print(f"\n{'#'*60}")
        print(f"# GRAPHSAGE — STRATEGY: {strat}")
        print(f"{'#'*60}\n")
        results[key] = run_pipeline(df, cfg, graph_strategy=strat, model_arch="sage")
        plot_results(results[key])
        cleanup()

    print("\n--- GraphSAGE topology ranking ---")
    ranked = sorted(
        {strat: results[f"{strat}_sage"]["test_metrics"]["f1"]
         for strat in ["multi_relation", "hybrid", "intra_group"]}.items(),
        key=lambda x: -x[1],
    )
    for strat, score in ranked:
        print(f"  {strat}: F1={score:.3f}")

    return results


# ============================================================================
#  MAIN
# ============================================================================
if __name__ == "__main__":
    cfg = CardFraudConfig()
    df  = load_and_preprocess("data/fraudTrain.csv", "data/fraudTest.csv", cfg)

    # Option A — full matrix (3 strategies × 3 architectures + MLP):
    results = run_all_strategies(df, cfg)

    # Option B — GraphSAGE arm only:
    # sage_results = run_sage_comparison(df, cfg)

    # Option C — GCN arm only:
    # gcn_results = run_gcn_comparison(df, cfg)

    # Option D — single run:
    # result = run_pipeline(df, cfg, graph_strategy="multi_relation", model_arch="sage")
    # plot_results(result)