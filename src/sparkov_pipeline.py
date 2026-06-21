##############################################################################
# CARD TRANSACTION FRAUD (SPARKOV) — STANDALONE GAT PIPELINE
# + EDGE-TYPE ABLATION FOR INTRA-GROUP STRATEGY
#
# Dataset: fraudTrain.csv + fraudTest.csv (Sparkov simulated credit card
# transactions, Kaggle: kartik2112/fraud-detection)
# Columns: cc_num, merchant, category, amt, lat, long, merch_lat, merch_long,
#          city, state, city_pop, job, dob, trans_date_trans_time, is_fraud, ...
#
# Graph strategies (pass via `graph_strategy`):
#   "multi_relation"  – temporal edges along cc_num, merchant, category, zip
#   "hybrid"          – multi_relation + FAISS k-NN feature-similarity edges
#   "intra_group"     – intra-cc_num temporal + similarity + merchant chains
#                        (full model; equivalent to edge_types={"temporal",
#                        "similarity","merchant"})
#
# Edge-type ablation strategies (same intra-group machinery, restricted
# edge_types set — isolates the contribution of each edge type while
# holding architecture/training/splits/threshold-tuning fixed):
#   "intra_temporal_only"    – edge_types={"temporal"}
#   "intra_similarity_only"  – edge_types={"similarity"}
#   "intra_merchant_only"    – edge_types={"merchant"}
#   "intra_group" (full)     – edge_types={"temporal","similarity","merchant"}
#
# Fixed architecture:
#   • GATv2Conv (2 layers, 4 heads, LayerNorm)
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
from typing import Optional, List, Dict, Any, Set, FrozenSet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

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
    """All tuneable knobs for the card-transaction (Sparkov) pipeline."""

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

    # --- model ---
    HIDDEN_DIM: int = 64
    HEADS: int = 4
    DROPOUT: float = 0.30
    EMBEDDING_DIM: int = 8

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

    # --- graph: intra-group edge-type ablation ---
    INTRA_ABLATION_EDGE_TYPES: Dict[str, FrozenSet[str]] = {
        "intra_group":            frozenset({"temporal", "similarity", "merchant"}),  # full model
        "intra_temporal_only":    frozenset({"temporal"}),
        "intra_similarity_only":  frozenset({"similarity"}),
        "intra_merchant_only":    frozenset({"merchant"}),
    }

    # --- downsampling ---
    DOWNSAMPLE_NONFRAUD: int = 1_800_000   # how many non-fraud to DROP


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
    """
    Load fraudTrain + fraudTest, engineer features, downsample, and return
    a clean DataFrame ready for the pipeline.
    """
    if cfg is None:
        cfg = CardFraudConfig()

    # ------------------------------------------------------------------
    # 1) Load and combine
    # ------------------------------------------------------------------
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2], ignore_index=True).dropna()
    del df1, df2
    gc.collect()
    print(f"Combined dataset: {len(df):,} transactions")

    # ------------------------------------------------------------------
    # 2) Parse datetime & compute age
    # ------------------------------------------------------------------
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = ((df["trans_date_trans_time"] - df["dob"]).dt.days // 365).astype("int16")

    # ------------------------------------------------------------------
    # 3) Time features
    # ------------------------------------------------------------------
    df["hour"] = df["trans_date_trans_time"].dt.hour.astype("int8")
    df["weekday"] = df["trans_date_trans_time"].dt.weekday.astype("int8")
    df["trans_month"] = df["trans_date_trans_time"].dt.month.astype("int8")
    df["is_fraud"] = df["is_fraud"].astype("int8")

    # cyclic hour
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype("float32")
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype("float32")

    # business hours flag
    df["business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype("int8")

    # ------------------------------------------------------------------
    # 4) Downsample non-fraud
    # ------------------------------------------------------------------
    nonfraud_idx = df.index[df["is_fraud"] == 0]
    n_drop = min(len(nonfraud_idx), cfg.DOWNSAMPLE_NONFRAUD)
    drop_idx = np.random.RandomState(cfg.SEED).choice(nonfraud_idx, size=n_drop, replace=False)
    df = df.drop(drop_idx).reset_index(drop=True)
    print(f"After downsampling: {len(df):,} transactions  "
          f"(fraud rate: {df['is_fraud'].mean():.4f})")

    # ------------------------------------------------------------------
    # 5) Drop columns not needed for modelling
    # ------------------------------------------------------------------
    cols_to_drop = [
        "Unnamed: 0", "first", "last", "street", "trans_num", "dob",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ------------------------------------------------------------------
    # 6) Sort by (cc_num, trans_date_trans_time) and create timestamp index
    # ------------------------------------------------------------------
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    df["timestamp"] = np.arange(len(df), dtype=np.int64)

    # ------------------------------------------------------------------
    # 7) Inter-transaction time difference (within each cardholder)
    # ------------------------------------------------------------------
    df["trans_time_diff"] = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
        .astype("float32")
    )

    # ------------------------------------------------------------------
    # 8) Distance: customer <-> merchant
    # ------------------------------------------------------------------
    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        df["distance"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2
            + (df["long"] - df["merch_long"]) ** 2
        ).astype("float32")

    # ------------------------------------------------------------------
    # 9) Amount relative to cardholder average
    # ------------------------------------------------------------------
    if "cc_num" in df.columns and "amt" in df.columns:
        cc_avg = df.groupby("cc_num")["amt"].transform("mean")
        df["amt_to_avg"] = (df["amt"] / (cc_avg + 1e-10)).astype("float32")

    # ------------------------------------------------------------------
    # 10) Ensure cc_num is numeric for group splitting
    # ------------------------------------------------------------------
    df["cc_num"] = pd.to_numeric(df["cc_num"], errors="coerce").fillna(-1)

    # Drop the datetime column (no longer needed; timestamp is the proxy)
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
        f"Acc {m['acc']:.{nd}f} | "
        f"F1 {m['f1']:.{nd}f} | "
        f"P {m['prec']:.{nd}f} | "
        f"R {m['rec']:.{nd}f} | "
        f"AUC {m['auc']:.{nd}f} | "
        f"AP {m['ap']:.{nd}f} | "
        f"LL {m['logloss']:.{ndloss}f} | "
        f"Brier {m['brier']:.{ndloss}f}"
    )
    print(f"           CM:\n{m['cm']}\n")


# ============================================================================
#  FOLD PREPROCESSOR (fit on train only)
# ============================================================================
class FoldPreprocessor:
    """LabelEncoders (categoricals) + StandardScaler (numericals), no leakage."""

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
            self.cardinalities[col] = len(le.classes_)
        self.num_cols = [c for c in self.num_cols if c in df_train.columns]
        self.scaler.fit(df_train[self.num_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cat_cols:
            le = self.label_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x, _k=known, _le=le: (
                    _le.transform([x])[0] if x in _k else len(_le.classes_)
                )
            )
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df


# ============================================================================
#  DATA SPLITTING (group-aware)
# ============================================================================
def split_groups_holdout(df, key, target_col, train_ratio=0.8, bins=10, seed=42):
    """Group-aware hold-out split stratified by per-group fraud prevalence."""
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

    df_dev = df[df[key].isin(set(dev_groups))].reset_index(drop=True)
    df_test = df[df[key].isin(set(test_groups))].reset_index(drop=True)
    return df_dev, df_test


def build_stratified_folds(df_dev, target_col, n_splits=5, group_key=None, bins=10, seed=42):
    """Group-stratified or row-level stratified folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if group_key is not None and group_key in df_dev.columns:
        grp = df_dev.groupby(group_key)[target_col].mean().rename("prev").reset_index()
        try:
            grp["bin"] = pd.qcut(grp["prev"], q=bins, labels=False, duplicates="drop")
        except ValueError:
            grp["bin"] = 0
        groups = grp[group_key].values
        ybins = grp["bin"].fillna(0).astype(int).values

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
#  GRAPH BUILDERS — strategies
# ============================================================================

def _make_sort_key(df: pd.DataFrame, sort_key_spec: Dict[str, int]) -> np.ndarray:
    key = np.zeros(len(df), dtype=np.int64)
    for col, mult in (sort_key_spec or {}).items():
        if col in df.columns:
            key += pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64).values * mult
    return key


# -------- strategy 1: multi-relation temporal -------------------------------
def _build_multi_relation_edges(df_raw, cfg: CardFraudConfig) -> set:
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

    # global time edges
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


# -------- strategy 2: hybrid (multi-relation + FAISS) -----------------------
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


# -------- strategy 3: intra-group multi-strategy (with edge-type ablation) --
def _build_intra_group_edges(
    df_raw,
    cat_np,
    num_np,
    cfg: CardFraudConfig,
    edge_types: Optional[Set[str]] = None,
) -> set:
    """
    Build intra-group edges for the cardholder-isolated graph.

    edge_types controls which of the three edge families to build:
        {"temporal", "similarity", "merchant"}  (default: all three -> full model)
    This is the hook used by the edge-type ablation: pass a restricted
    set (e.g. {"temporal"}) to build only that edge family while keeping
    everything else (architecture, training, splits, threshold tuning)
    identical.
    """
    if edge_types is None:
        edge_types = {"temporal", "similarity", "merchant"}

    edge_set = set()
    group_col = cfg.INTRA_GROUP_KEY
    if group_col not in df_raw.columns:
        raise ValueError(f"Intra-group key '{group_col}' not in dataframe")

    sort_key = _make_sort_key(df_raw, cfg.SORT_KEY_COLS)
    all_features = np.hstack([cat_np.astype("float32"), num_np.astype("float32")])

    for _, indices in df_raw.groupby(group_col).groups.items():
        indices = list(indices)
        if len(indices) > cfg.INTRA_MAX_GROUP_SIZE:
            indices = np.random.choice(indices, cfg.INTRA_MAX_GROUP_SIZE, replace=False).tolist()
        if len(indices) <= 1:
            continue

        sorted_idx = sorted(indices, key=lambda i: sort_key[i])

        # (a) temporal sequential edges
        if "temporal" in edge_types:
            for i in range(len(sorted_idx)):
                for j in range(max(0, i - cfg.INTRA_K_TEMPORAL), i):
                    a, b = sorted_idx[i], sorted_idx[j]
                    edge_set.add((a, b))
                    edge_set.add((b, a))

        # (b) cosine-similarity edges
        if "similarity" in edge_types:
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

        # (c) sub-relation chain edges (e.g. same merchant within cardholder)
        if "merchant" in edge_types:
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


# -------- unified dispatcher ------------------------------------------------
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
        edge_set = _build_intra_group_edges(
            df_raw, cat_np, num_np, cfg,
            edge_types=cfg.INTRA_ABLATION_EDGE_TYPES.get(
                "intra_group", {"temporal", "similarity", "merchant"}
            ),
        )
    elif strategy in cfg.INTRA_ABLATION_EDGE_TYPES:
        # Ablation strategies: intra_temporal_only, intra_similarity_only,
        # intra_merchant_only (and any custom entries added to the config dict)
        edge_set = _build_intra_group_edges(
            df_raw, cat_np, num_np, cfg,
            edge_types=cfg.INTRA_ABLATION_EDGE_TYPES[strategy],
        )
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
#  MODEL (identical across strategies)
# ============================================================================
class UnifiedGATNet(nn.Module):
    """
    GATv2 node classifier.
    Categorical embeddings + scaled numericals -> GAT1 -> LN -> GAT2 -> LN -> MLP.
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
        self.dropout = dropout

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinalities[col] + 1, embedding_dim)
            for col in cat_cols
        })

        in_channels = len(cat_cols) * embedding_dim + num_input_dim

        self.gat1 = GATv2Conv(in_channels, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden * heads)

        d = hidden * heads
        self.cls = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
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


# ============================================================================
#  TRAIN ONE FOLD
# ============================================================================
def train_one_fold(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    cfg: CardFraudConfig,
    graph_strategy: str,
    fold_idx: int = 0,
    device: torch.device = None,
):
    if device is None:
        device = get_device()

    # --- preprocess ---
    prep = FoldPreprocessor(cfg.CATEGORICAL_COLS, cfg.NUMERICAL_COLS)
    prep.fit(df_tr)

    df_tr_raw, df_va_raw = df_tr.copy(), df_va.copy()
    df_tr_enc = prep.transform(df_tr)
    df_va_enc = prep.transform(df_va)

    cat_tr = df_tr_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
    num_tr = df_tr_enc[prep.num_cols].to_numpy(dtype=np.float32)
    y_tr_np = df_tr_enc[cfg.TARGET_COL].to_numpy(dtype=np.int64)

    cat_va = df_va_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
    num_va = df_va_enc[prep.num_cols].to_numpy(dtype=np.float32)
    y_va_np = df_va_enc[cfg.TARGET_COL].to_numpy(dtype=np.int64)

    # --- build graphs ---
    verbose = fold_idx == 0
    if verbose:
        print(f"  Building TRAIN graph ({graph_strategy})...")
    edge_tr = build_graph_edges(graph_strategy, df_tr_raw, cat_tr, num_tr, cfg, verbose)
    if verbose:
        print(f"  Building VAL graph ({graph_strategy})...")
    edge_va = build_graph_edges(graph_strategy, df_va_raw, cat_va, num_va, cfg, verbose)

    # --- to device ---
    cat_tr_d = {col: torch.tensor(cat_tr[:, i], dtype=torch.long).to(device)
                for i, col in enumerate(cfg.CATEGORICAL_COLS)}
    num_tr_t = torch.tensor(num_tr, dtype=torch.float).to(device)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float).to(device)
    edge_tr = edge_tr.to(device)

    cat_va_d = {col: torch.tensor(cat_va[:, i], dtype=torch.long).to(device)
                for i, col in enumerate(cfg.CATEGORICAL_COLS)}
    num_va_t = torch.tensor(num_va, dtype=torch.float).to(device)
    edge_va = edge_va.to(device)

    # --- model ---
    model = UnifiedGATNet(
        cardinalities=prep.cardinalities,
        cat_cols=cfg.CATEGORICAL_COLS,
        num_input_dim=num_tr.shape[1],
        embedding_dim=cfg.EMBEDDING_DIM,
        hidden=cfg.HIDDEN_DIM,
        heads=cfg.HEADS,
        dropout=cfg.DROPOUT,
    ).to(device)

    pos = max(1, y_tr_np.sum())
    neg = max(1, len(y_tr_np) - pos)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float, device=device)

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-5)

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
        opt.step()
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

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(model(cat_tr_d, num_tr_t, edge_tr)).cpu().numpy()
        p_va = torch.sigmoid(model(cat_va_d, num_va_t, edge_va)).cpu().numpy()

    thr = choose_threshold(y_va_np, p_va, cfg)
    return {
        "model": model, "preprocessor": prep, "thr": thr,
        "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
        "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr),
    }


# ============================================================================
#  MAIN PIPELINE
# ============================================================================
def run_pipeline(
    df: pd.DataFrame,
    cfg: CardFraudConfig,
    graph_strategy: str = "multi_relation",
):
    set_seed(cfg.SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"Graph strategy: {graph_strategy}")
    print(f"Dataset: {df.shape},  fraud rate: {df[cfg.TARGET_COL].mean():.4f}\n")

    # --- hold-out test ---
    df_dev, df_test = split_groups_holdout(
        df, key=cfg.GROUP_KEY, target_col=cfg.TARGET_COL,
        train_ratio=cfg.TRAIN_RATIO, bins=cfg.STRATIFY_BINS, seed=cfg.SEED,
    )
    print(f"DEV  {df_dev.shape}  fraud={df_dev[cfg.TARGET_COL].mean():.4f}")
    print(f"TEST {df_test.shape}  fraud={df_test[cfg.TARGET_COL].mean():.4f}")

    # --- folds ---
    folds = build_stratified_folds(
        df_dev, target_col=cfg.TARGET_COL, n_splits=cfg.N_SPLITS,
        group_key=cfg.GROUP_KEY, bins=cfg.STRATIFY_BINS, seed=cfg.SEED,
    )

    # --- train ---
    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"\n{'='*20} Fold {i}/{cfg.N_SPLITS} {'='*20}")
        out = train_one_fold(df_tr, df_va, cfg, graph_strategy, fold_idx=i - 1, device=device)
        print(f"  Threshold: {out['thr']:.4f}")
        print_metrics("TRAIN", out["m_tr"])
        print_metrics("VAL", out["m_va"])
        all_out.append(out)
        cleanup()

    # --- CV summary ---
    def mean_of(ms, k):
        return float(np.mean([m[k] for m in ms]))

    def std_of(ms, k):
        return float(np.std([m[k] for m in ms]))

    tr_m = [o["m_tr"] for o in all_out]
    va_m = [o["m_va"] for o in all_out]

    print(f"\n{'='*20} CV SUMMARY {'='*20}")
    for tag, ms in [("TRAIN", tr_m), ("VAL", va_m)]:
        print(
            f"  {tag:>5}  "
            f"ACC {mean_of(ms,'acc'):.3f} | F1 {mean_of(ms,'f1'):.3f}+/-{std_of(ms,'f1'):.3f} | "
            f"P {mean_of(ms,'prec'):.3f} | R {mean_of(ms,'rec'):.3f} | "
            f"AUC {mean_of(ms,'auc'):.3f} | AP {mean_of(ms,'ap'):.3f} | "
            f"LL {mean_of(ms,'logloss'):.4f} | Brier {mean_of(ms,'brier'):.4f}"
        )

    # --- test ensemble ---
    test_probs = []
    for out in all_out:
        prep = out["preprocessor"]
        df_te_enc = prep.transform(df_test)
        cat_te = df_te_enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
        num_te = df_te_enc[prep.num_cols].to_numpy(dtype=np.float32)

        edge_te = build_graph_edges(graph_strategy, df_test, cat_te, num_te, cfg).to(device)
        cat_te_d = {col: torch.tensor(cat_te[:, i], dtype=torch.long).to(device)
                    for i, col in enumerate(cfg.CATEGORICAL_COLS)}
        num_te_t = torch.tensor(num_te, dtype=torch.float).to(device)

        out["model"].eval()
        with torch.no_grad():
            test_probs.append(torch.sigmoid(out["model"](cat_te_d, num_te_t, edge_te)).cpu().numpy())

    p_test_ens = np.mean(np.vstack(test_probs), axis=0)
    y_test = df_test[cfg.TARGET_COL].astype(int).to_numpy()

    thr_global = choose_threshold(
        np.concatenate([o["y_va"] for o in all_out]),
        np.concatenate([o["p_va"] for o in all_out]),
        cfg,
    )
    m_test = eval_from_probs(y_test, p_test_ens, thr_global)

    print(f"\n{'='*20} TEST (ENSEMBLED) {'='*20}")
    print(f"  Global threshold: {thr_global:.4f}")
    print_metrics("TEST", m_test)

    return {
        "all_out": all_out, "df_dev": df_dev, "df_test": df_test,
        "thr_global": thr_global, "test_metrics": m_test,
        "p_test_ens": p_test_ens, "y_test": y_test,
        "graph_strategy": graph_strategy,
    }


# ============================================================================
#  VISUALIZATION
# ============================================================================
def plot_results(result: dict):
    y = result["y_test"]
    p = result["p_test_ens"]
    m = result["test_metrics"]
    strat = result.get("graph_strategy", "")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Test Results — {strat}", fontsize=14)

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
    f1s = [eval_from_probs(y, p, t)["f1"] for t in thrs]
    prs = [eval_from_probs(y, p, t)["prec"] for t in thrs]
    rcs = [eval_from_probs(y, p, t)["rec"] for t in thrs]
    axes[1, 1].plot(thrs, f1s, label="F1")
    axes[1, 1].plot(thrs, prs, label="Precision")
    axes[1, 1].plot(thrs, rcs, label="Recall")
    axes[1, 1].set(xlabel="Threshold", ylabel="Score", title="Threshold Tuning")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def compare_strategies(results: Dict[str, dict]):
    metrics = ["f1", "prec", "rec", "auc", "ap"]
    strats = list(results.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    w = 0.8 / len(strats)

    for i, s in enumerate(strats):
        vals = [results[s]["test_metrics"][m] for m in metrics]
        ax.bar(x + i * w, vals, w, label=s)

    ax.set_xticks(x + w * (len(strats) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Card Fraud — Strategy Comparison")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
#  RUN ALL THREE (ORIGINAL) GRAPH-CONSTRUCTION STRATEGIES
# ============================================================================
def run_all_strategies(df: pd.DataFrame, cfg: CardFraudConfig = None):
    """Original comparison: multi_relation vs hybrid vs intra_group (full)."""
    if cfg is None:
        cfg = CardFraudConfig()

    strategies = ["multi_relation", "hybrid", "intra_group"]
    results = {}

    for strat in strategies:
        print(f"\n{'#'*60}")
        print(f"# STRATEGY: {strat}")
        print(f"{'#'*60}\n")
        results[strat] = run_pipeline(df, cfg, graph_strategy=strat)
        plot_results(results[strat])

    compare_strategies(results)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    for strat, res in results.items():
        m = res["test_metrics"]
        print(
            f"  {strat:<20s} | "
            f"F1 {m['f1']:.3f} | P {m['prec']:.3f} | R {m['rec']:.3f} | "
            f"AUC {m['auc']:.3f} | AP {m['ap']:.3f}"
        )

    return results


# ============================================================================
#  RUN EDGE-TYPE ABLATION FOR THE INTRA-GROUP STRATEGY
# ============================================================================
def run_edge_ablation(df: pd.DataFrame, cfg: CardFraudConfig = None):
    """
    Edge-type ablation: holds architecture, training protocol, splits, and
    threshold-tuning fixed, and varies ONLY which intra-group edge types
    are constructed.

    Produces 4 runs:
        intra_temporal_only    -> E = E_temporal + self-loops
        intra_similarity_only  -> E = E_similarity + self-loops
        intra_merchant_only    -> E = E_merchant + self-loops
        intra_group            -> E = E_temporal U E_similarity U E_merchant
                                   + self-loops   (full model, main paper result)
    """
    if cfg is None:
        cfg = CardFraudConfig()

    ablation_strategies = [
        "intra_temporal_only",
        "intra_similarity_only",
        "intra_merchant_only",
        "intra_group",  # full model — run last so it's the "reference" row
    ]

    results = {}
    for strat in ablation_strategies:
        print(f"\n{'#' * 60}")
        print(f"# ABLATION VARIANT: {strat}  "
              f"(edges={sorted(cfg.INTRA_ABLATION_EDGE_TYPES[strat])})")
        print(f"{'#' * 60}\n")
        results[strat] = run_pipeline(df, cfg, graph_strategy=strat)
        plot_results(results[strat])
        cleanup()

    compare_strategies(results)

    print("\n" + "=" * 70)
    print("EDGE-TYPE ABLATION SUMMARY — SPARKOV CARD FRAUD (intra-group)")
    print("=" * 70)
    print(f"  {'Variant':<24s} | {'F1':>6s} | {'P':>6s} | {'R':>6s} | {'AUC':>6s} | {'AP':>6s} | {'Brier':>7s}")
    for strat in ablation_strategies:
        m = results[strat]["test_metrics"]
        print(
            f"  {strat:<24s} | {m['f1']:.3f} | {m['prec']:.3f} | {m['rec']:.3f} | "
            f"{m['auc']:.3f} | {m['ap']:.3f} | {m['brier']:.4f}"
        )

    return results


# ============================================================================
#  MAIN
# ============================================================================
if __name__ == "__main__":
    # 1) Load and preprocess
    cfg = CardFraudConfig()
    df = load_and_preprocess("data/fraudTrain.csv", "data/fraudTest.csv", cfg)

    # 2) Run all three original graph strategies
    # results = run_all_strategies(df, cfg)

    # Or run a single strategy:
    # result = run_pipeline(df, cfg, graph_strategy="hybrid")
    # plot_results(result)

    # 3) Edge-type ablation for the intra-group strategy
    ablation_results = run_edge_ablation(df, cfg)
