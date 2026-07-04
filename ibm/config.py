"""
config.py — IBM dataset configuration and shared settings.
"""
from typing import Optional, List, Dict, Any


class IBMFraudConfig:
    SEED: int = 42
    N_SPLITS: int = 5
    TRAIN_RATIO: float = 0.8
    STRATIFY_BINS: int = 10
    MAX_EPOCHS: int = 200
    EVAL_EVERY: int = 5
    PATIENCE_CHECKS: int = 12
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    USE_AMP: bool = True

    DROPOUT: float = 0.30
    HIGH_CARD_EMB_DIM: int = 8
    LOW_CARD_EMB_DIM: int = 4

    HIDDEN_DIM: int = 32       # GAT hidden per head
    HEADS: int = 2             # GAT heads
    GCN_HIDDEN_DIM: int = 64  # matches GAT effective output (32*2)
    SAGE_HIDDEN_DIM: int = 64
    SAGE_AGGR: str = "mean"
    MLP_HIDDEN_DIMS: List[int] = [256, 128]

    THRESHOLD_OVERRIDE: Optional[float] = None
    USE_COST_THRESHOLD: bool = False
    COST_FN: float = 100.0
    COST_FP: float = 1.0
    TARGET_RECALL: Optional[float] = None

    TARGET_COL: str = "Is Fraud?"
    GROUP_KEY: str = "User"

    HIGH_CARD_COLS: List[str] = ["Merchant City", "Merchant State", "MCC"]
    LOW_CARD_COLS: List[str]  = ["Use Chip", "Errors?"]
    DENSE_FEATURE_COLS: List[str] = [
        "Amount", "user_avg_amount", "amount_over_user_avg",
        "amount_minus_user_avg", "Zip", "day_of_week", "is_weekend",
        "is_work_hour", "hour", "minute", "hour_sin", "hour_cos",
        "Year", "Month", "Day",
    ]

    SORT_KEY_COLS: Dict[str, int] = {
        "Year": 100_000_000, "Month": 1_000_000,
        "Day": 10_000, "hour": 100, "minute": 1,
    }

    MULTI_REL_SPECS: List[Dict[str, Any]] = [
        {"col": "Card",          "k": 1, "max_group_size": 500},
        {"col": "User",          "k": 1, "max_group_size": 500},
        {"col": "Merchant Name", "k": 1, "max_group_size": 500},
        {"col": "MCC",           "k": 1, "max_group_size": 1000},
    ]
    ADD_GLOBAL_TIME_EDGES: bool = False
    GLOBAL_TIME_K: int = 2
    SELF_LOOPS: bool = True

    FAISS_K: int = 4
    FAISS_HNSW_M: int = 32
    FAISS_EF_SEARCH: int = 64

    INTRA_GROUP_KEY: str = "Card"
    INTRA_MAX_GROUP_SIZE: int = 150
    INTRA_K_TEMPORAL: int = 2
    INTRA_K_SIMILAR: int = 4
    INTRA_SIM_THRESHOLD: float = 0.5
    INTRA_SUB_RELATION_COLS: List[str] = ["Merchant Name"]

    DOWNSAMPLE: bool = True
    DOWNSAMPLE_RATIO: int = 10

    OUTCOME_DIR: str = "../outcomes"
