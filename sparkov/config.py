"""
config.py — Sparkov dataset configuration and shared utilities.

All graph strategies, model hyperparameters, and dataset schema live here.
Import this in each model file to keep settings consistent across runs.
"""

from typing import Optional, List, Dict, Any


class CardFraudConfig:
    """All tuneable hyperparameters for the Sparkov pipeline."""

    # Reproducibility
    SEED: int = 42

    # Cross-validation
    N_SPLITS: int = 5
    TRAIN_RATIO: float = 0.8
    STRATIFY_BINS: int = 10

    # Training
    MAX_EPOCHS: int = 200
    EVAL_EVERY: int = 5
    PATIENCE_CHECKS: int = 12
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4

    # Shared model settings
    EMBEDDING_DIM: int = 8
    DROPOUT: float = 0.30

    # GATv2-specific
    HIDDEN_DIM: int = 64
    HEADS: int = 4

    # GCN-specific (wider to match GATv2 effective output: 64*4=256)
    GCN_HIDDEN_DIM: int = 256

    # GraphSAGE-specific
    SAGE_HIDDEN_DIM: int = 256

    # MLP baseline
    MLP_HIDDEN_DIMS: List[int] = [256, 128]

    # Threshold policy
    THRESHOLD_OVERRIDE: Optional[float] = None
    USE_COST_THRESHOLD: bool = False
    COST_FN: float = 100.0
    COST_FP: float = 1.0
    TARGET_RECALL: Optional[float] = None

    # Dataset schema
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

    # Temporal sort key
    SORT_KEY_COLS: Dict[str, int] = {"timestamp": 1}

    # Multi-relation graph
    MULTI_REL_SPECS: List[Dict[str, Any]] = [
        {"col": "cc_num",    "k": 2, "max_group_size": 2000},
        {"col": "merchant",  "k": 2, "max_group_size": 1000},
        {"col": "category",  "k": 1, "max_group_size": 3000},
        {"col": "zip",       "k": 1, "max_group_size": 2000},
    ]
    ADD_GLOBAL_TIME_EDGES: bool = True
    GLOBAL_TIME_K: int = 2
    SELF_LOOPS: bool = True

    # Hybrid graph: FAISS k-NN
    FAISS_K: int = 6
    FAISS_HNSW_M: int = 32
    FAISS_EF_SEARCH: int = 64

    # Intra-group graph
    INTRA_GROUP_KEY: str = "cc_num"
    INTRA_MAX_GROUP_SIZE: int = 200
    INTRA_K_TEMPORAL: int = 3
    INTRA_K_SIMILAR: int = 5
    INTRA_SIM_THRESHOLD: float = 0.5
    INTRA_SUB_RELATION_COLS: List[str] = ["merchant"]

    # Downsampling
    DOWNSAMPLE_NONFRAUD: int = 1_800_000

    # Output folder for all saved plots and CSVs
    OUTCOME_DIR: str = "../outcomes"
