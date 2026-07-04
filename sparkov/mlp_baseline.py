"""
mlp_baseline.py — Feature-identical MLP baseline for the Sparkov dataset.

No graph is constructed. The MLP receives the same per-transaction features
as the GNN models (same embeddings, same numerical scaling, same loss,
same splits, same threshold tuning) so that any performance gap between
this and the GNN models is attributable solely to neighbourhood aggregation.

Run:
    python mlp_baseline.py

Outputs saved to ../outcomes/
"""

import gc
import numpy as np
import torch
import torch.nn as nn

from config import CardFraudConfig
from utils import (
    set_seed, get_device, cleanup,
    load_and_preprocess,
    FoldPreprocessor, split_groups_holdout, build_stratified_folds,
    prepare_tensors, train_loop, run_test_ensemble,
    eval_from_probs, print_metrics, choose_threshold,
    save_results_plots, save_results_csv, print_final_comparison,
    ensure_outcome_dir,
)

MODEL_ARCH = "mlp"


# ============================================================================
#  MODEL
# ============================================================================
class MLPFraudModel(nn.Module):
    """
    Pure MLP baseline — identical feature encoding to the GNN models, no graph.
    Accepts edge_index for API compatibility with utils.run_test_ensemble
    but the argument is ignored; each transaction is classified independently.

    Architecture:
        embeddings + numerics
        → Linear(in→256) → LeakyReLU → Dropout
        → Linear(256→128) → LeakyReLU → Dropout
        → Linear(128→1)
    """

    def __init__(self, cardinalities, cat_cols, num_input_dim,
                 embedding_dim=8, hidden_dims=None, dropout=0.30):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.dropout  = dropout
        if hidden_dims is None: hidden_dims = [256, 128]
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinalities[col] + 1, embedding_dim)
            for col in cat_cols})
        in_dim = len(cat_cols) * embedding_dim + num_input_dim
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(), nn.Dropout(dropout)]; prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cat_dict, x_num, edge_index=None):
        cat_embs = [self.embeddings[col](x_cat_dict[col]) for col in self.cat_cols]
        x = torch.cat([torch.cat(cat_embs, dim=1), x_num], dim=1)
        return self.net(x).view(-1)


# ============================================================================
#  FOLD TRAINING
# ============================================================================
def train_one_fold(df_tr, df_va, cfg, fold_idx=0, device=None):
    if device is None: device = get_device()
    prep = FoldPreprocessor(cfg.CATEGORICAL_COLS, cfg.NUMERICAL_COLS)
    prep.fit(df_tr)
    df_tr_enc = prep.transform(df_tr)
    df_va_enc = prep.transform(df_va)

    _, num_tr, y_tr_np, cat_tr_d, num_tr_t = prepare_tensors(
        df_tr_enc, cfg.CATEGORICAL_COLS, prep.num_cols, cfg.TARGET_COL, device)
    _, num_va, y_va_np, cat_va_d, num_va_t = prepare_tensors(
        df_va_enc, cfg.CATEGORICAL_COLS, prep.num_cols, cfg.TARGET_COL, device)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float).to(device)

    model = MLPFraudModel(
        cardinalities=prep.cardinalities, cat_cols=cfg.CATEGORICAL_COLS,
        num_input_dim=num_tr.shape[1], embedding_dim=cfg.EMBEDDING_DIM,
        hidden_dims=cfg.MLP_HIDDEN_DIMS, dropout=cfg.DROPOUT).to(device)

    # edge_tr / edge_va are None — model ignores them
    model = train_loop(model, cat_tr_d, num_tr_t, y_tr_t, None,
                       cat_va_d, num_va_t, y_va_np, None, cfg, verbose=(fold_idx == 0))
    model.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(model(cat_tr_d, num_tr_t, None)).cpu().numpy()
        p_va = torch.sigmoid(model(cat_va_d, num_va_t, None)).cpu().numpy()
    thr = choose_threshold(y_va_np, p_va, cfg)
    cleanup()
    return {"model": model, "preprocessor": prep, "thr": thr,
            "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
            "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr)}


# ============================================================================
#  PIPELINE
# ============================================================================
def run_pipeline(df, cfg):
    set_seed(cfg.SEED); device = get_device()
    print(f"\n{'#'*60}\n# MLP Baseline (no graph)\n{'#'*60}")
    df_dev, df_test = split_groups_holdout(
        df, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    folds = build_stratified_folds(df_dev, cfg.TARGET_COL, cfg.N_SPLITS,
                                   cfg.GROUP_KEY, cfg.STRATIFY_BINS, cfg.SEED)
    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"  Fold {i}/{cfg.N_SPLITS}")
        out = train_one_fold(df_tr, df_va, cfg, i - 1, device)
        print_metrics("TRAIN", out["m_tr"]); print_metrics("VAL", out["m_va"])
        all_out.append(out); cleanup()
    p_ens, y_test, thr = run_test_ensemble(all_out, df_test, cfg, "mlp_baseline", device)
    m_test = eval_from_probs(y_test, p_ens, thr)
    print_metrics("TEST", m_test)
    result = {"all_out": all_out, "df_dev": df_dev, "df_test": df_test,
              "thr_global": thr, "test_metrics": m_test,
              "p_test_ens": p_ens, "y_test": y_test,
              "graph_strategy": "mlp_baseline", "model_arch": MODEL_ARCH}
    save_results_plots(result, cfg)
    save_results_csv({"mlp_baseline": result}, cfg)
    return result


if __name__ == "__main__":
    cfg = CardFraudConfig()
    df  = load_and_preprocess(train_path="fraudTrain.csv", test_path="fraudTest.csv", cfg=cfg)
    result = run_pipeline(df, cfg)
