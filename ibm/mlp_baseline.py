"""
mlp_baseline.py — Feature-identical MLP baseline for the IBM dataset.

No graph constructed. Same features, loss, splits, threshold as GNN models.
Run:  python mlp_baseline.py
Outputs saved to ../outcomes/
"""

import gc
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from config import IBMFraudConfig
from utils import (
    set_seed, get_device, cleanup,
    load_and_preprocess, add_user_spending_features,
    FoldPreprocessor, split_groups_holdout, build_group_stratified_folds,
    train_loop, safe_inference, run_test_ensemble,
    eval_from_probs, print_metrics, choose_threshold,
    save_results_plots, save_results_csv, print_final_comparison,
)

MODEL_ARCH = "mlp"


class MLPFraudModel(nn.Module):
    """Feature-identical MLP — no graph. edge_index accepted but ignored."""

    def __init__(self, dense_in_dim, high_card_cols, low_card_cols,
                 factor_cardinalities, hidden_dims=None,
                 dropout=0.30, high_card_emb_dim=8, low_card_emb_dim=4):
        super().__init__()
        self.dropout = dropout
        self.high_card_cols = list(high_card_cols)
        self.low_card_cols  = list(low_card_cols)
        self.high_emb = nn.ModuleDict({
            col: nn.Embedding(int(factor_cardinalities[col]) + 2, high_card_emb_dim)
            for col in high_card_cols})
        self.low_emb = nn.ModuleDict({
            col: nn.Embedding(int(factor_cardinalities[col]) + 2, low_card_emb_dim)
            for col in low_card_cols})
        if hidden_dims is None: hidden_dims = [256, 128]
        in_dim = (dense_in_dim + len(high_card_cols)*high_card_emb_dim
                  + len(low_card_cols)*low_card_emb_dim)
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(), nn.Dropout(dropout)]; prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def encode(self, x_dense, x_high, x_low):
        parts = [x_dense]
        if self.high_card_cols:
            parts.append(torch.cat([self.high_emb[col](x_high[:, i]+1)
                                    for i, col in enumerate(self.high_card_cols)], dim=1))
        if self.low_card_cols:
            parts.append(torch.cat([self.low_emb[col](x_low[:, i]+1)
                                    for i, col in enumerate(self.low_card_cols)], dim=1))
        return torch.cat(parts, dim=1)

    def forward(self, x_dense, x_high, x_low, edge_index=None):
        return self.net(self.encode(x_dense, x_high, x_low)).view(-1)


def train_one_fold(df_tr, df_va, cfg, fold_idx=0, device=None):
    if device is None: device = get_device()
    df_tr, df_va, user_avg_map, global_avg = add_user_spending_features(df_tr, df_va)
    prep = FoldPreprocessor(cfg); prep.fit(df_tr)
    X_d_tr, X_h_tr, X_l_tr, y_tr_np = prep.transform(df_tr)
    X_d_va, X_h_va, X_l_va, y_va_np = prep.transform(df_va)

    x_d_tr = torch.tensor(X_d_tr, dtype=torch.float32).to(device)
    x_h_tr = torch.tensor(X_h_tr, dtype=torch.long).to(device)
    x_l_tr = torch.tensor(X_l_tr, dtype=torch.long).to(device)
    y_tr_t  = torch.tensor(y_tr_np, dtype=torch.float32).to(device)
    x_d_va = torch.tensor(X_d_va, dtype=torch.float32).to(device)
    x_h_va = torch.tensor(X_h_va, dtype=torch.long).to(device)
    x_l_va = torch.tensor(X_l_va, dtype=torch.long).to(device)
    del X_d_tr, X_h_tr, X_l_tr, X_d_va, X_h_va, X_l_va; gc.collect()

    model = MLPFraudModel(
        dense_in_dim=x_d_tr.shape[1],
        high_card_cols=prep.high_card_cols, low_card_cols=prep.low_card_cols,
        factor_cardinalities=prep.factor_cardinalities,
        hidden_dims=cfg.MLP_HIDDEN_DIMS, dropout=cfg.DROPOUT,
        high_card_emb_dim=cfg.HIGH_CARD_EMB_DIM, low_card_emb_dim=cfg.LOW_CARD_EMB_DIM,
    ).to(device)

    use_amp = cfg.USE_AMP and device.type == "cuda"
    model = train_loop(model, x_d_tr, x_h_tr, x_l_tr, y_tr_t, None,
                       x_d_va, x_h_va, x_l_va, y_va_np, None, cfg, device,
                       verbose=(fold_idx == 0))
    model.eval()
    p_tr = safe_inference(model, x_d_tr, x_h_tr, x_l_tr, None, device, use_amp)
    p_va = safe_inference(model, x_d_va, x_h_va, x_l_va, None, device, use_amp)
    thr  = choose_threshold(y_va_np, p_va, cfg)
    del x_d_tr, x_h_tr, x_l_tr, y_tr_t, x_d_va, x_h_va, x_l_va; cleanup()
    return {"model": model.cpu(), "preprocessor": prep, "thr": thr,
            "user_avg_map": user_avg_map, "global_avg": global_avg,
            "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
            "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr)}


def run_pipeline(df, cfg):
    set_seed(cfg.SEED); device = get_device()
    print(f"\n{'#'*60}\n# MLP Baseline (no graph)\n{'#'*60}")
    df_dev, df_test = split_groups_holdout(
        df, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    folds = build_group_stratified_folds(
        df_dev, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.N_SPLITS, cfg.STRATIFY_BINS, cfg.SEED)
    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"  Fold {i}/{cfg.N_SPLITS}")
        out = train_one_fold(df_tr, df_va, cfg, i-1, device)
        print_metrics("TRAIN", out["m_tr"]); print_metrics("VAL", out["m_va"])
        all_out.append(out); cleanup()
    p_ens, y_test, thr = run_test_ensemble(all_out, df_test, cfg, "mlp_baseline", device)
    m_test = eval_from_probs(y_test, p_ens, thr); print_metrics("TEST", m_test)
    result = {"all_out": all_out, "df_dev": df_dev, "df_test": df_test,
              "thr_global": thr, "test_metrics": m_test,
              "p_test_ens": p_ens, "y_test": y_test,
              "graph_strategy": "mlp_baseline", "model_arch": MODEL_ARCH}
    save_results_plots(result, cfg)
    save_results_csv({"mlp_baseline": result}, cfg)
    return result


if __name__ == "__main__":
    cfg = IBMFraudConfig()
    df  = load_and_preprocess(path="/path/to/ibm_transactions.parquet", cfg=cfg)
    result = run_pipeline(df, cfg)
