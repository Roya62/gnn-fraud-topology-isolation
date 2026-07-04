"""
GATv2 fraud detector for the IBM dataset.

Graph strategies: multi_relation | hybrid | intra_group
Run:  python gatv2_model.py
Outputs saved to ../outcomes/
"""

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from config import IBMFraudConfig
from utils import (
    set_seed, get_device, cleanup,
    load_and_preprocess, add_user_spending_features,
    FoldPreprocessor, split_groups_holdout, build_group_stratified_folds,
    build_graph_edges, train_loop, safe_inference,
    run_test_ensemble, eval_from_probs, print_metrics, choose_threshold,
    save_results_plots, save_strategy_comparison,
    save_results_csv, print_final_comparison,
)

MODEL_ARCH = "gatv2"


# ============================================================================
#  MODEL
# ============================================================================
class GATFraudModel(nn.Module):
    """Two-layer GATConv classifier with multi-head attention."""

    def __init__(self, dense_in_dim, high_card_cols, low_card_cols,
                 factor_cardinalities, hidden, heads=2,
                 dropout=0.30, high_card_emb_dim=8, low_card_emb_dim=4):
        super().__init__()
        self.dropout        = dropout
        self.high_card_cols = list(high_card_cols)
        self.low_card_cols  = list(low_card_cols)
        self.high_emb = nn.ModuleDict({
            col: nn.Embedding(int(factor_cardinalities[col]) + 2, high_card_emb_dim)
            for col in high_card_cols})
        self.low_emb = nn.ModuleDict({
            col: nn.Embedding(int(factor_cardinalities[col]) + 2, low_card_emb_dim)
            for col in low_card_cols})
        in_dim = (dense_in_dim + len(high_card_cols)*high_card_emb_dim
                  + len(low_card_cols)*low_card_emb_dim)
        self.gat1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden*heads, hidden, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden*heads)
        self.norm2 = nn.LayerNorm(hidden*heads)
        self.cls = nn.Sequential(
            nn.Linear(hidden*heads, 128), nn.LeakyReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def encode(self, x_dense, x_high, x_low):
        parts = [x_dense]
        if self.high_card_cols:
            parts.append(torch.cat([self.high_emb[col](x_high[:, i]+1)
                                    for i, col in enumerate(self.high_card_cols)], dim=1))
        if self.low_card_cols:
            parts.append(torch.cat([self.low_emb[col](x_low[:, i]+1)
                                    for i, col in enumerate(self.low_card_cols)], dim=1))
        return torch.cat(parts, dim=1)

    def forward(self, x_dense, x_high, x_low, edge_index):
        x = F.dropout(self.encode(x_dense, x_high, x_low), p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm1(self.gat1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm2(self.gat2(h, edge_index)))
        return self.cls(h).view(-1)


# ============================================================================
#  FOLD TRAINING
# ============================================================================
def train_one_fold(df_tr, df_va, cfg, graph_strategy, fold_idx=0, device=None):
    if device is None: device = get_device()
    df_tr, df_va, user_avg_map, global_avg = add_user_spending_features(df_tr, df_va)
    prep = FoldPreprocessor(cfg); prep.fit(df_tr)
    df_tr_raw, df_va_raw = df_tr.copy(), df_va.copy()
    X_d_tr, X_h_tr, X_l_tr, y_tr_np = prep.transform(df_tr)
    X_d_va, X_h_va, X_l_va, y_va_np = prep.transform(df_va)

    verbose = (fold_idx == 0)
    edge_tr = build_graph_edges(graph_strategy, df_tr_raw, X_d_tr, X_h_tr, X_l_tr, cfg, verbose)
    edge_va = build_graph_edges(graph_strategy, df_va_raw, X_d_va, X_h_va, X_l_va, cfg, verbose)
    del df_tr_raw, df_va_raw; gc.collect()

    x_d_tr = torch.tensor(X_d_tr, dtype=torch.float32).to(device)
    x_h_tr = torch.tensor(X_h_tr, dtype=torch.long).to(device)
    x_l_tr = torch.tensor(X_l_tr, dtype=torch.long).to(device)
    y_tr_t  = torch.tensor(y_tr_np, dtype=torch.float32).to(device)
    edge_tr = edge_tr.to(device)
    x_d_va = torch.tensor(X_d_va, dtype=torch.float32).to(device)
    x_h_va = torch.tensor(X_h_va, dtype=torch.long).to(device)
    x_l_va = torch.tensor(X_l_va, dtype=torch.long).to(device)
    edge_va = edge_va.to(device)
    del X_d_tr, X_h_tr, X_l_tr, X_d_va, X_h_va, X_l_va; gc.collect()

    model = GATFraudModel(
        dense_in_dim=x_d_tr.shape[1],
        high_card_cols=prep.high_card_cols, low_card_cols=prep.low_card_cols,
        factor_cardinalities=prep.factor_cardinalities,
        hidden=cfg.HIDDEN_DIM, heads=cfg.HEADS,
        dropout=cfg.DROPOUT,
        high_card_emb_dim=cfg.HIGH_CARD_EMB_DIM, low_card_emb_dim=cfg.LOW_CARD_EMB_DIM,
    ).to(device)

    use_amp = cfg.USE_AMP and device.type == "cuda"
    model = train_loop(model, x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr,
                       x_d_va, x_h_va, x_l_va, y_va_np, edge_va, cfg, device, verbose)
    model.eval()
    p_tr = safe_inference(model, x_d_tr, x_h_tr, x_l_tr, edge_tr, device, use_amp)
    p_va = safe_inference(model, x_d_va, x_h_va, x_l_va, edge_va, device, use_amp)
    thr  = choose_threshold(y_va_np, p_va, cfg)
    del x_d_tr, x_h_tr, x_l_tr, y_tr_t, edge_tr, x_d_va, x_h_va, x_l_va, edge_va; cleanup()
    return {"model": model.cpu(), "preprocessor": prep, "thr": thr,
             "user_avg_map": user_avg_map, "global_avg": global_avg,
             "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
             "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr)}


# ============================================================================
#  PIPELINE
# ============================================================================
def run_pipeline(df, cfg, graph_strategy="multi_relation"):
    set_seed(cfg.SEED); device = get_device()
    print(f"\n{'#'*60}\n# GATv2 | {graph_strategy}\n{'#'*60}")
    df_dev, df_test = split_groups_holdout(
        df, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    folds = build_group_stratified_folds(
        df_dev, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.N_SPLITS, cfg.STRATIFY_BINS, cfg.SEED)
    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"  Fold {i}/{cfg.N_SPLITS}")
        out = train_one_fold(df_tr, df_va, cfg, graph_strategy, i-1, device)
        print_metrics("TRAIN", out["m_tr"]); print_metrics("VAL", out["m_va"])
        all_out.append(out); cleanup()
    p_ens, y_test, thr = run_test_ensemble(all_out, df_test, cfg, graph_strategy, device)
    m_test = eval_from_probs(y_test, p_ens, thr); print_metrics("TEST", m_test)
    result = {"all_out": all_out, "df_dev": df_dev, "df_test": df_test,
               "thr_global": thr, "test_metrics": m_test,
               "p_test_ens": p_ens, "y_test": y_test,
               "graph_strategy": graph_strategy, "model_arch": MODEL_ARCH}
    save_results_plots(result, cfg)
    return result


def run_all_strategies(df, cfg=None):
    if cfg is None: cfg = IBMFraudConfig()
    results = {}
    for strat in ["multi_relation", "hybrid", "intra_group"]:
        results[f"{strat}_{MODEL_ARCH}"] = run_pipeline(df, cfg, strat); cleanup()
    print_final_comparison(results)
    save_strategy_comparison(results, cfg)
    save_results_csv(results, cfg)
    return results


if __name__ == "__main__":
    cfg = IBMFraudConfig()
    df  = load_and_preprocess(path="/path/to/ibm_transactions.parquet", cfg=cfg)
    results = run_all_strategies(df, cfg)
