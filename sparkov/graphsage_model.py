"""
graphsage_model.py — GraphSAGE fraud detector for the Sparkov dataset.

Graph strategies: multi_relation | hybrid | intra_group
Architecture:     SAGEConv, 2 layers, mean aggregation, LayerNorm

Run:
    python graphsage_model.py

Outputs saved to ../outcomes/
"""

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from config import CardFraudConfig
from utils import (
    set_seed, get_device, cleanup,
    load_and_preprocess,
    FoldPreprocessor, split_groups_holdout, build_stratified_folds,
    build_graph_edges, prepare_tensors,
    train_loop, run_test_ensemble,
    eval_from_probs, print_metrics, choose_threshold,
    save_results_plots, save_strategy_comparison,
    save_results_csv, print_final_comparison,
)

MODEL_ARCH = "graphsage"


# ============================================================================
#  MODEL
# ============================================================================
class GraphSAGEFraudModel(nn.Module):
    """
    Two-layer SAGEConv node classifier with mean neighbourhood aggregation.
    GraphSAGE (Hamilton et al., 2017) samples and aggregates neighbour features
    inductively, making it well-suited to datasets where cardholders in the
    test set may have unseen neighbours.

    Architecture:
        embeddings + numerics
        → SAGEConv(in, 256, aggr=mean) → LayerNorm → LeakyReLU → Dropout
        → SAGEConv(256, 256, aggr=mean) → LayerNorm → LeakyReLU
        → Linear(256→128) → LeakyReLU → Dropout → Linear(128→1)
    """

    def __init__(self, cardinalities, cat_cols, num_input_dim,
                 embedding_dim=8, hidden=256, dropout=0.30):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.dropout  = dropout
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinalities[col] + 1, embedding_dim)
            for col in cat_cols})
        in_ch = len(cat_cols) * embedding_dim + num_input_dim
        self.sage1 = SAGEConv(in_ch, hidden, aggr="mean")
        self.sage2 = SAGEConv(hidden, hidden, aggr="mean")
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.cls   = nn.Sequential(
            nn.Linear(hidden, 128), nn.LeakyReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x_cat_dict, x_num, edge_index):
        cat_embs = [self.embeddings[col](x_cat_dict[col]) for col in self.cat_cols]
        x = F.dropout(torch.cat([torch.cat(cat_embs, dim=1), x_num], dim=1),
                      p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm1(self.sage1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.norm2(self.sage2(h, edge_index)))
        return self.cls(h).view(-1)


# ============================================================================
#  FOLD TRAINING
# ============================================================================
def train_one_fold(df_tr, df_va, cfg, graph_strategy, fold_idx=0, device=None):
    if device is None: device = get_device()
    prep = FoldPreprocessor(cfg.CATEGORICAL_COLS, cfg.NUMERICAL_COLS)
    prep.fit(df_tr)
    df_tr_enc = prep.transform(df_tr)
    df_va_enc = prep.transform(df_va)

    cat_tr, num_tr, y_tr_np, cat_tr_d, num_tr_t = prepare_tensors(
        df_tr_enc, cfg.CATEGORICAL_COLS, prep.num_cols, cfg.TARGET_COL, device)
    cat_va, num_va, y_va_np, cat_va_d, num_va_t = prepare_tensors(
        df_va_enc, cfg.CATEGORICAL_COLS, prep.num_cols, cfg.TARGET_COL, device)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float).to(device)

    verbose = (fold_idx == 0)
    edge_tr = build_graph_edges(graph_strategy, df_tr, cat_tr, num_tr, cfg, verbose).to(device)
    edge_va = build_graph_edges(graph_strategy, df_va, cat_va, num_va, cfg, verbose).to(device)

    model = GraphSAGEFraudModel(
        cardinalities=prep.cardinalities, cat_cols=cfg.CATEGORICAL_COLS,
        num_input_dim=num_tr.shape[1], embedding_dim=cfg.EMBEDDING_DIM,
        hidden=cfg.SAGE_HIDDEN_DIM, dropout=cfg.DROPOUT).to(device)

    model = train_loop(model, cat_tr_d, num_tr_t, y_tr_t, edge_tr,
                       cat_va_d, num_va_t, y_va_np, edge_va, cfg, verbose)
    model.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(model(cat_tr_d, num_tr_t, edge_tr)).cpu().numpy()
        p_va = torch.sigmoid(model(cat_va_d, num_va_t, edge_va)).cpu().numpy()
    thr = choose_threshold(y_va_np, p_va, cfg)
    cleanup()
    return {"model": model, "preprocessor": prep, "thr": thr,
            "y_tr": y_tr_np, "p_tr": p_tr, "m_tr": eval_from_probs(y_tr_np, p_tr, thr),
            "y_va": y_va_np, "p_va": p_va, "m_va": eval_from_probs(y_va_np, p_va, thr)}


# ============================================================================
#  PIPELINE
# ============================================================================
def run_pipeline(df, cfg, graph_strategy="multi_relation"):
    set_seed(cfg.SEED); device = get_device()
    print(f"\n{'#'*60}\n# GraphSAGE | {graph_strategy}\n{'#'*60}")
    df_dev, df_test = split_groups_holdout(
        df, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    folds = build_stratified_folds(df_dev, cfg.TARGET_COL, cfg.N_SPLITS,
                                   cfg.GROUP_KEY, cfg.STRATIFY_BINS, cfg.SEED)
    all_out = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        print(f"  Fold {i}/{cfg.N_SPLITS}")
        out = train_one_fold(df_tr, df_va, cfg, graph_strategy, i - 1, device)
        print_metrics("TRAIN", out["m_tr"]); print_metrics("VAL", out["m_va"])
        all_out.append(out); cleanup()
    p_ens, y_test, thr = run_test_ensemble(all_out, df_test, cfg, graph_strategy, device)
    m_test = eval_from_probs(y_test, p_ens, thr)
    print_metrics("TEST", m_test)
    result = {"all_out": all_out, "df_dev": df_dev, "df_test": df_test,
              "thr_global": thr, "test_metrics": m_test,
              "p_test_ens": p_ens, "y_test": y_test,
              "graph_strategy": graph_strategy, "model_arch": MODEL_ARCH}
    save_results_plots(result, cfg)
    return result


def run_all_strategies(df, cfg=None):
    if cfg is None: cfg = CardFraudConfig()
    results = {}
    for strat in ["multi_relation", "hybrid", "intra_group"]:
        results[f"{strat}_{MODEL_ARCH}"] = run_pipeline(df, cfg, strat)
        cleanup()
    print_final_comparison(results)
    save_strategy_comparison(results, cfg)
    save_results_csv(results, cfg)
    return results


if __name__ == "__main__":
    cfg = CardFraudConfig()
    df  = load_and_preprocess(train_path="fraudTrain.csv", test_path="fraudTest.csv", cfg=cfg)
    results = run_all_strategies(df, cfg)
