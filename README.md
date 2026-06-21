# gnn-fraud-topology-isolation
# Isolating Graph Topology from Model Architecture in GNN-Based Fraud Detection

Code accompanying the paper *"Isolating Graph Topology from Model Architecture in
GNN-Based Fraud Detection: An Empirical Framework"* (Amiri & Jaf, 2026).

This repository implements a modular framework that isolates **graph topology**
as the sole experimental variable in GNN-based credit card fraud detection.
Three graph construction strategies — multi-relation temporal, hybrid
structural-similarity, and intra-group — are evaluated as pluggable components
under a fixed architecture, leakage-safe evaluation protocol, and a
feature-identical MLP baseline, on two public datasets (Sparkov and IBM).

## Repository structure

```
.
├── src/
│   ├── ibm_pipeline.py          # IBM: 3 graph strategies x 3 architectures
│   │                             # (GATv2 / GCN / GraphSAGE) + MLP baseline
│   │                             # -> reproduces Tables 4, 5, 6
│   ├── ibm_edge_ablation.py     # IBM: intra-group edge-type ablation
│   │                             # (temporal-only / similarity-only /
│   │                             #  merchant-only / full)
│   └── sparkov_pipeline.py      # Sparkov: 3 graph strategies (GATv2) +
│                                 # intra-group edge-type ablation
│                                 # -> reproduces Tables 2, 3
├── requirements.txt
├── LICENSE
└── README.md
```

## Graph construction strategies

| Strategy | Description |
|---|---|
| `multi_relation` | Temporal edges within groups sharing a relational attribute (card, user, merchant, MCC / category, zip), plus optional global-time edges. |
| `hybrid` | `multi_relation` edges + FAISS approximate-nearest-neighbour feature-similarity edges, merged into one homogeneous graph. |
| `intra_group` | Edges scoped entirely within a single cardholder: temporal chains + cosine-similarity edges + merchant sub-relation chains. No cross-cardholder edges. |
| `mlp_baseline` | No graph; identical features, loss, splits, and threshold-tuning protocol as the GNN runs — the no-graph empirical anchor. |

All strategies share identical: feature engineering (fit on training folds
only), model capacity, training loop (AdamW + CosineAnnealingWarmRestarts),
group-stratified 5-fold cross-validation, max-F1 threshold tuning on pooled
validation predictions, and 5-fold ensemble averaging at test time. **Graph
topology is the only variable that changes between strategies.**

## Leakage-safety protocol

- **Group-aware holdout split**: all transactions from a given cardholder
  appear exclusively in either the train/dev set or the held-out test set,
  never both (`split_groups_holdout`).
- **Group-stratified 5-fold CV** within the dev set, stratified by per-group
  fraud prevalence.
- **Train-only feature fitting**: all normalisation statistics, label
  encoders, and aggregate features (e.g. amount relative to cardholder
  average) are fit on the training fold only and applied to validation/test
  using training-derived statistics (`FoldPreprocessor`,
  `add_user_spending_features`).
- **Train-only FAISS index**: for the `hybrid` strategy, the similarity index
  is built exclusively from training-set features; test nodes query it
  inductively.

## Datasets

This code expects two public datasets (not included in this repository):

- **Sparkov** (simulated credit card transactions):
  [kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
  — `fraudTrain.csv` + `fraudTest.csv`
- **IBM** (synthetic credit card transactions for AML/fraud research):
  [kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

Download both and place them under a local `data/` directory (already
excluded via `.gitignore`), then update the file paths at the bottom of each
script (`if __name__ == "__main__":` block) to point to your local copies.

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for practical runtimes (mixed-precision training
is used where available); CPU execution is supported but slow on the larger
IBM dataset.

## Usage

**IBM — full strategy x architecture comparison (Tables 4, 5, 6):**
```bash
python src/ibm_pipeline.py
```
Runs all three graph strategies under GATv2, GCN, and GraphSAGE, plus the MLP
baseline, and prints the final comparison table and architecture-consistency
check.

**IBM — intra-group edge-type ablation:**
```bash
python src/ibm_edge_ablation.py
```

**Sparkov — strategy comparison and edge-type ablation (Tables 2, 3):**
```bash
python src/sparkov_pipeline.py
```

Each script exposes its pipeline functions (`run_pipeline`,
`run_all_strategies`, etc.) for use in a notebook if you'd rather run
individual configurations interactively than via the `__main__` block.

## Headline results

| Dataset | Strategy | F1 | AUC |
|---|---|---|---|
| Sparkov | Multi-relation | 0.908 | 0.994 |
| Sparkov | Hybrid | 0.861 | 0.982 |
| Sparkov | Intra-group | 0.904 | 0.992 |
| Sparkov | MLP baseline | 0.811 | 0.972 |
| IBM | Multi-relation | 0.828 | 0.984 |
| IBM | Hybrid | 0.816 | 0.984 |
| IBM | Intra-group | 0.771 | 0.962 |
| IBM | MLP baseline | 0.833 | 0.976 |

See the paper for the full architecture-sensitivity comparison (GATv2 vs.
GCN vs. GraphSAGE) and discussion.

## Citation

If you use this code, please cite:

```bibtex
@article{amiri2026isolating,
  title   = {Isolating Graph Topology from Model Architecture in GNN-Based Fraud Detection: An Empirical Framework},
  author  = {Amiri, Roya and Jaf, Sardar},
  journal = {Big Data and Cognitive Computing},
  year    = {2026}
}
```

## License

MIT License — see [LICENSE](LICENSE).
