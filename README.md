# GNN-Based Credit Card Fraud Detection

Roya Amiri & Sardar Jaf — University of Sunderland, 2026

Paper: *Isolating Graph Topology from Model Architecture in GNN-Based Fraud Detection: An Empirical Framework*

---

## Project Structure

```
fraud_gnn_project/
├── sparkov/               # Sparkov dataset pipeline
│   ├── config.py          # All hyperparameters and dataset schema
│   ├── utils.py           # Shared data loading, graph builders, training loop, plotting
│   ├── gatv2_model.py     # GATv2 model — run all 3 strategies
│   ├── gcn_model.py       # GCN model — run all 3 strategies
│   ├── graphsage_model.py # GraphSAGE model — run all 3 strategies
│   └── mlp_baseline.py    # MLP baseline (no graph)
│
├── ibm/                   # IBM dataset pipeline
│   ├── config.py
│   ├── utils.py
│   ├── gatv2_model.py
│   ├── gcn_model.py
│   ├── graphsage_model.py
│   └── mlp_baseline.py
│
└── outcomes/              # All plots and CSVs are saved here automatically
```

---

## Graph Construction Strategies

| Strategy | Description |
|---|---|
| `multi_relation` | Temporal edges along card, merchant, category, zip/MCC |
| `hybrid` | multi_relation + FAISS k-NN feature-similarity edges |
| `intra_group` | Intra-cardholder temporal + similarity + merchant chains |
| `mlp_baseline` | No graph — controlled ablation anchor |

---

## Datasets

- **Sparkov**: https://www.kaggle.com/datasets/kartik2112/fraud-detection
  - Place `fraudTrain.csv` and `fraudTest.csv` in `sparkov/`
- **IBM**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
  - Update `path=` in `ibm/gatv2_model.py` (and other IBM files) to your local file

---

## Installation

```bash
pip install torch torch-geometric scikit-learn pandas numpy faiss-cpu matplotlib pyarrow
```

---

## Usage

```bash
# Sparkov — run one model across all 3 strategies
cd sparkov
python gatv2_model.py
python gcn_model.py
python graphsage_model.py
python mlp_baseline.py

# IBM — same pattern
cd ibm
python gatv2_model.py
```

All diagnostic plots (confusion matrix, ROC, PR curve, threshold tuning) and
A results CSV are saved automatically to `../outcomes/` after each run.

---

## Outcomes Folder

After running all models, `outcomes/` will contain:

```
outcomes/
├── sparkov_gatv2_multi_relation_diagnostics.png
├── sparkov_gatv2_hybrid_diagnostics.png
├── sparkov_gatv2_intra_group_diagnostics.png
├── sparkov_gcn_*.png
├── sparkov_graphsage_*.png
├── sparkov_mlp_mlp_baseline_diagnostics.png
├── sparkov_strategy_comparison.png
├── sparkov_results_summary.csv
├── ibm_gatv2_*.png
├── ibm_gcn_*.png
├── ibm_graphsage_*.png
├── ibm_mlp_*.png
├── ibm_strategy_comparison.png
└── ibm_results_summary.csv
```
