# Extensions — connectivity, multi-hop and failure analysis

Branch: `research/connectivity-extensions`. **Nothing outside `extensions/` and
`outcomes_extended/` is modified.** The original pipelines in `sparkov/` and
`ibm/`, their configuration and `outcomes/` stay exactly as published.

## Three result tiers (never mixed)

| Tier | Where | Meaning |
|---|---|---|
| Published | the paper / `outcomes/` | Numbers as published. Never regenerated or overwritten here. |
| Reproduced | `outcomes_extended/reproduced/` | Original code + original config, rerun to recover per-transaction predictions. |
| Revised | `outcomes_extended/sensitivity/<S-id>/` | One identified issue corrected at a time (sensitivity experiments). |

## Status

| Step | Script | Status |
|---|---|---|
| Verification of issues 1–11 | `verification/run_verification.py` | done (synthetic); real-data run pending |
| Exp 1–7 and sensitivity runs | — | not started; awaiting discussion of the verification report |

## Verification

```bash
pip install -r extensions/requirements.txt
# A. Mechanism checks on synthetic data in the original schemas (CPU, < 1 min)
python extensions/verification/run_verification.py
# B. Same label-free checks on the real files (no model training; CPU, minutes)
python extensions/verification/run_verification.py \
    --sparkov-dir sparkov \
    --ibm-path /path/to/the/exact/ibm/file/used/in/the/paper
```

Outputs go to `outcomes_extended/verification/{synthetic,real}/`:
`checks.csv` (one row per check, with verdict), `graph_summary.csv`
(structure per dataset × strategy), `feature_auc.csv`, `verification.json`.

**Synthetic numbers demonstrate mechanisms only.** They are not estimates for
Sparkov or IBM and must not be quoted as results.

## Layout

```
extensions/
  common/original_loader.py  imports the unmodified sparkov/ or ibm/ modules (both use the names config/utils)
  common/graph_metrics.py    degree, isolates, components, edge label mix, nearest-other-fraud hop distance, Exp 2 groups
  common/synthetic.py        small schema-identical synthetic data for code-path tests
  verification/run_verification.py
```
