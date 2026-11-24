"""
run_verification.py — mechanism checks for issues 1–8 (+ issues found during
verification), run against the ORIGINAL, unmodified pipeline code.

Mode A (default): synthetic data in the exact raw schemas. Demonstrates that a
    mechanism exists. The numbers are NOT estimates for Sparkov or IBM.
Mode B: --sparkov-dir / --ibm-path point at the real files. Runs the same
    label-free graph checks on the real preprocessed data (no model training).

Nothing here trains a model on real data or changes any original file.
Outputs: outcomes_extended/verification/<mode>/
    checks.csv           one row per check: issue, metric, value, verdict
    graph_summary.csv    structural summary per dataset × strategy × condition
    feature_auc.csv      single-feature ROC-AUC before/after downsampling (5b)
    verification.json    everything above + environment

Run from the repo root:
    python extensions/verification/run_verification.py                 # synthetic
    python extensions/verification/run_verification.py \
        --sparkov-dir sparkov --ibm-path /path/to/card_transaction.v1.csv  # real
"""
import argparse
import ast
import inspect
import json
import os
import platform
import sys
import tempfile
import textwrap

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.original_loader import load_original, REPO_ROOT  # noqa: E402
from common import graph_metrics as gm  # noqa: E402
from common import synthetic  # noqa: E402

CHECKS, GRAPH_ROWS, FEAT_ROWS = [], [], []


def record(issue, check, metric, value, verdict, note=""):
    CHECKS.append(dict(issue=issue, check=check, metric=metric,
                       value=(round(value, 4) if isinstance(value, float) else value),
                       verdict=verdict, note=note))
    print(f"  [{issue}] {check}: {metric} = {value}  -> {verdict}")


def auc_any_direction(y, x):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if len(np.unique(y[ok])) < 2:
        return float("nan")
    a = roc_auc_score(y[ok], x[ok])
    return max(a, 1 - a)


# =============================================================================
#  STATIC CHECKS (code reading made executable)
# =============================================================================
def static_checks():
    print("\n== Static checks ==")
    ibm = load_original("ibm")
    conv = [n for n in dir(ibm.gatv2) if n in ("GATConv", "GATv2Conv")]
    layer_type = type(ibm.gatv2.GATFraudModel.__init__)
    src = inspect.getsource(ibm.gatv2.GATFraudModel.__init__)
    uses = "GATv2Conv" if "GATv2Conv(" in src else ("GATConv" if "GATConv(" in src else "?")
    record("3", "IBM 'gatv2_model.py' attention layer", "layer class used", uses,
           "CONFIRMED" if uses == "GATConv" else "NOT CONFIRMED",
           f"imported names in module: {conv}")
    spk = load_original("sparkov")
    src_s = inspect.getsource(spk.gatv2.GATv2FraudModel.__init__)
    record("3", "Sparkov 'gatv2_model.py' attention layer", "layer class used",
           "GATv2Conv" if "GATv2Conv(" in src_s else "?", "INFO")
    c_s, c_i = spk.Config(), ibm.Config()
    record("3", "Hidden size / heads", "sparkov (hidden, heads)", f"({c_s.HIDDEN_DIM}, {c_s.HEADS})", "INFO",
           "Conference paper text states 32 hidden, 2 heads for both datasets.")
    record("3", "Hidden size / heads", "ibm (hidden, heads)", f"({c_i.HIDDEN_DIM}, {c_i.HEADS})", "INFO")

    # Issue 4: is `break` reachable only when verbose is True?
    for name, mod in (("ibm", ibm), ("sparkov", spk)):
        tree = ast.parse(textwrap.dedent(inspect.getsource(mod.utils.train_loop)))
        break_under_verbose = False
        break_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "verbose":
                if any(isinstance(b, ast.Break) for b in node.body):
                    break_under_verbose = True
            if isinstance(node, ast.Break):
                break_found = True
        record("4", f"{name} train_loop early stopping", "break only executes when verbose=True",
               break_under_verbose,
               "CONFIRMED (error)" if break_under_verbose else ("OK" if break_found else "no break"))


def dynamic_early_stop_check():
    """Run the ORIGINAL ibm train_loop with a model whose validation AP cannot
    improve after the first evaluation, and count epochs actually trained."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        record("4", "dynamic early-stop test", "skipped", "torch not installed", "SKIPPED")
        return
    ibm = load_original("ibm", with_models=False)
    cfg = ibm.Config(); cfg.USE_AMP = False; cfg.MAX_EPOCHS = 200

    class Constant(nn.Module):
        def __init__(self):
            super().__init__(); self.b = nn.Parameter(torch.zeros(1)); self.calls = 0
        def forward(self, xd, xh, xl, ei):
            if self.training: self.calls += 1
            return self.b.expand(xd.shape[0]) + 0 * xd[:, 0]

    dev = torch.device("cpu")
    n = 200
    xd = torch.randn(n, 3); xh = torch.zeros(n, 0, dtype=torch.long); xl = xh
    y = torch.tensor((np.arange(n) % 5 == 0).astype(np.float32))
    for verbose in (True, False):
        m = Constant()
        ibm.utils.train_loop(m, xd, xh, xl, y, None, xd, xh, xl, y.numpy().astype(int), None,
                             cfg, dev, verbose=verbose)
        record("4", "dynamic early-stop test (ibm)", f"epochs trained, verbose={verbose}", m.calls,
               "INFO", "patience = 12 checks x 5 epochs = 60 epochs expected if early stopping works")
    spk = load_original("sparkov", with_models=False)
    cfg_s = spk.Config(); cfg_s.MAX_EPOCHS = 200

    class ConstantS(nn.Module):
        def __init__(self):
            super().__init__(); self.b = nn.Parameter(torch.zeros(1)); self.calls = 0
        def forward(self, xc, xn, ei):
            if self.training: self.calls += 1
            return self.b.expand(xn.shape[0]) + 0 * xn[:, 0]
    for verbose in (True, False):
        m = ConstantS()
        spk.utils.train_loop(m, {}, xd, y, None, {}, xd, y.numpy().astype(int), None, cfg_s, verbose)
        record("4", "dynamic early-stop test (sparkov)", f"epochs trained, verbose={verbose}", m.calls, "INFO")


# =============================================================================
#  DATA PREPARATION (original functions)
# =============================================================================
def prepare_sparkov(real_dir=None):
    spk = load_original("sparkov", with_models=False)
    cfg = spk.Config()
    if real_dir:
        tr, te = os.path.join(real_dir, "fraudTrain.csv"), os.path.join(real_dir, "fraudTest.csv")
        raw = pd.concat([pd.read_csv(te), pd.read_csv(tr)], ignore_index=True).dropna()
    else:
        raw = synthetic.make_sparkov(seed=0)
        n_f = int(raw.is_fraud.sum()); n_l = len(raw) - n_f
        # scale the original absolute drop count so the published 18.4% fraud rate is reproduced
        cfg.DOWNSAMPLE_NONFRAUD = int(n_l - n_f * (1 - 0.184) / 0.184)
        d = tempfile.mkdtemp()
        tr, te = os.path.join(d, "fraudTrain.csv"), os.path.join(d, "fraudTest.csv")
        raw.iloc[: len(raw) // 2].to_csv(tr, index=False)
        raw.iloc[len(raw) // 2:].to_csv(te, index=False)
    df = spk.utils.load_and_preprocess(train_path=tr, test_path=te, cfg=cfg)
    return spk, cfg, raw, df


def prepare_ibm(real_path=None):
    ibm = load_original("ibm", with_models=False)
    cfg = ibm.Config()
    if real_path:
        path = real_path
    else:
        d = tempfile.mkdtemp(); path = os.path.join(d, "ibm.csv")
        synthetic.make_ibm(seed=0).to_csv(path, index=False)
    df = ibm.utils.load_and_preprocess(path, cfg)
    raw = pd.read_parquet(path) if path.endswith((".parquet", ".pq")) else pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    return ibm, cfg, raw, df


def sparkov_test_graph(spk, cfg, df, strategy, df_dev=None, df_test=None):
    if df_test is None:
        df_dev, df_test = spk.utils.split_groups_holdout(
            df, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    prep = spk.utils.FoldPreprocessor(cfg.CATEGORICAL_COLS, cfg.NUMERICAL_COLS).fit(df_dev)
    enc = prep.transform(df_test)
    cat = enc[cfg.CATEGORICAL_COLS].to_numpy(dtype=np.int64)
    num = enc[prep.num_cols].to_numpy(dtype=np.float32)
    ei = spk.utils.build_graph_edges(strategy, df_test, cat, num, cfg)
    return df_test, ei, cat, num


def ibm_test_graph(ibm, cfg, df, strategy):
    df_dev, df_test = ibm.utils.split_groups_holdout(
        df, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    dev_f, test_f, _, _ = ibm.utils.add_user_spending_features(df_dev, df_test)
    prep = ibm.utils.FoldPreprocessor(cfg).fit(dev_f)
    Xd, Xh, Xl, y = prep.transform(test_f)
    ei = ibm.utils.build_graph_edges(strategy, df_test, Xd, Xh, Xl, cfg)
    return df_test, ei, (Xd, Xh, Xl)


# =============================================================================
#  ISSUE CHECKS ON DATA
# =============================================================================
def issue1_sort_key(spk, cfg, df):
    print("\n== Issue 1: Sparkov sort key ==")
    if "unix_time" not in df.columns:
        record("1", "sort key", "unix_time available", False, "SKIPPED"); return
    t = df["unix_time"].to_numpy()
    sk = spk.utils._make_sort_key(df, cfg.SORT_KEY_COLS)
    # (a) is the sort key monotone in time globally?
    rho = pd.Series(sk).corr(pd.Series(t), method="spearman")
    record("1", "global sort key vs true time", "Spearman rho", float(rho),
           "CONFIRMED (not chronological)" if rho < 0.9 else "chronological")
    # (b) merchant chains: are linked transactions temporal neighbours at that merchant?
    import copy
    c = copy.deepcopy(cfg)
    c.MULTI_REL_SPECS = [s for s in cfg.MULTI_REL_SPECS if s["col"] == "merchant"]
    c.ADD_GLOBAL_TIME_EDGES = False
    edges = spk.utils._build_multi_relation_edges(df, c)
    k = c.MULTI_REL_SPECS[0]["k"]
    rank = df.groupby("merchant")["unix_time"].rank(method="first").to_numpy()
    e = np.array([(a, b) for a, b in edges if a < b])
    gap = np.abs(rank[e[:, 0]] - rank[e[:, 1]])
    same_card = (df["cc_num"].to_numpy()[e[:, 0]] == df["cc_num"].to_numpy()[e[:, 1]])
    sz = df.groupby("merchant")["merchant"].transform("size").to_numpy()
    small = sz[e[:, 0]] <= c.MULTI_REL_SPECS[0]["max_group_size"]
    record("1", "merchant-chain edges", f"% linking chronological neighbours (rank gap <= {k})",
           float(100 * np.mean(gap[small] <= k)), "INFO",
           "100% expected if chains were time-ordered (groups below max_group_size)")
    record("1", "merchant-chain edges", "% linking two transactions of the same card",
           float(100 * same_card.mean()), "INFO")
    # (c) global 'time' edges
    c2 = copy.deepcopy(cfg); c2.MULTI_REL_SPECS = []; c2.ADD_GLOBAL_TIME_EDGES = True
    ge = np.array([(a, b) for a, b in spk.utils._build_multi_relation_edges(df, c2) if a < b])
    cross = df["cc_num"].to_numpy()[ge[:, 0]] != df["cc_num"].to_numpy()[ge[:, 1]]
    hours = np.abs(t[ge[:, 0]] - t[ge[:, 1]]) / 3600
    record("1", "global 'time' edges", "% that join two different cards", float(100 * cross.mean()), "INFO")
    record("1", "global 'time' edges", "% that duplicate within-card chain edges",
           float(100 * (~cross).mean()), "INFO")
    record("1", "global 'time' edges (cross-card)", "median |time gap| in days",
           float(np.median(hours[cross]) / 24) if cross.any() else float("nan"), "INFO")


def graph_structure(dataset, mod, cfg, df, condition):
    print(f"\n== Graph structure: {dataset} / {condition} ==")
    for strat in ["intra_group", "multi_relation", "hybrid"]:
        np.random.seed(cfg.SEED)
        if dataset == "sparkov":
            df_test, ei, _, _ = sparkov_test_graph(mod, cfg, df, strat)
        else:
            df_test, ei, _ = ibm_test_graph(mod, cfg, df, strat)
        y = df_test[cfg.TARGET_COL].to_numpy().astype(int)
        a = gm.to_csr(ei, len(df_test))
        s = gm.summary(a, y)
        groups = gm.hop_group(a, y)
        s.update({f"fraud_{k}": v for k, v in gm.group_counts(groups).items()})
        s.update(dataset=dataset, strategy=strat, condition=condition, split="test")
        GRAPH_ROWS.append(s)
        if strat in ("multi_relation", "hybrid") and dataset == "sparkov":
            record("2", f"{dataset} {strat} test graph ({condition})", "isolated nodes",
                   s["isolated_nodes"], "CONFIRMED (zero isolates)" if s["isolated_nodes"] == 0 else "isolates exist")
        else:
            record("2", f"{dataset} {strat} test graph ({condition})", "isolated nodes / isolated fraud",
                   f"{s['isolated_nodes']} / {s['isolated_fraud']}", "INFO")


def issue5_downsampling_graph(spk, cfg, raw):
    """Same label-free builder; graph built on ALL rows vs on the original
    class-conditionally downsampled rows. Compare what fraud nodes 'see'."""
    print("\n== Issue 5a: downsampling before graph construction ==")
    import copy
    full_cfg = copy.deepcopy(cfg); full_cfg.DOWNSAMPLE_NONFRAUD = 0
    d = tempfile.mkdtemp()
    tr, te = os.path.join(d, "a.csv"), os.path.join(d, "b.csv")
    raw.iloc[: len(raw) // 2].to_csv(tr, index=False); raw.iloc[len(raw) // 2:].to_csv(te, index=False)
    df_full = spk.utils.load_and_preprocess(tr, te, full_cfg)
    graph_structure("sparkov", spk, full_cfg, df_full, "no_downsampling_SYNTHETIC")


def issue5b_feature_artefact(dataset, raw_df, ds_df, cfg):
    """Single-feature ROC-AUC of order-dependent features, computed the
    ORIGINAL way, on all rows vs after class-conditional downsampling."""
    print(f"\n== Issue 5b: features computed after downsampling ({dataset}) ==")
    if dataset != "sparkov":
        return
    full = raw_df.copy()
    full["trans_date_trans_time"] = pd.to_datetime(full["trans_date_trans_time"])
    full = full.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    full["trans_time_diff"] = (full.groupby("cc_num")["trans_date_trans_time"].diff()
                               .dt.total_seconds().div(60).fillna(0))
    full["amt_to_avg"] = full["amt"] / (full.groupby("cc_num")["amt"].transform("mean") + 1e-10)
    for feat in ["trans_time_diff", "amt_to_avg", "amt"]:
        a_full = auc_any_direction(full["is_fraud"].to_numpy(), full[feat].to_numpy())
        a_ds = auc_any_direction(ds_df["is_fraud"].to_numpy(), ds_df[feat].to_numpy())
        FEAT_ROWS.append(dict(dataset=dataset, feature=feat, auc_all_rows=a_full,
                              auc_after_downsampling=a_ds, delta=a_ds - a_full))
        record("5b", f"single-feature ROC-AUC: {feat}", "all rows -> after downsampling",
               f"{a_full:.3f} -> {a_ds:.3f}",
               "INFO" if feat == "amt" else ("CHANGED (up)" if a_ds - a_full > 0.02 else
                                          "CHANGED (down)" if a_ds - a_full < -0.02 else "stable"),
               "'amt' is a row-local control feature: it should not change")
    first = ds_df.groupby("cc_num").cumcount() == 0
    record("5b", "rows whose trans_time_diff is the fill value 0 (first retained row of a card)",
           "% of legit / % of fraud rows",
           f"{100*first[ds_df.is_fraud==0].mean():.1f} / {100*first[ds_df.is_fraud==1].mean():.1f}", "INFO")
    # median time gap by class, after downsampling
    for lab in (0, 1):
        v = ds_df.loc[ds_df.is_fraud == lab, "trans_time_diff"]
        record("5b", "median trans_time_diff (minutes) after downsampling", f"class {lab}",
               float(v.median()), "INFO")


def issue6_future(dataset, df_test, ei, time_values, strat):
    a = gm.to_csr(ei, len(df_test)).tocoo()
    later_to_earlier = time_values[a.row] > time_values[a.col]  # message from row(src) to col(dst)
    deg = np.bincount(a.col, minlength=len(df_test))
    fut_in = np.bincount(a.col[later_to_earlier], minlength=len(df_test))
    has_nb = deg > 0
    record("6", f"{dataset} {strat}: messages from later to earlier transactions",
           "% of directed non-self edges", float(100 * later_to_earlier.mean()), "INFO")
    record("6", f"{dataset} {strat}: nodes receiving >=1 message from the future",
           "% of non-isolated nodes", float(100 * np.mean(fut_in[has_nb] > 0)), "INFO")


def issue7_user_avg(ibm, cfg, df):
    print("\n== Issue 7: IBM user_avg_amount on unseen users ==")
    df_dev, df_test = ibm.utils.split_groups_holdout(
        df, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.TRAIN_RATIO, cfg.STRATIFY_BINS, cfg.SEED)
    overlap = len(set(df_dev.User) & set(df_test.User))
    dev_f, test_f, _, _ = ibm.utils.add_user_spending_features(df_dev, df_test)
    record("7", "users shared by dev and test", "count", overlap, "INFO (0 = proper group split)")
    record("7", "user_avg_amount distinct values", "dev / test",
           f"{dev_f.user_avg_amount.nunique()} / {test_f.user_avg_amount.nunique()}",
           "CONFIRMED (constant on test)" if test_f.user_avg_amount.nunique() == 1 else "varies")
    r = np.corrcoef(test_f.amount_over_user_avg, test_f.Amount)[0, 1]
    record("7", "amount_over_user_avg vs Amount on test", "Pearson r", float(r),
           "CONFIRMED (redundant copy of Amount)" if r > 0.9999 else "INFO")
    folds = ibm.utils.build_group_stratified_folds(
        df_dev, cfg.GROUP_KEY, cfg.TARGET_COL, cfg.N_SPLITS, cfg.STRATIFY_BINS, cfg.SEED)
    tr, va = folds[0]
    _, va_f, _, _ = ibm.utils.add_user_spending_features(tr, va)
    record("7", "user_avg_amount distinct values on validation fold 1", "count",
           int(va_f.user_avg_amount.nunique()), "INFO")


def issue8_schema(dataset, raw):
    print(f"\n== Issue 8 / Exp 7: schema ({dataset}) ==")
    cols = set(raw.columns)
    aml = {"From Bank", "To Bank", "Account", "Account.1", "Is Laundering"}
    tab = {"User", "Card", "Merchant Name", "MCC", "Is Fraud?"}
    sparkov = {"cc_num", "merchant", "is_fraud"}
    kind = ("IBM AML (transfers)" if aml <= cols else "IBM TabFormer (card purchases)" if tab <= cols
            else "Sparkov (card purchases)" if sparkov <= cols else "unknown")
    record("8", f"{dataset} raw schema", "matches", kind, "INFO")
    receiver = [c for c in raw.columns if c.lower() in
                ("account.1", "to account", "to bank", "receiver", "beneficiary", "nameDest".lower())]
    record("8", f"{dataset} receiver-account column present", "columns", receiver or "none",
           "chains reconstructable" if receiver else "NO account-to-account chains possible")
    if "Card" in cols and "User" in cols:
        users_per_card = raw.groupby("Card")["User"].nunique()
        record("10", "IBM 'Card' column semantics", "distinct Card values / max users sharing one Card value",
               f"{raw['Card'].nunique()} / {int(users_per_card.max())}",
               "CONFIRMED (Card is a per-user index)" if users_per_card.max() > 1 else "Card is global ID",
               "On synthetic data this holds by construction; run on the real file to confirm.")


def issue9_similarity_features(dataset, cat_or_parts, num=None):
    """Share of the FAISS/cosine feature vector norm owned by raw integer
    category codes, and neighbour overlap with/without them."""
    from sklearn.neighbors import NearestNeighbors
    if dataset == "sparkov":
        cat = cat_or_parts.astype("float32"); dense = num.astype("float32")
    else:
        Xd, Xh, Xl = cat_or_parts
        cat = np.hstack([Xh, Xl]).astype("float32"); dense = Xd.astype("float32")
    full = np.hstack([cat, dense])
    share = float(np.mean((cat ** 2).sum(1) / np.maximum((full ** 2).sum(1), 1e-12)))
    record("9", f"{dataset} similarity-edge features", "mean share of squared norm from integer category codes",
           100 * share, "CONFIRMED (codes dominate)" if share > 0.8 else "INFO")
    k = 6
    def knn(X):
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    nf, nc, nd = knn(full), knn(cat), knn(dense)
    ov = lambda A, B: float(np.mean([len(set(a) & set(b)) / k for a, b in zip(A, B)]))
    record("9", f"{dataset} kNN overlap (k=6)", "full vs category-codes-only", 100 * ov(nf, nc), "INFO")
    record("9", f"{dataset} kNN overlap (k=6)", "full vs numeric-only", 100 * ov(nf, nd), "INFO")


def issue11_intra_group_randomness(dataset, mod, cfg, df):
    eis = []
    for seed in (1, 2):
        np.random.seed(seed)
        if dataset == "sparkov":
            _, ei, _, _ = sparkov_test_graph(mod, cfg, df, "intra_group")
        else:
            _, ei, _ = ibm_test_graph(mod, cfg, df, "intra_group")
        e = ei.numpy(); eis.append(set(map(tuple, e[:, e[0] != e[1]].T.tolist())))
    j = len(eis[0] & eis[1]) / max(1, len(eis[0] | eis[1]))
    record("11", f"{dataset} intra_group test graph, two RNG states", "edge Jaccard similarity", float(j),
           "INFO (1.0 = identical graph for every fold model)")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparkov-dir", default=None)
    ap.add_argument("--ibm-path", default=None)
    args = ap.parse_args()
    mode = "real" if (args.sparkov_dir or args.ibm_path) else "synthetic"
    out = os.path.join(REPO_ROOT, "outcomes_extended", "verification", mode)
    os.makedirs(out, exist_ok=True)
    print(f"MODE: {mode}  ->  {out}")

    static_checks()
    dynamic_early_stop_check()

    if mode == "synthetic" or args.sparkov_dir:
        spk, cfg_s, raw_s, df_s = prepare_sparkov(args.sparkov_dir)
        issue1_sort_key(spk, cfg_s, df_s)
        graph_structure("sparkov", spk, cfg_s, df_s, "original_protocol")
        if mode == "synthetic":
            issue5_downsampling_graph(spk, cfg_s, raw_s)
        issue5b_feature_artefact("sparkov", raw_s, df_s, cfg_s)
        for strat in ["intra_group", "multi_relation", "hybrid"]:
            np.random.seed(cfg_s.SEED)
            df_t, ei, cat, num = sparkov_test_graph(spk, cfg_s, df_s, strat)
            issue6_future("sparkov", df_t, ei, df_t["unix_time"].to_numpy(), strat)
            if strat == "hybrid":
                issue9_similarity_features("sparkov", cat, num)
        issue8_schema("sparkov", raw_s)
        issue11_intra_group_randomness("sparkov", spk, cfg_s, df_s)

    if mode == "synthetic" or args.ibm_path:
        ibm, cfg_i, raw_i, df_i = prepare_ibm(args.ibm_path)
        graph_structure("ibm", ibm, cfg_i, df_i, "original_protocol")
        issue7_user_avg(ibm, cfg_i, df_i)
        tval = df_i["transaction_dt"].astype("int64").to_numpy()
        for strat in ["intra_group", "multi_relation", "hybrid"]:
            np.random.seed(cfg_i.SEED)
            df_t, ei, parts = ibm_test_graph(ibm, cfg_i, df_i, strat)
            issue6_future("ibm", df_t, ei, df_t["transaction_dt"].astype("int64").to_numpy(), strat)
            if strat == "hybrid":
                issue9_similarity_features("ibm", parts)
            if strat == "multi_relation":
                # Card-relation edges that join different users (issue 10)
                import copy
                c = copy.deepcopy(cfg_i); c.MULTI_REL_SPECS = [s for s in cfg_i.MULTI_REL_SPECS if s["col"] == "Card"]
                ce = np.array([(a, b) for a, b in ibm.utils._build_multi_relation_edges(df_t, c) if a < b])
                if len(ce):
                    u = df_t["User"].to_numpy()
                    record("10", "IBM 'Card' relation edges (test graph)", "% joining two different Users",
                           float(100 * np.mean(u[ce[:, 0]] != u[ce[:, 1]])), "INFO")
        np.random.seed(cfg_i.SEED)
        df_t, ei, _ = ibm_test_graph(ibm, cfg_i, df_i, "intra_group")
        a = gm.to_csr(ei, len(df_t)).tocoo()
        u = df_t["User"].to_numpy()
        record("10", "IBM intra_group edges (test graph)", "% joining two different Users",
               float(100 * np.mean(u[a.row] != u[a.col])) if a.nnz else 0.0, "INFO",
               "Paper: 'No edges cross cardholder boundaries.'")
        issue8_schema("ibm", raw_i)
        issue11_intra_group_randomness("ibm", ibm, cfg_i, df_i)

    pd.DataFrame(CHECKS).to_csv(os.path.join(out, "checks.csv"), index=False)
    pd.DataFrame(GRAPH_ROWS).to_csv(os.path.join(out, "graph_summary.csv"), index=False)
    pd.DataFrame(FEAT_ROWS).to_csv(os.path.join(out, "feature_auc.csv"), index=False)
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    try:
        import torch, torch_geometric
        env.update(torch=torch.__version__, torch_geometric=torch_geometric.__version__)
    except ImportError:
        env.update(torch="not installed")
    json.dump(dict(mode=mode, env=env, checks=CHECKS, graphs=GRAPH_ROWS, features=FEAT_ROWS),
              open(os.path.join(out, "verification.json"), "w"), indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
