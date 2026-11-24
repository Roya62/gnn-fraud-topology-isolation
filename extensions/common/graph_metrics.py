"""
graph_metrics.py — label-free structure metrics + post-hoc label analysis
on an edge_index produced by the ORIGINAL build_graph_edges().

Conventions
-----------
* Self-loops are removed before every structural metric. A node whose only
  edge is its self-loop is "isolated".
* Edges are treated as undirected (the original builders always add both
  directions; we symmetrise defensively).
* Labels are used only AFTER graph construction, for description.
"""
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def to_csr(edge_index, n: int) -> sp.csr_matrix:
    ei = edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.asarray(edge_index)
    src, dst = ei[0], ei[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    a = sp.coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.int8)
    a.eliminate_zeros()
    return a


def degree(a: sp.csr_matrix) -> np.ndarray:
    return np.diff(a.indptr)


def summary(a: sp.csr_matrix, y: np.ndarray) -> dict:
    n = a.shape[0]
    deg = degree(a)
    iso = deg == 0
    fraud = y.astype(bool)
    n_comp, lab = connected_components(a, directed=False)
    sizes = np.bincount(lab)
    und = sp.triu(a, k=1).tocoo()
    yr, yc = y[und.row], y[und.col]
    ff = int(np.sum((yr == 1) & (yc == 1)))
    ll = int(np.sum((yr == 0) & (yc == 0)))
    fl = int(len(yr) - ff - ll)
    m = len(yr)
    return {
        "n_nodes": n, "n_undirected_edges": m,
        "deg_min": int(deg.min()), "deg_median": float(np.median(deg)),
        "deg_mean": float(deg.mean()), "deg_p95": float(np.percentile(deg, 95)),
        "deg_max": int(deg.max()),
        "isolated_nodes": int(iso.sum()), "isolated_pct": 100 * iso.mean(),
        "fraud_nodes": int(fraud.sum()),
        "isolated_fraud": int((iso & fraud).sum()),
        "isolated_fraud_pct_of_fraud": 100 * (iso & fraud).sum() / max(1, fraud.sum()),
        "n_components": int(n_comp), "largest_component": int(sizes.max()),
        "largest_component_pct": 100 * sizes.max() / n,
        "singleton_components": int(np.sum(sizes == 1)),
        "edges_fraud_fraud": ff, "edges_fraud_legit": fl, "edges_legit_legit": ll,
        "edges_ff_pct": 100 * ff / max(1, m), "edges_fl_pct": 100 * fl / max(1, m),
        "edges_ll_pct": 100 * ll / max(1, m),
        # fraud share among neighbours of fraud vs legit nodes (homophily view)
        "fraud_nbr_share_of_fraud_nodes": _mean_nbr_label(a, y, fraud),
        "fraud_nbr_share_of_legit_nodes": _mean_nbr_label(a, y, ~fraud),
    }


def _mean_nbr_label(a, y, mask):
    deg = degree(a)
    s = a @ y.astype(np.float64)
    ok = mask & (deg > 0)
    return float(np.mean(s[ok] / deg[ok])) if ok.any() else float("nan")


def nearest_other_fraud_distance(a: sp.csr_matrix, y: np.ndarray, max_hops: int = 3) -> np.ndarray:
    """For every node with y==1: hop distance to the nearest OTHER fraud node,
    searched up to `max_hops`. Returns float array (len n): 1..max_hops,
    np.inf if none within max_hops, np.nan for non-fraud nodes.
    Truncated BFS per fraud node; exact."""
    n = a.shape[0]
    out = np.full(n, np.nan)
    fraud = y.astype(bool)
    indptr, indices = a.indptr, a.indices
    for v in np.flatnonzero(fraud):
        seen = {int(v)}
        frontier = np.array([v])
        found = np.inf
        for h in range(1, max_hops + 1):
            if frontier.size == 0:
                break
            nb = np.unique(np.concatenate([indices[indptr[u]:indptr[u + 1]] for u in frontier]))
            nb = np.array([w for w in nb if w not in seen], dtype=np.int64)
            if nb.size == 0:
                break
            if fraud[nb].any():
                found = h
                break
            seen.update(nb.tolist())
            frontier = nb
        out[v] = found
    return out


def hop_group(a: sp.csr_matrix, y: np.ndarray, max_hops: int = 3) -> np.ndarray:
    """Mutually exclusive groups for fraud nodes (Exp 2 definition):
    'isolated' | 'hop1' | 'hop2' | 'hop3' | 'connected_none_within3'.
    Non-fraud nodes -> ''."""
    deg = degree(a)
    d = nearest_other_fraud_distance(a, y, max_hops)
    g = np.full(a.shape[0], "", dtype=object)
    f = y.astype(bool)
    g[f & (deg == 0)] = "isolated"
    for h in range(1, max_hops + 1):
        g[f & (deg > 0) & (d == h)] = f"hop{h}"
    g[f & (deg > 0) & np.isinf(d)] = f"connected_none_within{max_hops}"
    return g


def group_counts(groups: np.ndarray) -> dict:
    c = Counter(groups[groups != ""])
    return {k: int(c.get(k, 0)) for k in
            ["isolated", "connected_none_within3", "hop1", "hop2", "hop3"]}
