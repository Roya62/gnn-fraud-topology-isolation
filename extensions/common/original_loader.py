"""
original_loader.py — import the ORIGINAL, unmodified dataset pipelines.

sparkov/ and ibm/ both define top-level modules called `config` and `utils`,
so they cannot be imported side by side with a plain `import`. This helper
loads one dataset's modules at a time, from the original folders, without
copying or editing them.

Usage:
    orig = load_original("sparkov")
    orig.utils.build_graph_edges(...)
    orig.Config()            # CardFraudConfig or IBMFraudConfig
    orig.gatv2, orig.mlp     # the original model scripts (as modules)
"""
import importlib
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SHARED_NAMES = ["config", "utils", "gatv2_model", "mlp_baseline",
                 "gcn_model", "graphsage_model", "eda"]


def load_original(dataset: str, with_models: bool = True) -> SimpleNamespace:
    dataset = dataset.lower()
    if dataset not in ("sparkov", "ibm"):
        raise ValueError("dataset must be 'sparkov' or 'ibm'")
    folder = os.path.join(REPO_ROOT, dataset)
    for name in _SHARED_NAMES:          # drop the other dataset's modules
        sys.modules.pop(name, None)
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p) not in (os.path.join(REPO_ROOT, "sparkov"),
                                                 os.path.join(REPO_ROOT, "ibm"))]
    sys.path.insert(0, folder)
    config = importlib.import_module("config")
    utils = importlib.import_module("utils")
    ns = SimpleNamespace(
        dataset=dataset, folder=folder, config=config, utils=utils,
        Config=(config.CardFraudConfig if dataset == "sparkov" else config.IBMFraudConfig))
    if with_models:
        ns.gatv2 = importlib.import_module("gatv2_model")
        ns.mlp = importlib.import_module("mlp_baseline")
    return ns
