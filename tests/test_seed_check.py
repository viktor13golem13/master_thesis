import numpy as np
import pandas as pd

from thesis.grid_search import all_graph_configs, centrality_scores, prepare_search_data
from thesis.seed_check import seed_spread


def small_search_data():
    dates = pd.bdate_range("2017-01-02", "2019-12-31")
    rng = np.random.default_rng(0)
    market = rng.normal(size=(len(dates), 1))
    returns = pd.DataFrame((market @ rng.random((1, 30)) + rng.normal(size=(len(dates), 30))) / 100, index=dates)
    rf = pd.Series(0.0001, index=dates)
    return prepare_search_data(returns, rf, [2017, 2018], processes=1)


def strategy_row(graph_config, centrality, alpha, selection, m):
    t = graph_config.graph_type
    return {"matrix": graph_config.matrix, "weighting": graph_config.weighting, "shrink": graph_config.shrink,
            "sign": t.sign, "loops": t.loops, "weighted": t.weighted, "threshold": graph_config.threshold,
            "centrality": centrality, "alpha": alpha, "selection": selection, "m": m}


def find_config(**settings):
    for g in all_graph_configs():
        if all(getattr(g, k, getattr(g.graph_type, k, None)) == v for k, v in settings.items()):
            return g


def test_seeds_change_nothing_without_ties_and_something_with_them():
    data = small_search_data()
    dense = find_config(matrix="correlation", weighting="uniform", shrink=False, sign="absolute",
                        loops=False, weighted=True, threshold=0.0)
    sparse = find_config(matrix="correlation", weighting="uniform", shrink=False, sign="absolute",
                         loops=False, weighted=False, threshold=0.9)
    assert centrality_scores(sparse, data)[1] > 0.5                  # most bonds isolated: many ties
    strategies = pd.DataFrame([
        strategy_row(dense, "degree", np.nan, "peripheral", 10),
        strategy_row(sparse, "degree", np.nan, "peripheral", 10),
    ])
    spread = seed_spread(strategies, data, n_seeds=5, processes=1).groupby("strategy").SR_Lo.std()
    assert spread[0] < 1e-12
    assert spread[1] > 0
