"""
How much would a strategy's result depend on luck if ties were broken randomly?

On sparse graphs the m most peripheral bonds can be a draw among many tied
bonds (e.g. isolated nodes). The grid search avoids luck by sharing the
tied places among all tied bonds (backtest.select_weights). This module
shows what random tie-breaking (centrality.rank_by_centrality) would have
given instead: each strategy is re-evaluated with n_seeds random
tie-breaks, which can be compared with the shared-places result.

The centrality scores do not depend on the seed, so they are computed once
per graph configuration and only the ranking is redrawn.
"""
import numpy as np
import pandas as pd

from thesis import config
from thesis.backtest import evaluate_portfolio, select_bonds
from thesis.centrality import rank_by_centrality
from thesis.graph import GraphType
from thesis.grid_search import GraphConfig, SearchData, build_graph
from thesis.parallel import parallel_map


def seed_spread(strategies: pd.DataFrame, data: SearchData, n_seeds: int,
                processes: int | None = None) -> pd.DataFrame:
    """
    The rank metric (config.RANK_BY) of each strategy under n_seeds new
    tie-breaking seeds.

    Parameters
    ----------
    strategies
        Rows of the grid-search results (graph strategies only).

    Returns
    -------
    One row per strategy and seed: the strategy's index in `strategies`,
    the seed number (1, ..., n_seeds) and every metric.
    """
    tasks = [(graph_config_of(row), group) for row, group in _by_graph_config(strategies)]
    rows = []
    for result in parallel_map(_seed_task, tasks, (data, n_seeds), processes):
        rows.extend(result)
    return pd.DataFrame(rows).sort_values(["strategy", "seed"], ignore_index=True)


def graph_config_of(row) -> GraphConfig:
    """The graph configuration of one results row."""
    return GraphConfig(row.matrix, row.weighting, bool(row.shrink),
                       GraphType(row.sign, bool(row.loops), bool(row.weighted)), float(row.threshold))


def _by_graph_config(strategies: pd.DataFrame):
    """(first row, rows) for each graph configuration among the strategies."""
    keys = ["matrix", "weighting", "shrink", "sign", "loops", "weighted", "threshold"]
    for _, group in strategies.groupby(keys, sort=False):
        yield next(group.itertuples()), group


def _seed_task(task, shared):
    graph_config, group = task
    data, n_seeds = shared
    centralities = {(f.__name__, p): (f, p) for f, p in config.CENTRALITIES}
    wanted = {_centrality_key(row) for row in group.itertuples()}

    # scores[key][year]: centrality scores, which do not depend on the seed
    scores = {key: {} for key in wanted}
    for year in data.formation_years:
        graph = build_graph(graph_config, data, year)
        for key in wanted:
            function, parameter = centralities[key]
            scores[key][year] = function(graph, parameter)

    rows = []
    for row in group.itertuples():
        key = _centrality_key(row)
        for seed in range(1, n_seeds + 1):
            holdings = {}
            for year in data.formation_years:
                rng = np.random.default_rng([graph_config.seed(), year, seed])
                ranking = rank_by_centrality(scores[key][year], rng)
                holdings[year + 1] = select_bonds(ranking, row.selection, row.m)
            metrics = evaluate_portfolio(data.returns_by_year, data.rf_by_year, holdings)
            rows.append({"strategy": row.Index, "seed": seed, **metrics})
    return rows


def _centrality_key(row) -> tuple:
    """(centrality name, parameter) as in config.CENTRALITIES; parameter-free measures are NaN in the results."""
    parameter = None if pd.isna(row.alpha) else float(row.alpha)
    return row.centrality, parameter
