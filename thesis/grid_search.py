"""
Step 6 — Try every strategy (paper, Sec. 4).

A strategy is one choice of every setting:

    graph configuration   window weighting, shrinkage, graph type, threshold θ
    centrality            measure and its parameter
    selection             central or peripheral bonds
    portfolio size        m

A graph configuration is the unit of work: for every formation year it
builds one graph, ranks the bonds by each centrality, and then evaluates
every (centrality, selection, m) strategy on that ranking. The possible
values of each setting are listed in thesis.config.
"""
import itertools
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from thesis import config
from thesis.backtest import evaluate_portfolio, select_bonds, split_by_year
from thesis.centrality import rank_by_centrality
from thesis.correlation import correlation_for_year
from thesis.graph import Graph, GraphType, adjacency_matrix
from thesis.metrics import LOWER_IS_BETTER, metric_name
from thesis.parallel import parallel_map


@dataclass(frozen=True)
class GraphConfig:
    weighting: str
    shrink: bool
    graph_type: GraphType
    threshold: float

    def seed(self) -> int:
        """Fixed random seed for breaking ties, so every run gives the same results."""
        return zlib.crc32(repr(self).encode())


def all_graph_configs() -> list[GraphConfig]:
    """Every combination of the graph settings in config."""
    return [
        GraphConfig(weighting, shrink, graph_type, threshold)
        for weighting, shrink, graph_type, threshold in itertools.product(
            config.WINDOW_WEIGHTINGS, config.SHRINKAGE_OPTIONS, config.GRAPH_TYPES, config.THRESHOLDS,
        )
    ]


@dataclass
class SearchData:
    """What evaluating a graph configuration needs, shared by all workers."""
    correlations: dict        # (weighting, shrink, year) -> correlation matrix
    formation_years: list[int]
    returns_by_year: dict     # holding year -> daily bond returns (days x bonds)
    rf_by_year: dict          # holding year -> daily risk-free returns


# ------------------------------------------------------ one graph configuration

def rank_bonds(graph_config: GraphConfig, data: SearchData) -> tuple[dict, float]:
    """
    For every centrality and formation year: all bonds, most central first.

    Returns
    -------
    rankings : dict
        (centrality name, parameter) -> {formation year: bond indices, most central first}
    isolated_fraction : float
        Average share of bonds with no edges, a warning sign that the ranking
        is mostly random tie-breaking.
    """
    rankings = {}
    isolated = []
    for year in data.formation_years:
        C = data.correlations[(graph_config.weighting, graph_config.shrink, year)]
        A = adjacency_matrix(C, graph_config.threshold, graph_config.graph_type)
        graph = Graph(A, graph_config.graph_type.weighted)
        isolated.append(graph.isolated_fraction)

        rng = np.random.default_rng([graph_config.seed(), year])
        for centrality, parameter in config.CENTRALITIES:
            scores = centrality(graph, parameter)
            if scores is None:          # measure not defined for this kind of graph
                continue
            key = (centrality.__name__, parameter)
            rankings.setdefault(key, {})[year] = rank_by_centrality(scores, rng)
    return rankings, float(np.mean(isolated))


def evaluate_graph_config(graph_config: GraphConfig, data: SearchData) -> list[dict]:
    """One results row per (centrality, selection, portfolio size) on this graph configuration."""
    rankings, isolated_fraction = rank_bonds(graph_config, data)
    settings = {
        "weighting": graph_config.weighting,
        "shrink": graph_config.shrink,
        "sign": graph_config.graph_type.sign,
        "loops": graph_config.graph_type.loops,
        "weighted": graph_config.graph_type.weighted,
        "threshold": graph_config.threshold,
        "isolated_frac": isolated_fraction,
    }
    rows = []
    for (centrality, parameter), ranking_by_year in rankings.items():
        for selection in config.SELECTIONS:
            for m in config.PORTFOLIO_SIZES:
                # chosen with the data of formation year y, held during year y + 1
                holdings = {year + 1: select_bonds(ranking, selection, m)
                            for year, ranking in ranking_by_year.items()}
                metrics = evaluate_portfolio(data.returns_by_year, data.rf_by_year, holdings)
                rows.append({**settings, "centrality": centrality, "alpha": parameter,
                             "selection": selection, "m": m, **metrics})
    return rows


# ----------------------------------------------------------------- the search

def _correlation_task(key, returns):
    weighting, shrink, year = key
    return key, correlation_for_year(returns, year, weighting, shrink)


def compute_correlations(returns, graph_configs, formation_years, processes=None) -> dict:
    """Every yearly correlation matrix the graph configurations need, computed in parallel."""
    keys = sorted({(g.weighting, g.shrink) for g in graph_configs})
    tasks = [(weighting, shrink, year) for weighting, shrink in keys for year in formation_years]
    return dict(parallel_map(_correlation_task, tasks, returns, processes))


def run_grid_search(
    returns: pd.DataFrame,
    rf_returns: pd.Series,
    formation_years: list[int],
    graph_configs: list[GraphConfig] | None = None,
    processes: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate every strategy, walking forward from each formation year to the next.

    Parameters
    ----------
    returns, rf_returns
        Daily bond and risk-free returns covering every formation year and
        the year after each.
    formation_years
        Years whose data select the portfolio held in the following year.
    graph_configs
        Graph configurations to try (default: all of them).
    processes
        Number of CPU cores to use (default: all).

    Returns
    -------
    One row per strategy, best first according to config.RANK_BY.
    """
    graph_configs = graph_configs if graph_configs is not None else all_graph_configs()
    holding_years = [y + 1 for y in formation_years]

    if verbose:
        print("Computing yearly correlation matrices...", flush=True)
    data = SearchData(
        correlations=compute_correlations(returns, graph_configs, formation_years, processes),
        formation_years=formation_years,
        returns_by_year=split_by_year(returns, holding_years),
        rf_by_year=split_by_year(rf_returns, holding_years),
    )

    if verbose:
        print(f"Evaluating {len(graph_configs)} graph configurations...", flush=True)
    rows = []
    for done, result in enumerate(parallel_map(evaluate_graph_config, graph_configs, data, processes), 1):
        rows.extend(result)
        if verbose and (done % 25 == 0 or done == len(graph_configs)):
            print(f"  {done}/{len(graph_configs)} done ({len(rows)} strategies)", flush=True)

    rank_column = metric_name(config.RANK_BY)
    best_first_ascending = rank_column in LOWER_IS_BETTER
    return pd.DataFrame(rows).sort_values(rank_column, ascending=best_first_ascending, ignore_index=True)
