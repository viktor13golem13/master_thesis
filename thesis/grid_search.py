"""
Step 6 — Try every strategy (paper, Sec. 4).

A strategy is one choice of every setting:

    graph configuration   matrix (correlation or covariance), window weighting,
                          shrinkage, graph type, threshold θ
    centrality            measure and its parameter
    selection             central or peripheral bonds
    portfolio size        m

A graph configuration is the unit of work: for every formation year it
builds one graph, scores the bonds by each centrality, and then evaluates
every (centrality, selection, m) strategy on those scores. The possible
values of each setting are listed in thesis.config.

The simple rules of thesis.simple_rules (no graph) are evaluated the same
way, as a baseline; their rows have no graph settings.

Many strategies end up holding the same bonds (Katz and NBTW often agree,
for example). The daily returns of every distinct portfolio are kept once,
and each row's "series" column points to them; the overfitting tests
(thesis.overfitting) need them.

run_grid_search is the entry point; the functions below it follow the order
in which it uses them.
"""
import itertools
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from thesis import config, simple_rules
from thesis.backtest import select_weights, split_by_year, weighted_portfolio_returns
from thesis.correlation import correlation_for_year, covariance_from_correlation, volatilities_for_year
from thesis.graph import Graph, GraphType, adjacency_matrix, matched_threshold
from thesis.metrics import LOWER_IS_BETTER, all_metrics, metric_name
from thesis.parallel import parallel_map


@dataclass(frozen=True)
class GraphConfig:
    matrix: str
    weighting: str
    shrink: bool
    graph_type: GraphType
    threshold: float

    def seed(self) -> int:
        """Fixed random seed for breaking ties at random (only the seed check does that)."""
        return zlib.crc32(repr(self).encode())


def all_graph_configs() -> list[GraphConfig]:
    """Every combination of the graph settings in config."""
    return [
        GraphConfig(matrix, weighting, shrink, graph_type, threshold)
        for matrix, weighting, shrink, graph_type, threshold in itertools.product(
            config.GRAPH_MATRICES, config.WINDOW_WEIGHTINGS, config.SHRINKAGE_OPTIONS,
            config.GRAPH_TYPES, config.THRESHOLDS,
        )
    ]


@dataclass
class SearchData:
    """What evaluating a graph configuration needs, shared by all workers."""
    correlations: dict        # (weighting, shrink, year) -> correlation matrix
    volatilities: dict        # formation year -> daily standard deviation of each bond
    formation_years: list[int]
    returns_by_year: dict     # holding year -> daily bond returns (days x bonds)
    rf_by_year: dict          # holding year -> daily risk-free returns


# ----------------------------------------------------------------- the search

def run_grid_search(
    returns: pd.DataFrame,
    rf_returns: pd.Series,
    formation_years: list[int],
    graph_configs: list[GraphConfig] | None = None,
    processes: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
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
    results
        One row per strategy, best first according to config.RANK_BY.
    strategy_returns : np.ndarray of shape (days, distinct portfolios), float32
        Daily returns of every distinct portfolio; results["series"] is the column of each row.
    """
    graph_configs = graph_configs if graph_configs is not None else all_graph_configs()
    if verbose:
        print("Computing yearly correlation matrices...", flush=True)
    data = prepare_search_data(returns, rf_returns, formation_years, processes)

    if verbose:
        print(f"Evaluating {len(graph_configs)} graph configurations...", flush=True)
    rows = []
    series = SeriesStore()
    for done, result in enumerate(parallel_map(evaluate_graph_config, graph_configs, data, processes), 1):
        rows.extend(series.add_rows(result))
        if verbose and (done % 25 == 0 or done == len(graph_configs)):
            print(f"  {done}/{len(graph_configs)} done ({len(rows)} strategies)", flush=True)
    rows.extend(series.add_rows(evaluate_simple_rules(data)))

    rank_column = metric_name(config.RANK_BY)
    best_first_ascending = rank_column in LOWER_IS_BETTER
    results = pd.DataFrame(rows).sort_values(rank_column, ascending=best_first_ascending, ignore_index=True)
    return results, series.matrix()


class SeriesStore:
    """Keeps each distinct daily return series once and numbers them 0, 1, 2, ..."""

    def __init__(self):
        self.index = {}             # bytes of a float32 series -> its number

    def add_rows(self, rows: list[dict]) -> list[dict]:
        """Replace each row's "daily" returns by the number of that series."""
        for row in rows:
            key = row.pop("daily").astype(np.float32).tobytes()
            row["series"] = self.index.setdefault(key, len(self.index))
        return rows

    def matrix(self) -> np.ndarray:
        """All series as columns, in order of their number (dicts keep insertion order)."""
        return np.frombuffer(b"".join(self.index), dtype=np.float32).reshape(len(self.index), -1).T


# ------------------------------------------ step 1: correlation matrices

def prepare_search_data(returns, rf_returns, formation_years, processes=None) -> SearchData:
    """Correlations and volatilities of every formation year, and returns of every holding year."""
    holding_years = [y + 1 for y in formation_years]
    return SearchData(
        correlations=compute_correlations(returns, formation_years, processes),
        volatilities={year: volatilities_for_year(returns, year) for year in formation_years},
        formation_years=formation_years,
        returns_by_year=split_by_year(returns, holding_years),
        rf_by_year=split_by_year(rf_returns, holding_years),
    )


def compute_correlations(returns, formation_years, processes=None) -> dict:
    """Every yearly correlation matrix (each window weighting and shrinkage option), computed in parallel."""
    tasks = list(itertools.product(config.WINDOW_WEIGHTINGS, config.SHRINKAGE_OPTIONS, formation_years))
    return dict(parallel_map(_correlation_task, tasks, returns, processes))


def _correlation_task(key, returns):
    weighting, shrink, year = key
    return key, correlation_for_year(returns, year, weighting, shrink)


# ------------------------------------- step 2: one graph configuration

def evaluate_graph_config(graph_config: GraphConfig, data: SearchData) -> list[dict]:
    """One results row per (centrality, selection, portfolio size) on this graph configuration."""
    scores_per_centrality, isolated_fraction = centrality_scores(graph_config, data)
    rows = []
    for (centrality, parameter), scores_by_year in scores_per_centrality.items():
        settings = {
            "matrix": graph_config.matrix,
            "weighting": graph_config.weighting,
            "shrink": graph_config.shrink,
            "sign": graph_config.graph_type.sign,
            "loops": graph_config.graph_type.loops,
            "weighted": graph_config.graph_type.weighted,
            "threshold": graph_config.threshold,
            "isolated_frac": isolated_fraction,
            "centrality": centrality,
            "alpha": parameter,
        }
        rows.extend(strategy_rows(settings, scores_by_year, data))
    return rows


def centrality_scores(graph_config: GraphConfig, data: SearchData) -> tuple[dict, float]:
    """
    For every centrality and formation year: the score of every bond.

    Returns
    -------
    scores : dict
        (centrality name, parameter) -> {formation year: one score per bond}
    isolated_fraction : float
        Average share of bonds with no edges, which all tie with each other.
    """
    scores = {}
    isolated = []
    for year in data.formation_years:
        graph = build_graph(graph_config, data, year)
        isolated.append(graph.isolated_fraction)

        for centrality, parameter in config.CENTRALITIES:
            year_scores = centrality(graph, parameter)
            if year_scores is None:     # measure not defined for this kind of graph
                continue
            scores.setdefault((centrality.__name__, parameter), {})[year] = year_scores
    return scores, float(np.mean(isolated))


def build_graph(graph_config: GraphConfig, data: SearchData, year: int) -> Graph:
    """The graph of one formation year under this graph configuration."""
    C = data.correlations[(graph_config.weighting, graph_config.shrink, year)]
    if graph_config.matrix == "covariance":
        M = covariance_from_correlation(C, data.volatilities[year])
        A = adjacency_matrix(M, matched_threshold(M, C, graph_config.threshold, graph_config.graph_type),
                             graph_config.graph_type)
    else:
        A = adjacency_matrix(C, graph_config.threshold, graph_config.graph_type)
    return Graph(A, graph_config.graph_type.weighted)


def strategy_rows(settings: dict, scores_by_year: dict, data: SearchData) -> list[dict]:
    """
    One results row per (selection, portfolio size) for one score per bond
    and formation year. bonds_held is the average number of bonds with a
    weight, more than m when bonds tie for the last places. Each row also
    carries its daily returns under "daily", which run_grid_search stores
    separately.
    """
    rows = []
    for selection in config.SELECTIONS:
        for m in config.PORTFOLIO_SIZES:
            weights = {}
            for year, scores in scores_by_year.items():
                weights[year + 1] = select_weights(scores, selection, m)

            daily = weighted_portfolio_returns(data.returns_by_year, weights)
            rf = np.concatenate([data.rf_by_year[year] for year in sorted(weights)])
            bonds_held = np.mean([np.count_nonzero(w) for w in weights.values()])
            rows.append({**settings, "selection": selection, "m": m, "bonds_held": bonds_held,
                         **all_metrics(daily, rf), "daily": daily})
    return rows


# ------------------------------------- step 3: rules without a graph

def evaluate_simple_rules(data: SearchData) -> list[dict]:
    """Results rows for the rules in thesis.simple_rules, ranked like centralities."""
    versions = [(rule, weighting, shrink)
                for rule in simple_rules.CORRELATION_RULES
                for weighting, shrink in itertools.product(config.WINDOW_WEIGHTINGS, config.SHRINKAGE_OPTIONS)]
    versions.append((simple_rules.volatility, None, None))

    rows = []
    for rule, weighting, shrink in versions:
        scores_by_year = {}
        for year in data.formation_years:
            C = data.correlations[(weighting, shrink, year)] if weighting is not None else None
            scores_by_year[year] = rule(C, data.volatilities[year])
        settings = {"matrix": None, "weighting": weighting, "shrink": shrink, "sign": None, "loops": None,
                    "weighted": None, "threshold": None, "isolated_frac": None,
                    "centrality": rule.__name__, "alpha": None}
        rows.extend(strategy_rows(settings, scores_by_year, data))
    return rows
