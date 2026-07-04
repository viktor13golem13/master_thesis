import itertools
import numpy as np
import pandas as pd

from centrality import compute_centralities
from cache import load_or_build_cache, cache_to_numpy
from evaluation import avg_sharpe_from_rankings, geometric_sharpe_from_rankings


def compute_walk_forward_rankings(
    all_years: list,
    np_cache: list,
    col_names: list[str],
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    weighted: bool,
    relu_threshold: float,
    centrality_fn,
) -> dict:
    """
    Compute centrality rankings for each eval year using only prior-year data.

    Parameters
    ----------
    all_years : list
        All years in the training set in sorted order.
    np_cache : list
        Output of cache_to_numpy — one entry per eval year.
    col_names : list of str
    i_idx, j_idx : np.ndarray
        Pair index arrays for reconstructing 3D matrices from flat pair values.
    weighted : bool
        Whether to use time-weighted yearly averages.
    relu_threshold : float
        Correlation values below this are set to 0.
    centrality_fn : callable
        A centrality function from centrality.py.

    Returns
    -------
    dict mapping eval_year to (last_ranking, last_matrix, col_names).
    """
    p = len(col_names)
    result = {}
    for i, eval_year in enumerate(all_years[1:], start=1):
        years, values = np_cache[i - 1][weighted]
        pos_values = np.where(values >= relu_threshold, values, 0.0)
        n = len(years)
        matrices = np.zeros((n, p, p), dtype=float)
        matrices[:, i_idx, j_idx] = pos_values
        matrices[:, j_idx, i_idx] = pos_values

        rankings = compute_centralities(centrality_fn, years, col_names, matrices)
        last_ranking = list(rankings.values())[-1]
        last_matrix = matrices[-1]
        result[eval_year] = (last_ranking, last_matrix, col_names)
    return result


def hyperparameter_search(
    df: pd.DataFrame,
    param_grid: dict,
    cache_path: str = "cached_averages.pkl",
    eval_fn=geometric_sharpe_from_rankings,
    verbose: bool = True,
) -> tuple[dict, list]:
    """
    Grid search over hyperparameters using walk-forward Sharpe ratio.

    Rankings are computed once per unique (weighted, threshold, centrality_fn)
    triple, then reused across all top_n / use_top combinations — reducing the
    number of centrality computations from 96 to 16.

    Parameters
    ----------
    eval_fn : callable
        Function used to score each combination. Defaults to
        geometric_sharpe_from_rankings (compounded returns). Pass
        avg_sharpe_from_rankings for the arithmetic version.

    Parameters
    ----------
    df : pd.DataFrame
        Full training dataset with 'Date' column.
    param_grid : dict
        Keys: weighted, relu_threshold, centrality_fn, top_n, use_top.
        Values: lists of values to try.
    cache_path : str
        Path to the pre-computed correlation averages pickle file.
    verbose : bool

    Returns
    -------
    best_params : dict
    results : list of (params, avg_sharpe) sorted descending by avg_sharpe.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    all_years = sorted(df["Date"].dt.year.unique())

    cached_averages = load_or_build_cache(df, cache_path, verbose)

    if verbose:
        print("Converting cache to numpy arrays...", flush=True)
    np_cache, col_names, i_idx, j_idx = cache_to_numpy(cached_averages)
    del cached_averages
    if verbose:
        print("  Done.", flush=True)

    # Phase 1: compute rankings for each unique (weighted, threshold, centrality) triple
    unique_triples = list(itertools.product(
        param_grid["weighted"],
        param_grid["relu_threshold"],
        param_grid["centrality_fn"],
    ))
    if verbose:
        print(f"\nComputing rankings for {len(unique_triples)} unique triples...")

    rankings_cache = {}
    for t_idx, (weighted, relu_threshold, centrality_fn) in enumerate(unique_triples, 1):
        key = (weighted, relu_threshold, centrality_fn.__name__)
        rankings_cache[key] = compute_walk_forward_rankings(
            all_years, np_cache, col_names, i_idx, j_idx,
            weighted, relu_threshold, centrality_fn,
        )
        if verbose:
            print(
                f"  [{t_idx}/{len(unique_triples)}] weighted={weighted}, "
                f"threshold={relu_threshold}, centrality={centrality_fn.__name__}",
                flush=True,
            )

    # Phase 2: evaluate all combinations — only top_n and use_top vary here
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    if verbose:
        print(f"\nSearching {len(combinations)} hyperparameter combinations...\n")

    results = []
    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        key = (params["weighted"], params["relu_threshold"], params["centrality_fn"].__name__)
        score = eval_fn(df, rankings_cache[key], params["top_n"], params["use_top"])
        results.append((params, score))
        if verbose:
            print(
                f"[{idx}/{len(combinations)}] weighted={params['weighted']}, "
                f"threshold={params['relu_threshold']}, "
                f"centrality={params['centrality_fn'].__name__}, "
                f"top_n={params['top_n']}, "
                f"use_top={params['use_top']}  →  sharpe={score:.4f}",
                flush=True,
            )

    results.sort(key=lambda x: x[1], reverse=True)
    best_params, best_score = results[0]

    if verbose:
        print(f"\n=== BEST PARAMS (Sharpe={best_score:.4f}) ===")
        for k, v in best_params.items():
            print(f"  {k}: {v.__name__ if callable(v) else v}")

    return best_params, results
