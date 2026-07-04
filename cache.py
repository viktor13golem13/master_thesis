import os
import pickle
import numpy as np
import pandas as pd

from correlation import compute_rolling_correlations, compute_yearly_averages


def build_cache(df: pd.DataFrame, all_years: list, verbose: bool = True) -> list:
    """
    Pre-compute rolling correlation averages for each prior-year window.

    For year t, computes averages over years 1..t-1 so that walk-forward
    evaluation has no data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Full training dataset with 'Date' column (already parsed to datetime).
    all_years : list
        Sorted list of all years in the dataset.
    verbose : bool

    Returns
    -------
    list of dicts, one per eval year, each mapping weighted (bool) to a
    DataFrame of yearly-averaged correlations for the prior-year window.
    """
    cached_averages = []
    for i in range(1, len(all_years)):
        prior_years = all_years[:i]
        subset = df[df["Date"].dt.year.isin(prior_years)]
        raw_corr = compute_rolling_correlations(subset)
        cached_averages.append({
            False: compute_yearly_averages(raw_corr, weighted=False),
            True:  compute_yearly_averages(raw_corr, weighted=True),
        })
        del raw_corr
        if verbose:
            print(f"  Done for prior years: {prior_years}", flush=True)
    return cached_averages


def load_or_build_cache(
    df: pd.DataFrame,
    cache_path: str = "cached_averages.pkl",
    verbose: bool = True,
) -> list:
    """
    Load pre-computed averages from disk, or build and save them if absent.

    Parameters
    ----------
    df : pd.DataFrame
        Full training dataset with 'Date' column.
    cache_path : str
        Path to the pickle file.
    verbose : bool

    Returns
    -------
    list — same format as build_cache output.
    """
    if os.path.exists(cache_path):
        if verbose:
            print(f"Loading pre-computed averages from {cache_path}...", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if verbose:
        print("Pre-computing rolling correlations for each prior-year window...")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    all_years = sorted(df["Date"].dt.year.unique())

    cached_averages = build_cache(df, all_years, verbose=verbose)

    with open(cache_path, "wb") as f:
        pickle.dump(cached_averages, f)
    if verbose:
        print(f"Saved to {cache_path}.", flush=True)

    return cached_averages


def cache_to_numpy(cached_averages: list) -> tuple:
    """
    Convert the DataFrame-based cache to numpy arrays.

    Called once at startup so that all per-combination work uses pure numpy
    instead of pandas, which is far slower on wide DataFrames.

    Returns
    -------
    np_cache : list of {weighted: (years, values)}
        years  : np.ndarray of shape (n_prior_years,)
        values : np.ndarray of shape (n_prior_years, n_pairs), NaN → 0
    col_names : list of str
    i_idx, j_idx : np.ndarray — pair index arrays for building 3D matrices
    """
    ref_df = cached_averages[0][False]
    pair_cols = [c for c in ref_df.columns if c != "Date"]

    seen, col_names = set(), []
    for pair in pair_cols:
        a, b = pair.split("__", 1)
        for name in (a, b):
            if name not in seen:
                seen.add(name)
                col_names.append(name)
    col_index = {name: idx for idx, name in enumerate(col_names)}
    i_idx = np.array([col_index[pair.split("__", 1)[0]] for pair in pair_cols])
    j_idx = np.array([col_index[pair.split("__", 1)[1]] for pair in pair_cols])

    np_cache = []
    for entry in cached_averages:
        np_entry = {}
        for weighted, df in entry.items():
            years = df["Date"].values
            values = df.iloc[:, 1:].to_numpy(dtype=float)
            np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(values, -1.0, 1.0, out=values)
            np_entry[weighted] = (years, values)
        np_cache.append(np_entry)

    return np_cache, col_names, i_idx, j_idx
