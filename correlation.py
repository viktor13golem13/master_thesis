import pandas as pd
import numpy as np


def compute_rolling_correlations(
    df: pd.DataFrame, window: int = 125, chunk_size: int = 50_000
) -> pd.DataFrame:
    """
    For each year and each pair of non-Date columns, compute a rolling Pearson
    correlation over a given window. The window resets at the start of each year.

    Pairs are processed in chunks to keep peak memory usage bounded regardless
    of the number of columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'Date' column and numeric value columns.
    window : int
        Rolling window size in trading days (default: 125).
    chunk_size : int
        Number of column pairs to process at a time (default: 50_000).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Date' as the first column, followed by one column per
        pair of input columns (named 'col_a__col_b'), containing the rolling
        Pearson correlation. Only rows where a full window is available within
        each year are included.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    value_cols = [c for c in df.columns if c != "Date"]
    p = len(value_cols)

    i_idx, j_idx = np.triu_indices(p, k=1)
    n_pairs = len(i_idx)
    pair_names = [f"{value_cols[i]}__{value_cols[j]}" for i, j in zip(i_idx, j_idx)]

    yearly_results = []

    for year, group in df.groupby(df["Date"].dt.year):
        group = group.reset_index(drop=True)
        X = group[value_cols].to_numpy(dtype=float)
        n = len(X)
        m = n - window + 1

        psum = np.zeros((n + 1, p))
        psum[1:] = np.cumsum(X, axis=0)
        psum2 = np.zeros((n + 1, p))
        psum2[1:] = np.cumsum(X ** 2, axis=0)

        rsum = psum[window:] - psum[:m]
        rsum2 = psum2[window:] - psum2[:m]

        corr = np.empty((m, n_pairs), dtype=float)
        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            ci, cj = i_idx[start:end], j_idx[start:end]

            XiXj = X[:, ci] * X[:, cj]
            psum_ij = np.zeros((n + 1, end - start))
            psum_ij[1:] = np.cumsum(XiXj, axis=0)
            rsum_ij = psum_ij[window:] - psum_ij[:m]

            num = window * rsum_ij - rsum[:, ci] * rsum[:, cj]
            denom = np.sqrt(
                (window * rsum2[:, ci] - rsum[:, ci] ** 2) *
                (window * rsum2[:, cj] - rsum[:, cj] ** 2)
            )
            corr[:, start:end] = num / denom

        dates = group["Date"].iloc[window - 1:].reset_index(drop=True)
        result = pd.DataFrame(corr, columns=pair_names)
        result.insert(0, "Date", dates)
        yearly_results.append(result)

    return pd.concat(yearly_results, ignore_index=True)


def compute_yearly_averages(
    df: pd.DataFrame,
    weighted: bool = False,
    c2: float = 125,
) -> pd.DataFrame:
    """
    Calculate the average of each non-Date column for each year.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'Date' column and numeric value columns.
    weighted : bool
        If True, each value is weighted by exp((365 - day - c2) / c2),
        normalised to sum to 1 within each year.
    c2 : float
        Decay parameter (default 125).

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_years, 1 + n_value_cols) with a 'Date' column
        followed by the (weighted) mean of each value column for that year.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    value_cols = [c for c in df.columns if c != "Date"]

    if not weighted:
        result = df.groupby(df["Date"].dt.year)[value_cols].mean().reset_index()
        result = result.rename(columns={"Date": "Date"})
        return result

    rows = []
    for year, group in df.groupby(df["Date"].dt.year):
        day = group["Date"].dt.day_of_year.to_numpy(dtype=float)
        raw_weights = np.exp((365 - day - c2) / c2)
        weights = raw_weights / raw_weights.sum()
        values = group[value_cols].to_numpy(dtype=float)
        avg = (weights[:, None] * values).sum(axis=0)
        rows.append([year] + avg.tolist())

    return pd.DataFrame(rows, columns=["Date"] + value_cols)


def apply_threshold(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Set correlation values below threshold to 0, keep values >= threshold unchanged."""
    result = df.copy()
    value_cols = [c for c in result.columns if c != "Date"]
    result[value_cols] = result[value_cols].where(result[value_cols] >= threshold, 0)
    return result


def correlations_to_3d(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Reshape a pairwise correlation DataFrame into a 3D array of symmetric matrices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'Date' column followed by pair columns named 'col_a__col_b'.

    Returns
    -------
    dates : np.ndarray of shape (n_dates,)
    col_names : list of str
        Original column names. Entry [t, i, j] in matrices is the correlation
        between col_names[i] and col_names[j] on date t.
    matrices : np.ndarray of shape (n_dates, n_cols, n_cols)
        Symmetric correlation matrices with 0 on the diagonal.
    """
    pair_cols = [c for c in df.columns if c != "Date"]
    dates = df["Date"].values

    seen, col_names = set(), []
    for pair in pair_cols:
        a, b = pair.split("__", 1)
        for name in (a, b):
            if name not in seen:
                seen.add(name)
                col_names.append(name)
    p = len(col_names)
    col_index = {name: idx for idx, name in enumerate(col_names)}

    i_idx = np.array([col_index[pair.split("__", 1)[0]] for pair in pair_cols])
    j_idx = np.array([col_index[pair.split("__", 1)[1]] for pair in pair_cols])

    values = df[pair_cols].to_numpy(dtype=float)

    matrices = np.zeros((len(dates), p, p), dtype=float)
    matrices[:, i_idx, j_idx] = values
    matrices[:, j_idx, i_idx] = values

    return dates, col_names, matrices
