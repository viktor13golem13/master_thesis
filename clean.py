import re
import numpy as np
import pandas as pd

RF_COL = "FTSE 3-Month Treasury Bill Index - Total Return"

CASH_PATTERN = r"(?i)treasury bill|eurodeposit|monetary rate"

# Genuine market events whose large daily moves must not be mistaken for data
# glitches: the September 2022 UK gilt / LDI crisis (Bank of England intervention).
REAL_EVENT_WINDOWS = [("2022-09-23", "2022-10-14")]


def load_raw(train_path: str = "dataset/train.csv", test_path: str = "dataset/test.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test price levels indexed by Date."""
    train = pd.read_csv(train_path, index_col="Date", parse_dates=True)
    test = pd.read_csv(test_path, index_col="Date", parse_dates=True)
    return train, test


def find_dropped_series(
    prices: pd.DataFrame,
    jump_threshold: float = 0.20,
    stale_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Decide which series to exclude from the investable universe.

    Rules are applied in order and each series is attributed to the first rule
    that catches it:

    1. duplicate   — exact copy of an earlier column (file-merge artefacts).
    2. cash        — Treasury bill, eurodeposit and monetary-rate indices (money
                     market, not bonds).
                     The risk-free column is extracted separately.
    3. nonpositive — any price <= 0 (e.g. Russian indices set to 0 from March 2022).
    4. glitch      — any |daily return| > jump_threshold outside REAL_EVENT_WINDOWS.
    5. stale       — more than stale_threshold of daily returns exactly zero in any
                     calendar year (dead or unrefreshed series).

    Parameters
    ----------
    prices : pd.DataFrame
        Price levels indexed by Date, one column per series, over the full period.
    jump_threshold : float
        Absolute simple daily return above which a move is treated as a glitch.
    stale_threshold : float
        Maximum fraction of zero daily returns allowed in any calendar year.

    Returns
    -------
    pd.DataFrame with columns ['series', 'reason'], one row per dropped series.
    """
    dropped = {}

    def mark(cols, reason):
        for c in cols:
            dropped.setdefault(c, reason)

    mark(prices.columns[prices.T.duplicated(keep="first")], "duplicate")
    mark([c for c in prices.columns if re.search(CASH_PATTERN, c)], "cash")
    mark(prices.columns[(prices <= 0).any()], "nonpositive")

    returns = prices.where(prices > 0).pct_change(fill_method=None).iloc[1:]
    in_event = np.zeros(len(returns), dtype=bool)
    for start, end in REAL_EVENT_WINDOWS:
        in_event |= (returns.index >= start) & (returns.index <= end)
    jumps = returns.abs().gt(jump_threshold) & ~in_event[:, None]
    mark(prices.columns[jumps.any()], "glitch")

    zero_frac = returns.eq(0).groupby(returns.index.year).mean()
    mark(prices.columns[zero_frac.gt(stale_threshold).any()], "stale")

    return pd.DataFrame(list(dropped.items()), columns=["series", "reason"])


def clean_dataset(
    train_path: str = "dataset/train.csv",
    test_path: str = "dataset/test.csv",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the raw data and return the cleaned investable universe.

    The drop decision is made once over train and test combined, so both sets
    share the same fixed universe.

    Returns
    -------
    train, test : pd.DataFrame
        Price levels of the retained series, indexed by Date.
    rf : pd.Series
        Risk-free (3-month T-bill) price levels over the full period.
    report : pd.DataFrame
        Dropped series and the reason for each.
    """
    train, test = load_raw(train_path, test_path)
    full = pd.concat([train, test])

    rf = full[RF_COL]
    report = find_dropped_series(full)
    keep = [c for c in full.columns if c not in set(report["series"])]

    if verbose:
        print(f"Series: {full.shape[1]} raw -> {len(keep)} kept")
        for reason, n in report["reason"].value_counts().items():
            print(f"  dropped {n:4d}  {reason}")

    return train[keep], test[keep], rf, report


if __name__ == "__main__":
    train, test, rf, report = clean_dataset()
    report.to_csv("dataset/cleaning_report.csv", index=False)
    print("Report written to dataset/cleaning_report.csv")
