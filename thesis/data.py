"""
Steps 0 and 1 — Build, load and clean the bond index data.

Step 0 rebuilds train.csv and test.csv from the raw Datastream downloads:
945 FTSE and iBoxx bond indices with daily price levels (train: 2017-2023,
test: 2024-2025).

Step 1 removes series that are unusable for the analysis. Each rule lists
the series it removes; together they define the investable universe, which
is the same for train and test.
"""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from thesis import config


@dataclass
class Dataset:
    train: pd.DataFrame        # price levels of investable series, indexed by date
    test: pd.DataFrame
    risk_free: pd.Series       # 3-month T-bill price levels, train and test
    benchmark: pd.Series       # Vanguard Total Bond Market ETF prices on the same days
    report: pd.DataFrame       # every removed series and the rule that removed it


def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Daily simple returns (p_t / p_{t-1} - 1); the first day is dropped."""
    return prices.pct_change(fill_method=None).iloc[1:]


# ------------------------------------------- step 0: building train and test

def read_export(path: str) -> pd.DataFrame:
    """One raw download: dates in the first column (headed 'Name' or 'Date'), one column per series."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.index.name = "Date"
    return prices


def join_raw_files() -> pd.DataFrame:
    """
    All raw downloads side by side, on the union of their dates.

    Datastream writes a column headed '#ERROR' for every series the licence
    could not download ('$$ER: E100, ACCESS DENIED'); those are dropped, as
    is 'Name.1', a second copy of the dates in the later download.
    """
    exports = pd.concat([read_export(path) for path in config.DATASTREAM_EXPORTS], axis=1)
    exports = exports.loc[:, ~exports.columns.str.startswith("#ERROR")]
    later = read_export(config.LATER_EXPORT).drop(columns="Name.1")
    return pd.concat([exports, later], axis=1).sort_index()


def series_without_full_history(prices: pd.DataFrame) -> list[str]:
    """Series with a missing value anywhere between the first train day and the last test day."""
    period = prices.loc[f"{config.TRAIN_YEARS[0]}-01-01":f"{config.TEST_YEARS[1]}-12-31"]
    return list(period.columns[period.isna().any()])


def build_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rebuild train and test price levels from the raw downloads.

    Joins every download, keeps only series with data on every day of
    2017-2025 (and not listed in config.REMOVED_BY_HAND), then splits by year.
    """
    prices = join_raw_files()
    removed = set(series_without_full_history(prices)) | set(config.REMOVED_BY_HAND)
    prices = prices[[c for c in prices.columns if c not in removed]]
    train = prices.loc[f"{config.TRAIN_YEARS[0]}-01-01":f"{config.TRAIN_YEARS[1]}-12-31"]
    test = prices.loc[f"{config.TEST_YEARS[0]}-01-01":f"{config.TEST_YEARS[1]}-12-31"]
    return train, test


# ------------------------------------------- step 1: cleaning rules

def duplicate_series(prices: pd.DataFrame) -> list[str]:
    """Exact copies of an earlier column (left over from merging the raw files)."""
    return list(prices.columns[prices.T.duplicated(keep="first")])


def cash_series(prices: pd.DataFrame) -> list[str]:
    """Money-market indices (T-bills, eurodeposits, monetary rates): not bonds."""
    return [c for c in prices.columns if re.search(config.CASH_PATTERN, c)]


def nonpositive_series(prices: pd.DataFrame) -> list[str]:
    """Series with a price of zero or below (Russian indices were set to 0 in March 2022)."""
    return list(prices.columns[(prices <= 0).any()])


def glitch_series(prices: pd.DataFrame) -> list[str]:
    """
    Series with an impossible one-day move (more than GLITCH_THRESHOLD),
    outside the genuine market events listed in config.REAL_EVENT_WINDOWS.
    """
    returns = daily_returns(prices.where(prices > 0))
    during_real_event = np.zeros(len(returns), dtype=bool)
    for start, end in config.REAL_EVENT_WINDOWS:
        during_real_event |= (returns.index >= start) & (returns.index <= end)
    glitches = returns.abs().gt(config.GLITCH_THRESHOLD) & ~during_real_event[:, None]
    return list(prices.columns[glitches.any()])


def stale_series(prices: pd.DataFrame) -> list[str]:
    """Series whose price did not change on more than STALE_THRESHOLD of the days of some year."""
    returns = daily_returns(prices.where(prices > 0))
    zero_fraction = returns.eq(0).groupby(returns.index.year).mean()
    return list(prices.columns[zero_fraction.gt(config.STALE_THRESHOLD).any()])


CLEANING_RULES = {
    "duplicate": duplicate_series,
    "cash": cash_series,
    "nonpositive": nonpositive_series,
    "glitch": glitch_series,
    "stale": stale_series,
}


def removed_series(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Apply every cleaning rule. A series caught by several rules is reported
    under the first one in CLEANING_RULES.

    Returns a table with columns 'series' and 'reason'.
    """
    reasons = {}
    for reason, rule in CLEANING_RULES.items():
        for series in rule(prices):
            reasons.setdefault(series, reason)
    return pd.DataFrame(list(reasons.items()), columns=["series", "reason"])


# -------------------------------------------------- step 1: loading

def load_dataset(verbose: bool = True) -> Dataset:
    """
    Load train and test prices and remove the unusable series.

    The rules look at train and test together, so both share one fixed
    universe of series.
    """
    train = pd.read_csv(config.TRAIN_PATH, index_col="Date", parse_dates=True)
    test = pd.read_csv(config.TEST_PATH, index_col="Date", parse_dates=True)
    full = pd.concat([train, test])

    report = removed_series(full)
    keep = [c for c in full.columns if c not in set(report["series"])]

    if verbose:
        print(f"Series: {full.shape[1]} raw -> {len(keep)} kept")
        for reason, count in report["reason"].value_counts().items():
            print(f"  removed {count:4d}  {reason}")

    return Dataset(train[keep], test[keep], full[config.RISK_FREE_SERIES],
                   load_benchmark(full.index), report)


def load_benchmark(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Vanguard Total Bond Market ETF prices on the given days (the paper
    compares against the S&P 500 index; this is the bond-market analogue).
    The raw file repeats two days in May 2026; the repeats are dropped.
    """
    prices = pd.read_csv(config.BENCHMARK_PATH, index_col="Date", parse_dates=True).iloc[:, 0]
    prices = prices[~prices.index.duplicated()].sort_index()
    return prices.reindex(dates)
