"""
Run the grid search on the training years and print the best strategies.

    .venv/bin/python main.py
"""
import os

# Use one core per worker process; the parallelism comes from running many
# workers at once. Must be set before numpy is imported.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thesis import config  # noqa: E402
from thesis.backtest import equal_weight_benchmark, split_by_year  # noqa: E402
from thesis.data import daily_returns, load_dataset  # noqa: E402
from thesis.grid_search import run_grid_search  # noqa: E402
from thesis.metrics import all_metrics  # noqa: E402

SHOWN_COLUMNS = ["matrix", "weighting", "shrink", "sign", "loops", "weighted", "threshold",
                 "centrality", "alpha", "selection", "m", "ER", "SD_Lo", "SR_Lo", "SR", "MaxDD"]


def main():
    # 1. Data
    dataset = load_dataset()
    dataset.report.to_csv(config.CLEANING_REPORT_PATH, index=False)
    returns = daily_returns(dataset.train)
    rf_returns = daily_returns(dataset.risk_free).loc[returns.index]

    # Every train year except the last forms a portfolio held in the next year.
    train_years = sorted(set(returns.index.year))
    formation_years = train_years[:-1]
    holding_years = train_years[1:]

    # 2-6. Grid search
    start = time.time()
    results, strategy_returns = run_grid_search(returns, rf_returns, formation_years, processes=config.PROCESSES)
    results.to_parquet(config.TRAIN_RESULTS_PATH, index=False)
    np.save(config.STRATEGY_RETURNS_PATH, strategy_returns)
    print(f"\n{len(results)} strategies ({strategy_returns.shape[1]} distinct portfolios) in "
          f"{time.time() - start:.0f}s, written to {config.TRAIN_RESULTS_PATH} and {config.STRATEGY_RETURNS_PATH}")

    # Benchmarks: every bond with equal weight, and the Vanguard bond ETF
    rf_by_year = split_by_year(rf_returns, holding_years)
    equal_weight = equal_weight_benchmark(split_by_year(returns, holding_years), rf_by_year, holding_years)
    vanguard_returns = daily_returns(dataset.benchmark).loc[returns.index]
    vanguard = all_metrics(
        np.concatenate(list(split_by_year(vanguard_returns, holding_years).values())),
        np.concatenate(list(rf_by_year.values())),
    )

    pd.set_option("display.width", 250)
    print(f"\nTop 15 strategies (holding years {holding_years[0]}-{holding_years[-1]}):")
    print(results[SHOWN_COLUMNS].head(15).round(4).to_string(index=False))
    print("\nEqual weight, all bonds:", {k: round(v, 4) for k, v in equal_weight.items()})
    print("Vanguard Total Bond ETF:", {k: round(v, 4) for k, v in vanguard.items()})


if __name__ == "__main__":
    main()
