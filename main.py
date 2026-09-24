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

import pandas as pd  # noqa: E402

from thesis import config  # noqa: E402
from thesis.backtest import equal_weight_benchmark, split_by_year  # noqa: E402
from thesis.data import daily_returns, load_dataset  # noqa: E402
from thesis.grid_search import run_grid_search  # noqa: E402

SHOWN_COLUMNS = ["weighting", "shrink", "sign", "loops", "weighted", "threshold",
                 "centrality", "alpha", "selection", "m", "ER", "SD", "SR", "MaxDD", "Sortino"]


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
    results = run_grid_search(returns, rf_returns, formation_years)
    results.to_parquet(config.TRAIN_RESULTS_PATH, index=False)
    print(f"\n{len(results)} strategies in {time.time() - start:.0f}s, "
          f"written to {config.TRAIN_RESULTS_PATH}")

    # Benchmark: every bond with equal weight
    benchmark = equal_weight_benchmark(
        split_by_year(returns, holding_years), split_by_year(rf_returns, holding_years), holding_years,
    )

    pd.set_option("display.width", 250)
    print(f"\nTop 15 strategies (holding years {holding_years[0]}-{holding_years[-1]}):")
    print(results[SHOWN_COLUMNS].head(15).round(4).to_string(index=False))
    print("\nEqual weight, all bonds:", {k: round(v, 4) for k, v in benchmark.items()})


if __name__ == "__main__":
    main()
