import os

# One BLAS thread per process: parallelism comes from the worker pool.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import time  # noqa: E402

import pandas as pd  # noqa: E402

from clean import clean_dataset  # noqa: E402
from correlation import daily_returns  # noqa: E402
from evaluation import equal_weight_all  # noqa: E402
from search import run_grid_search  # noqa: E402

RESULTS_PATH = "results_train.parquet"

if __name__ == "__main__":
    train, test, rf, report = clean_dataset()
    returns = daily_returns(train)
    rf_returns = daily_returns(rf).loc[returns.index]

    train_years = sorted(set(returns.index.year))
    formation_years = train_years[:-1]          # the last train year is only held, never formed

    start = time.time()
    results = run_grid_search(returns, rf_returns, formation_years)
    results.to_parquet(RESULTS_PATH, index=False)
    print(f"\n{len(results)} strategies in {time.time() - start:.0f}s, written to {RESULTS_PATH}")

    holding_years = [y + 1 for y in formation_years]
    by_year = {y: returns.loc[returns.index.year == y].to_numpy() for y in holding_years}
    rf_by_year = {y: rf_returns.loc[rf_returns.index.year == y].to_numpy() for y in holding_years}
    benchmark = equal_weight_all(by_year, rf_by_year, holding_years)

    pd.set_option("display.width", 250)
    shown = ["weighting", "shrink", "sign", "loops", "weighted", "threshold", "centrality", "alpha",
             "selection", "m", "ER", "SD", "SR", "MaxDD", "Sortino"]
    print(f"\nTop 15 strategies (holding years {holding_years[0]}-{holding_years[-1]}):")
    print(results[shown].head(15).round(4).to_string(index=False))
    print("\nEqual weight, all series:", {k: round(v, 4) for k, v in benchmark.items()})
