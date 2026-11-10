"""
Recheck the best training strategies with random tie-breaking
(see thesis/seed_check.py). Run main.py first.

    .venv/bin/python seed_check.py
"""
import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import pandas as pd  # noqa: E402

from thesis import config  # noqa: E402
from thesis.data import daily_returns, load_dataset  # noqa: E402
from thesis.grid_search import prepare_search_data  # noqa: E402
from thesis.metrics import metric_name  # noqa: E402
from thesis.seed_check import seed_spread  # noqa: E402


def main():
    dataset = load_dataset()
    returns = daily_returns(dataset.train)
    rf_returns = daily_returns(dataset.risk_free).loc[returns.index]
    formation_years = sorted(set(returns.index.year))[:-1]
    data = prepare_search_data(returns, rf_returns, formation_years, config.PROCESSES)

    metric = metric_name(config.RANK_BY)
    results = pd.read_parquet(config.TRAIN_RESULTS_PATH)
    top = results[results.matrix.notna()].nlargest(config.SEED_CHECK_TOP, metric)
    print(f"Rechecking the top {len(top)} strategies with {config.SEED_CHECK_SEEDS} other seeds...", flush=True)
    rechecked = seed_spread(top, data, config.SEED_CHECK_SEEDS, config.PROCESSES)
    rechecked.to_parquet(config.SEED_CHECK_RESULTS_PATH, index=False)

    spread = rechecked.groupby("strategy")[metric].agg(["mean", "std", "min", "max"])
    spread["share > 0"] = rechecked.groupby("strategy")[metric].apply(lambda s: (s > 0).mean())
    summary = top[["matrix", "centrality", "alpha", "sign", "loops", "weighted", "threshold",
                   "selection", "m", "bonds_held", "isolated_frac", metric]].join(spread)

    pd.set_option("display.width", 250)
    print(f"\n{metric}: tied places shared (as in the grid), and with {config.SEED_CHECK_SEEDS} random tie-breaks")
    print(summary.head(25).round(3).to_string())
    print(f"\nTop {len(top)}: {metric} with tied places shared, median {top[metric].median():.3f}; "
          f"mean over random tie-breaks, median {spread['mean'].median():.3f}")
    print(f"Strategies whose mean over random tie-breaks is > 0: {(spread['mean'] > 0).sum()} of {len(top)}")
    print(f"Strategies with identical results under every seed (no ties among the chosen bonds): "
          f"{(spread['std'] < 1e-12).sum()} of {len(top)}")


if __name__ == "__main__":
    main()
