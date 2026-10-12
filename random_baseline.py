"""
Draw the random-portfolio baseline on the training years and save it.

    .venv/bin/python random_baseline.py
"""
import pandas as pd

from thesis import config
from thesis.data import daily_returns, load_dataset
from thesis.random_baseline import run_random_baseline


def main():
    dataset = load_dataset()
    returns = daily_returns(dataset.train)
    rf_returns = daily_returns(dataset.risk_free).loc[returns.index]
    formation_years = sorted(set(returns.index.year))[:-1]

    results = run_random_baseline(returns, rf_returns, formation_years)
    results.to_parquet(config.RANDOM_RESULTS_PATH, index=False)

    summary = results.groupby("m").SR.describe(percentiles=[0.5, 0.9, 0.99])
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    print(f"\nWritten to {config.RANDOM_RESULTS_PATH}")


if __name__ == "__main__":
    main()
