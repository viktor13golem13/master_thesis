"""
Run the MIQP baseline on the training years and save the results.

    .venv/bin/python miqp_baseline.py

Needs a Gurobi licence (free academic licence for students).
"""
import pandas as pd

from thesis import config
from thesis.data import daily_returns, load_dataset
from thesis.miqp import run_miqp_baseline


def main():
    dataset = load_dataset()
    returns = daily_returns(dataset.train)
    rf_returns = daily_returns(dataset.risk_free).loc[returns.index]
    formation_years = sorted(set(returns.index.year))[:-1]

    results = run_miqp_baseline(returns, rf_returns, formation_years)
    results.to_parquet(config.MIQP_RESULTS_PATH, index=False)

    pd.set_option("display.width", 250)
    print(results.round(4).to_string(index=False))
    print(f"\nWritten to {config.MIQP_RESULTS_PATH}")


if __name__ == "__main__":
    main()
