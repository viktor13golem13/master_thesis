"""
Is the best strategy just the luckiest of many? (training years only)

Deflated Sharpe Ratio, Probability of Backtest Overfitting and a comparison
with random portfolios; see thesis.overfitting. The trials are the distinct
peripheral graph strategies: peripheral is the paper's hypothesis, fixed
before seeing the bond data, and central strategies are its control group.
All graph strategies are shown as a sensitivity check.

    .venv/bin/python overfitting.py      (after main.py and random_baseline.py)
"""
import numpy as np
import pandas as pd

from thesis import config
from thesis.backtest import split_by_year
from thesis.data import daily_returns, load_dataset
from thesis.overfitting import backtest_overfitting, deflated_sharpe_ratio, expected_best_of

CSCV_BLOCKS = 10
SMALLER_N = (1_000, 10_000)          # the distinct strategies are correlated, so fewer are really independent
RANDOM_N = (100, 1_000)


def holding_risk_free() -> np.ndarray:
    """Daily risk-free returns of the holding years, the rows of the strategy returns."""
    dataset = load_dataset()
    returns = daily_returns(dataset.train)
    rf_returns = daily_returns(dataset.risk_free).loc[returns.index]
    holding_years = sorted(set(returns.index.year))[1:]
    return np.concatenate(list(split_by_year(rf_returns, holding_years).values()))


def distinct(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct portfolio."""
    return rows.drop_duplicates("series")


def main():
    results = pd.read_parquet(config.TRAIN_RESULTS_PATH)
    strategy_returns = np.load(config.STRATEGY_RETURNS_PATH)
    risk_free = holding_risk_free()
    random = pd.read_parquet(config.RANDOM_RESULTS_PATH)

    graph = results[results.matrix.notna()]
    trial_sets = {"peripheral": distinct(graph[graph.selection == "peripheral"]), "all graph": distinct(graph)}
    pd.set_option("display.width", 200)

    print("1. Deflated Sharpe Ratio\n")
    rows = []
    for name, trials in trial_sets.items():
        best = trials.loc[trials.SR_Lo.idxmax()]
        excess = strategy_returns[:, best.series].astype(float) - risk_free
        for n in (len(trials), *SMALLER_N):
            rows.append({"trials": name, **deflated_sharpe_ratio(excess, best.SR_Lo, trials.SR_Lo.to_numpy(), n)})
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

    print(f"2. Probability of Backtest Overfitting ({CSCV_BLOCKS} blocks)\n")
    rows = []
    for name, trials in trial_sets.items():
        pbo = backtest_overfitting(strategy_returns[:, trials.series.to_numpy()], risk_free, CSCV_BLOCKS)
        rows.append({"trials": name, "strategies": len(trials), "PBO": pbo["PBO"],
                     "median in-sample SR_Lo of best": np.median(pbo["in_sample"]),
                     "median out-of-sample SR_Lo of it": np.median(pbo["out_of_sample"]),
                     "median out-of-sample rank": np.median(pbo["omega"]),
                     "share out-of-sample < 0": np.mean(pbo["out_of_sample"] < 0)})
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

    print("3. Best peripheral graph strategy against the expected best of n random portfolios (SR_Lo)\n")
    peripheral = trial_sets["peripheral"]
    rows = []
    for m, trials in peripheral.groupby("m"):
        random_sr = random.loc[random.m == m, "SR_Lo"].to_numpy()
        row = {"m": m, "best graph": trials.SR_Lo.max(), "distinct graph strategies": len(trials)}
        for n in (*RANDOM_N, len(trials)):
            row[f"random, n={n}" if n in RANDOM_N else "random, n=distinct"] = expected_best_of(random_sr, n)
        row["random, max of all"] = random_sr.max()
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
