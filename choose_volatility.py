"""
Which volatility version should the Sharpe ratio use? (training years only)

Daily bond-index returns are autocorrelated, so the volatility versions
(daily, Lo, monthly blocks, yearly blocks) give different Sharpe
ratios. They are judged against metrics that use no volatility at all:

1. Agreement with the Calmar ratio ER / MaxDD: a good Sharpe ratio should
   rank strategies much like return per unit of drawdown does.
2. The top 100 strategies by each Sharpe version: the version whose top 100
   has higher ER and lower MaxDD picks better strategies.

Reads results_train.parquet (run main.py first); the test set is not used.
"""
import pandas as pd

from thesis import config

SHARPE_VERSIONS = {"daily": "SR", "Lo": "SR_Lo", "monthly": "SR_M", "yearly": "SR_Y"}
TOP = 100


def calmar_agreement(results: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each Sharpe version with ER / MaxDD, overall and per selection."""
    calmar = results["ER"] / results["MaxDD"]
    groups = {"all": results.index, **results.groupby("selection").groups}
    return pd.DataFrame({
        group: {version: results.loc[rows, column].corr(calmar.loc[rows], method="spearman")
                for version, column in SHARPE_VERSIONS.items()}
        for group, rows in groups.items()
    })


def top_strategies(results: pd.DataFrame) -> pd.DataFrame:
    """ER and MaxDD of the TOP strategies chosen by each Sharpe version."""
    rows = {}
    for version, column in SHARPE_VERSIONS.items():
        top = results.nlargest(TOP, column)
        rows[version] = {
            "mean ER": top["ER"].mean(),
            "median ER": top["ER"].median(),
            "mean MaxDD": top["MaxDD"].mean(),
            "median MaxDD": top["MaxDD"].median(),
            "share peripheral": (top["selection"] == "peripheral").mean(),
        }
    return pd.DataFrame(rows).T


def top_overlap(results: pd.DataFrame) -> pd.DataFrame:
    """How many of the TOP strategies two Sharpe versions have in common."""
    tops = {version: set(results.nlargest(TOP, column).index) for version, column in SHARPE_VERSIONS.items()}
    return pd.DataFrame({a: {b: len(tops[a] & tops[b]) for b in tops} for a in tops})


def main():
    results = pd.read_parquet(config.TRAIN_RESULTS_PATH)
    pd.set_option("display.width", 200)
    print("1. Spearman correlation of each Sharpe version with ER / MaxDD\n")
    print(calmar_agreement(results).round(3), "\n")
    print(f"2. The top {TOP} strategies by each Sharpe version\n")
    print(top_strategies(results).round(4), "\n")
    print(f"   Strategies two versions' top {TOP} have in common\n")
    print(top_overlap(results), "\n")


if __name__ == "__main__":
    main()
