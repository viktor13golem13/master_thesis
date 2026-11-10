"""
Step 10 — Is the best strategy just the luckiest of many? (training years only)

The grid tries over a million strategies, so the best one looks good partly
by chance. Three checks:

1. Deflated Sharpe Ratio (Bailey & López de Prado 2014). If no strategy
   had any skill, the best of N would still reach a Sharpe ratio of about
   expected_maximum_sharpe. The DSR is the probability that the best
   strategy's true Sharpe ratio is above that level.
2. Probability of Backtest Overfitting (Bailey, Borwein, López de Prado &
   Zhu 2017, "CSCV"). The days are cut into blocks; for every way to pick
   half of the blocks, the best strategy on that half is looked up on the
   other half. PBO is the share of splits where it ranks below the median.
3. Random portfolios. The best strategy against the expected best of n
   random portfolios of the same size.

All Sharpe ratios are Lo's (metrics.sharpe_ratio_lo), as in the grid. For
the DSR they are divided by √261, which turns them into per-day values that
already include the autocorrelation correction.
"""
import itertools

import numpy as np
from scipy.stats import kurtosis, norm, skew

from thesis.metrics import LO_LAGS, TRADING_DAYS_PER_YEAR

EULER_GAMMA = 0.5772156649015329


# ------------------------------------------------------ 1. Deflated Sharpe Ratio

def expected_maximum_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """
    Expected best Sharpe ratio of n_trials strategies whose true Sharpe ratio
    is 0 and whose estimates have variance sharpe_variance (their eq. 6):

        √V · ((1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e)))

    with γ the Euler-Mascheroni constant.
    """
    if n_trials < 2:
        return 0.0
    return float(np.sqrt(sharpe_variance) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                                             + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))))


def probabilistic_sharpe_ratio(sharpe: float, benchmark: float, n_obs: int,
                               skewness: float, kurt: float) -> float:
    """
    Probability that the true Sharpe ratio is above benchmark, given an
    estimate from n_obs observations with this skewness and (non-excess)
    kurtosis. Both Sharpe ratios are per observation, not annualised.
    """
    spread = np.sqrt(1 - skewness * sharpe + (kurt - 1) / 4 * sharpe ** 2)
    return float(norm.cdf((sharpe - benchmark) * np.sqrt(n_obs - 1) / spread))


def deflated_sharpe_ratio(best_excess_returns: np.ndarray, best_sharpe: float,
                          trial_sharpes: np.ndarray, n_trials: int) -> dict:
    """
    DSR of the best strategy among n_trials.

    Parameters
    ----------
    best_excess_returns
        Daily returns minus the risk-free rate of the best strategy.
    best_sharpe
        Its annual Sharpe ratio (SR_Lo).
    trial_sharpes
        Annual Sharpe ratios of the strategies tried; only their variance is used.
    n_trials
        Number of independent trials N.
    """
    per_day = np.sqrt(TRADING_DAYS_PER_YEAR)
    variance = np.var(trial_sharpes / per_day, ddof=1)
    threshold = expected_maximum_sharpe(n_trials, variance)
    dsr = probabilistic_sharpe_ratio(best_sharpe / per_day, threshold, len(best_excess_returns),
                                     skew(best_excess_returns), kurtosis(best_excess_returns, fisher=False))
    return {"N": n_trials, "best SR_Lo": best_sharpe,
            "expected best SR_Lo without skill": threshold * per_day, "DSR": dsr}


# ------------------------------------------ 2. Probability of Backtest Overfitting

def sharpe_lo_columns(returns: np.ndarray, risk_free: np.ndarray) -> np.ndarray:
    """
    metrics.sharpe_ratio_lo of every column of returns (days x strategies) at
    once; NaN where Lo's volatility is undefined.
    """
    x = returns - returns.mean(axis=0)
    q = TRADING_DAYS_PER_YEAR
    total = np.einsum("ij,ij->j", x, x)
    for lag in range(1, LO_LAGS + 1):
        total += 2 * (1 - lag / q) * np.einsum("ij,ij->j", x[lag:], x[:-lag])
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(total / (len(x) - 1) * q)
        excess = (returns.mean(axis=0) - risk_free.mean()) * q
        return np.where(sd > 0, excess / sd, np.nan)


def backtest_overfitting(returns: np.ndarray, risk_free: np.ndarray, n_blocks: int = 10) -> dict:
    """
    PBO by combinatorially symmetric cross-validation.

    The days are cut into n_blocks consecutive blocks (leftover days go to the
    last one). Each way to choose n_blocks / 2 of them is one split: the chosen
    blocks are in-sample, the rest out-of-sample. Each half of a split is the
    other half of the opposite split, so every half's Sharpe ratios are
    computed once. Lags of Lo's volatility that cross a gap between blocks are
    included; with few blocks that matters little.

    Returns
    -------
    PBO, the share of splits whose in-sample best ranks below the out-of-sample
    median, and per split the in-sample and out-of-sample Sharpe ratio of that
    best strategy and its out-of-sample rank ω in (0, 1).
    """
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    blocks = np.array_split(np.arange(len(returns)), n_blocks)
    halves = list(itertools.combinations(range(n_blocks), n_blocks // 2))

    sharpes = {}
    for half in halves:
        days = np.concatenate([blocks[b] for b in half])
        sharpes[half] = np.nan_to_num(sharpe_lo_columns(returns[days], risk_free[days]), nan=-np.inf)

    n = returns.shape[1]
    in_sample, out_of_sample, omega = [], [], []
    for half in halves:
        other = tuple(b for b in range(n_blocks) if b not in half)
        best = int(np.argmax(sharpes[half]))
        value = sharpes[other][best]
        in_sample.append(sharpes[half][best])
        out_of_sample.append(value)
        omega.append((np.sum(sharpes[other] < value) + 1) / (n + 1))     # rank 1..n, over n + 1
    omega = np.array(omega)
    return {"PBO": float(np.mean(omega <= 0.5)), "in_sample": np.array(in_sample),
            "out_of_sample": np.array(out_of_sample), "omega": omega}


# ------------------------------------------------------- 3. random portfolios

def expected_best_of(values: np.ndarray, n: int) -> float:
    """
    Expected maximum of n draws (with replacement) from values: the largest of
    K sorted values is the i-th with probability (i/K)^n - ((i-1)/K)^n.
    """
    x = np.sort(values[~np.isnan(values)])
    k = len(x)
    i = np.arange(1, k + 1)
    return float(x @ ((i / k) ** n - ((i - 1) / k) ** n))
