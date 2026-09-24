"""
Step 5b — Performance metrics (paper, Sec. 2.3).

Every metric has the same form:

    metric(portfolio_returns, risk_free_returns) -> float

where both arguments are arrays of daily simple returns on the same days.
Metrics that do not need the risk-free rate simply ignore it. Because all
metrics are interchangeable, choosing which one ranks the strategies is a
matter of changing config.RANK_BY.

As in the paper, the risk-free rate is also the benchmark r_b for the
Omega, Sortino and Upside Potential ratios.
"""
import numpy as np

TRADING_DAYS_PER_YEAR = 252
VAR_LEVEL = 0.05        # tail probability for Value at Risk and CVaR


# ------------------------------------------------------------ return and risk

def cumulative_return(portfolio_returns, risk_free_returns) -> float:
    """Total growth over the whole period: 1 euro becomes 1 + CR euros."""
    return float(np.prod(1.0 + portfolio_returns) - 1.0)


def expected_return(portfolio_returns, risk_free_returns) -> float:
    """Average daily return, annualised (ER)."""
    return float(np.mean(portfolio_returns) * TRADING_DAYS_PER_YEAR)


def volatility(portfolio_returns, risk_free_returns) -> float:
    """Standard deviation of daily returns, annualised (SD)."""
    return float(np.std(portfolio_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(portfolio_returns, risk_free_returns) -> float:
    """
    Excess return per unit of risk, annualised (SR):
    (E[r_p] - E[r_f]) / sd(r_p).
    """
    sd = np.std(portfolio_returns, ddof=1)
    if sd == 0:
        return float("nan")
    excess = np.mean(portfolio_returns - risk_free_returns)
    return float(excess / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


# ---------------------------------------------------------------- tail risk

def value_at_risk(portfolio_returns, risk_free_returns) -> float:
    """Daily return exceeded on all but the worst VAR_LEVEL of days (a loss is negative)."""
    return float(np.quantile(portfolio_returns, VAR_LEVEL))


def conditional_value_at_risk(portfolio_returns, risk_free_returns) -> float:
    """Average daily return on the worst VAR_LEVEL of days (CVaR)."""
    var = np.quantile(portfolio_returns, VAR_LEVEL)
    return float(np.mean(portfolio_returns[portfolio_returns <= var]))


def max_drawdown(portfolio_returns, risk_free_returns) -> float:
    """Largest fall of the portfolio value from a previous peak, as a fraction of that peak."""
    value = np.cumprod(1.0 + portfolio_returns)
    peak = np.maximum.accumulate(np.concatenate([[1.0], value]))[1:]
    return float(np.max(1.0 - value / peak))


# ------------------------------------------------------ downside-aware ratios

def _lower_partial_moment(portfolio_returns, risk_free_returns, n: int) -> float:
    """LPM_n = E[((r_b - r_p)_+)^n]: average shortfall below the benchmark, to the power n."""
    shortfall = np.maximum(risk_free_returns - portfolio_returns, 0.0)
    return float(np.mean(shortfall ** n))


def omega_ratio(portfolio_returns, risk_free_returns) -> float:
    """Omega = K_1 + 1: mean excess return over the mean shortfall, plus one."""
    lpm1 = _lower_partial_moment(portfolio_returns, risk_free_returns, 1)
    if lpm1 == 0:
        return float("nan")
    return float(np.mean(portfolio_returns - risk_free_returns) / lpm1 + 1.0)


def sortino_ratio(portfolio_returns, risk_free_returns) -> float:
    """Sortino = K_2: mean excess return over the root-mean-square shortfall (daily)."""
    lpm2 = _lower_partial_moment(portfolio_returns, risk_free_returns, 2)
    if lpm2 == 0:
        return float("nan")
    return float(np.mean(portfolio_returns - risk_free_returns) / np.sqrt(lpm2))


def upside_potential_ratio(portfolio_returns, risk_free_returns) -> float:
    """Upside Potential: mean gain above the benchmark over the root-mean-square shortfall."""
    lpm2 = _lower_partial_moment(portfolio_returns, risk_free_returns, 2)
    if lpm2 == 0:
        return float("nan")
    upside = np.maximum(portfolio_returns - risk_free_returns, 0.0)
    return float(np.mean(upside) / np.sqrt(lpm2))


# ------------------------------------------------------------- registry

# Column name in the results table -> metric function.
METRICS = {
    "CR": cumulative_return,
    "ER": expected_return,
    "SD": volatility,
    "SR": sharpe_ratio,
    "VaR": value_at_risk,
    "CVaR": conditional_value_at_risk,
    "MaxDD": max_drawdown,
    "Omega": omega_ratio,
    "Sortino": sortino_ratio,
    "UP": upside_potential_ratio,
}


# Metrics where a smaller value is better (all others: larger is better).
LOWER_IS_BETTER = {"SD", "MaxDD"}


def metric_name(metric) -> str:
    """The results-table column name of a metric function, e.g. sharpe_ratio -> 'SR'."""
    for name, function in METRICS.items():
        if function is metric:
            return name
    raise ValueError(f"{metric.__name__} is not listed in METRICS")


def all_metrics(portfolio_returns, risk_free_returns) -> dict[str, float]:
    """Every metric in METRICS for one daily return series."""
    rp = np.asarray(portfolio_returns, dtype=float)
    rf = np.asarray(risk_free_returns, dtype=float)
    return {name: metric(rp, rf) for name, metric in METRICS.items()}
