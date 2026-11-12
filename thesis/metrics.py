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

# Days per year used to annualise daily figures. The usual convention is 252
# trading days, but the Datastream files have a price for every weekday
# (holidays repeat the previous price): 260-262 days a year, 261 on average.
TRADING_DAYS_PER_YEAR = 261
VAR_LEVEL = 0.05        # tail probability for Value at Risk and CVaR

# Daily returns of bond indices are autocorrelated (lag 1 about 0.3), so the
# daily SD times √261 understates risk. The other volatility versions correct
# for that. The correction stops growing at about 20 lags.
LO_LAGS = 20             # autocorrelations beyond this lag are taken to be zero
DAYS_PER_MONTH = 22


# ------------------------------------------------------------ return and risk

def cumulative_return(portfolio_returns, risk_free_returns) -> float:
    """Total growth over the whole period: 1 euro becomes 1 + CR euros."""
    return float(np.prod(1.0 + portfolio_returns) - 1.0)


def expected_return(portfolio_returns, risk_free_returns) -> float:
    """Average daily return, annualised (ER)."""
    return float(np.mean(portfolio_returns) * TRADING_DAYS_PER_YEAR)


def volatility_daily(portfolio_returns, risk_free_returns) -> float:
    """Standard deviation of daily returns, annualised (SD)."""
    return float(np.std(portfolio_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def volatility_lo(portfolio_returns, risk_free_returns) -> float:
    """
    Annual volatility by Lo (2002) with q = 261 days (SD_Lo), taking the
    autocorrelations ρ_k beyond LO_LAGS to be zero:

        SD_Lo = SD · √(1 + 2 Σ_{k=1}^{LO_LAGS} (1 - k/q) ρ_k)

    with ρ_k = S_k / S_0, S_k = Σ_t x_t x_{t-k} and x_t the return minus its mean.
    NaN if the estimated autocorrelations make the expression negative.
    """
    x = portfolio_returns - np.mean(portfolio_returns)
    if not x @ x > 0:                       # constant returns: no volatility, no autocorrelation
        return 0.0
    q = TRADING_DAYS_PER_YEAR
    correction = 1.0
    for k in range(1, LO_LAGS + 1):
        rho_k = (x[k:] @ x[:-k]) / (x @ x)
        correction += 2 * (1 - k / q) * rho_k
    if correction < 0:
        return float("nan")
    return volatility_daily(portfolio_returns, risk_free_returns) * float(np.sqrt(correction))


def volatility_monthly(portfolio_returns, risk_free_returns) -> float:
    """SD of the returns of consecutive 22-day blocks, annualised (SD_M)."""
    return _block_volatility(portfolio_returns, DAYS_PER_MONTH)


def volatility_yearly(portfolio_returns, risk_free_returns) -> float:
    """
    SD of the returns of consecutive 261-day blocks (SD_Y). No autocorrelation
    is left at this scale, but there are few blocks: 6 in training, 2 in test.
    """
    return _block_volatility(portfolio_returns, TRADING_DAYS_PER_YEAR)


def _block_volatility(portfolio_returns, days: int) -> float:
    """SD of compounded returns over consecutive blocks of `days` days, annualised; leftover days are dropped."""
    n_blocks = len(portfolio_returns) // days
    if n_blocks < 2:
        return float("nan")
    blocks = portfolio_returns[:n_blocks * days].reshape(n_blocks, days)
    block_returns = np.prod(1.0 + blocks, axis=1) - 1.0
    return float(np.std(block_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR / days))


def sharpe_ratio_daily(portfolio_returns, risk_free_returns) -> float:
    """
    Excess return per unit of risk, annualised (SR):
    (E[r_p] - E[r_f]) / sd(r_p).
    """
    return _sharpe(portfolio_returns, risk_free_returns, volatility_daily(portfolio_returns, risk_free_returns))


def sharpe_ratio_lo(portfolio_returns, risk_free_returns) -> float:
    """Sharpe ratio with Lo's annual volatility (SR_Lo)."""
    return _sharpe(portfolio_returns, risk_free_returns, volatility_lo(portfolio_returns, risk_free_returns))


def sharpe_ratio_monthly(portfolio_returns, risk_free_returns) -> float:
    """Sharpe ratio with the monthly-block volatility (SR_M)."""
    return _sharpe(portfolio_returns, risk_free_returns, volatility_monthly(portfolio_returns, risk_free_returns))


def sharpe_ratio_yearly(portfolio_returns, risk_free_returns) -> float:
    """Sharpe ratio with the yearly-block volatility (SR_Y)."""
    return _sharpe(portfolio_returns, risk_free_returns, volatility_yearly(portfolio_returns, risk_free_returns))


def _sharpe(portfolio_returns, risk_free_returns, sd: float) -> float:
    """Annualised mean excess return over an annualised volatility; NaN if sd is 0 or NaN."""
    if not sd > 0:
        return float("nan")
    return float(np.mean(portfolio_returns - risk_free_returns) * TRADING_DAYS_PER_YEAR / sd)


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
    "SD": volatility_daily,
    "SR": sharpe_ratio_daily,
    "SD_Lo": volatility_lo,
    "SR_Lo": sharpe_ratio_lo,
    "SD_M": volatility_monthly,
    "SR_M": sharpe_ratio_monthly,
    "SD_Y": volatility_yearly,
    "SR_Y": sharpe_ratio_yearly,
    "VaR": value_at_risk,
    "CVaR": conditional_value_at_risk,
    "MaxDD": max_drawdown,
    "Omega": omega_ratio,
    "Sortino": sortino_ratio,
    "UP": upside_potential_ratio,
}


# Metrics where a smaller value is better (all others: larger is better).
LOWER_IS_BETTER = {"SD", "SD_Lo", "SD_M", "SD_Y", "MaxDD"}


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
