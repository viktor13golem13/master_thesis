import numpy as np

TRADING_DAYS = 252


def lower_partial_moment(excess: np.ndarray, n: int) -> float:
    """LPM_n = E[((r_b - r_p)_+)^n], given excess = r_p - r_b."""
    return float(np.mean(np.maximum(-excess, 0.0) ** n))


def performance_metrics(rp: np.ndarray, rf: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """
    Out-of-sample performance of a daily portfolio return series.

    Follows Section 2.3 of Arslan, Noferini & Vrontos (2024); the risk-free
    rate also serves as the benchmark r_b for Omega, Sortino and Upside
    Potential.

    Parameters
    ----------
    rp : np.ndarray
        Daily simple portfolio returns.
    rf : np.ndarray
        Daily simple risk-free returns on the same days.
    alpha : float
        Tail probability for VaR and CVaR (default 5%).

    Returns
    -------
    dict with keys
        CR      cumulative return over the period (1.0 means +100%)
        ER      annualised mean return
        SD      annualised volatility
        SR      annualised Sharpe ratio, (E[r_p] - E[r_f]) / sd(r_p)
        VaR     daily Value at Risk at level alpha (a loss is negative)
        CVaR    daily Conditional VaR at level alpha
        MaxDD   maximum drawdown of the portfolio value (fraction of peak)
        Omega   K_1 + 1
        Sortino K_2
        UP      E[(r_p - r_b)_+] / sqrt(LPM_2)
    """
    r = np.asarray(rp, dtype=float)
    excess = r - np.asarray(rf, dtype=float)

    value = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], value]))[1:]
    var = float(np.quantile(r, alpha))
    sd = float(np.std(r, ddof=1))
    lpm1 = lower_partial_moment(excess, 1)
    lpm2 = lower_partial_moment(excess, 2)

    return {
        "CR": float(value[-1] - 1.0),
        "ER": float(np.mean(r) * TRADING_DAYS),
        "SD": float(sd * np.sqrt(TRADING_DAYS)),
        "SR": float(np.mean(excess) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "VaR": var,
        "CVaR": float(r[r <= var].mean()),
        "MaxDD": float(np.max(1.0 - value / peak)),
        "Omega": float(np.mean(excess) / lpm1 + 1.0) if lpm1 > 0 else float("nan"),
        "Sortino": float(np.mean(excess) / np.sqrt(lpm2)) if lpm2 > 0 else float("nan"),
        "UP": float(np.mean(np.maximum(excess, 0.0)) / np.sqrt(lpm2)) if lpm2 > 0 else float("nan"),
    }
