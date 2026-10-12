"""
Baseline: the best bonds by direct optimisation (literature study, Sec. 1.6).

Minimise the portfolio's variance over the formation year, optionally
requiring an average return at least that of the equal-weight portfolio of
all bonds. The covariance is the Ledoit-Wolf shrunk one (see
thesis.correlation). Two weighting rules (config.MIQP_WEIGHTINGS):

    equal       exactly m bonds at 1/m each: a mixed-integer quadratic
                programme (best_bonds)
    optimised   free weights between 0 and 1/m, summing to 1: long-only
                Markowitz minimum variance with a maximum weight of 1/m,
                a convex quadratic programme (best_weights)

Both are solved with Gurobi.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import gurobipy as gp
from gurobipy import GRB

from thesis import config
from thesis.backtest import evaluate_portfolio, evaluate_weighted_portfolio, split_by_year
from thesis.correlation import single_index_shrinkage


def best_bonds(returns: np.ndarray, m: int, objective: str,
               time_limit: float = config.MIQP_TIME_LIMIT) -> tuple[np.ndarray, float]:
    """
    The m bonds whose equal-weight portfolio had the lowest variance in
    returns (one formation year, days x bonds), and the optimality gap
    (0 if proven optimal; otherwise the best portfolio found in time_limit).

    objective
        "min_variance"                 variance only
        "min_variance_target_return"   also require an average return at least
                                       that of the equal-weight portfolio of all bonds
    """
    model, chosen = _solve(returns, m, objective, GRB.BINARY, time_limit)
    return np.flatnonzero(chosen.X > 0.5), float(model.MIPGap)


def best_weights(returns: np.ndarray, m: int, objective: str,
                 time_limit: float = config.MIQP_TIME_LIMIT) -> np.ndarray:
    """
    The same problem with free weights: each bond gets a weight between 0 and
    1/m (summing to 1), not exactly 0 or 1/m. This is long-only Markowitz
    minimum variance (or mean-variance, with the target return) with a
    maximum weight per bond of 1/m. It is a continuous, convex problem, so
    the solution is always the exact optimum.
    """
    model, chosen = _solve(returns, m, objective, GRB.CONTINUOUS, time_limit)
    weights = chosen.X / m
    weights[weights < 1e-9] = 0.0            # round solver noise to exactly 0
    return weights / weights.sum()


def _solve(returns: np.ndarray, m: int, objective: str, variable_type, time_limit: float):
    """
    Build and solve the model shared by best_bonds and best_weights; returns
    the model and z, the vector of chosen amounts (weight = z / m).

    The variance uses the shrunk covariance δF + (1 - δ)S. For z this needs
    no 745 x 745 matrix:
        zᵀSz = average over days of (x_t · z)²        x = centred daily returns
        zᵀFz = (cᵀz)² / v + Σ_i d_i z_i²              the single-index model
    with c the bonds' covariances with the market, v the market's variance and
    d_i the part of bond i's variance the market does not explain.
    """
    # Work in percent: variances of daily returns (~1e-6) are below the solver's
    # default precision. Rescaling does not change which portfolio is best.
    returns = 100 * returns
    n_days, n_bonds = returns.shape
    x = returns - returns.mean(axis=0)
    market = x.mean(axis=1)
    cov_market = x.T @ market / n_days
    var_market = market @ market / n_days
    own = (x ** 2).mean(axis=0) - cov_market ** 2 / var_market
    shrinkage, _ = single_index_shrinkage(returns)

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    chosen = model.addMVar(n_bonds, lb=0.0, ub=1.0, vtype=variable_type)
    daily = model.addMVar(n_days, lb=-GRB.INFINITY)          # x_t · z for each day
    with_market = model.addVar(lb=-GRB.INFINITY)            # cᵀz
    model.addConstr(chosen.sum() == m)
    model.addConstr(daily == x @ chosen)
    model.addConstr(with_market == cov_market @ chosen)
    if objective == "min_variance_target_return":
        model.addConstr(returns.mean(axis=0) @ chosen >= m * returns.mean())
    elif objective != "min_variance":
        raise ValueError(f"Unknown objective {objective!r}; expected one of {config.MIQP_OBJECTIVES}")

    sample = (daily @ daily) / n_days
    single_index = with_market * with_market / var_market + chosen @ sparse.diags(own) @ chosen
    model.setObjective((1 - shrinkage) * sample + shrinkage * single_index, GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"No portfolio found within {time_limit} s (m={m}, {objective})")
    return model, chosen


def run_miqp_baseline(returns: pd.DataFrame, rf_returns: pd.Series,
                      formation_years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    One row per (weighting, objective, portfolio size m): the portfolio chosen
    with data of formation year y is held during year y + 1 and scored like
    the graph strategies. bonds_held is the average number of bonds with a
    non-zero weight; max_gap the largest optimality gap over the years.
    """
    holding_years = [year + 1 for year in formation_years]
    formation_returns = split_by_year(returns, formation_years)
    returns_by_year = split_by_year(returns, holding_years)
    rf_by_year = split_by_year(rf_returns, holding_years)

    rows = []
    for weighting in config.MIQP_WEIGHTINGS:
        for objective in config.MIQP_OBJECTIVES:
            for m in config.PORTFOLIO_SIZES:
                if weighting == "equal":
                    holdings, gaps = {}, []
                    for year in formation_years:
                        bonds, gap = best_bonds(formation_returns[year], m, objective)
                        holdings[year + 1] = bonds
                        gaps.append(gap)
                    metrics = evaluate_portfolio(returns_by_year, rf_by_year, holdings)
                    bonds_held, max_gap = float(m), max(gaps)
                else:
                    weights = {year + 1: best_weights(formation_returns[year], m, objective)
                               for year in formation_years}
                    metrics = evaluate_weighted_portfolio(returns_by_year, rf_by_year, weights)
                    bonds_held = float(np.mean([np.count_nonzero(w) for w in weights.values()]))
                    max_gap = 0.0                # a convex problem: always solved exactly
                rows.append({"weighting": weighting, "objective": objective, "m": m,
                             "bonds_held": bonds_held, "max_gap": max_gap, **metrics})
                if verbose:
                    print(f"  {weighting}, {objective}, m={m}: SR={metrics['SR']:.3f}, "
                          f"{bonds_held:.0f} bonds held, max gap {max_gap:.2%}", flush=True)
    return pd.DataFrame(rows)
