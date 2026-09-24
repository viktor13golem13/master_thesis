# Graph centrality portfolios on bond indices

Master's thesis, KU Leuven (supervisors: Prof. Vannieuwenhoven and Prof. Noferini).

This code replicates Arslan, Noferini & Vrontos (2024), *Portfolio Management
Using Graph Centralities: Review and Comparison*, on 745 FTSE and iBoxx bond
indices instead of S&P 500 stocks, and adds a held-out test set
(2024–2025), which the paper does not have.

## The idea

1. Measure how bond indices move together: one correlation matrix per year.
2. Turn the correlations into a graph: bonds are nodes, strongly correlated
   bonds are joined by an edge.
3. Score how *central* each bond is in that graph.
4. Invest next year in the *m* most central, or most peripheral, bonds.
5. Repeat for every combination of settings and see what works.

## Running it

```bash
python3 -m venv --without-pip .venv          # the system has no python3-venv
.venv/bin/python get-pip.py                  # from https://bootstrap.pypa.io/get-pip.py
.venv/bin/pip install -r requirements.txt

.venv/bin/python main.py                     # grid search on the training years (~25 min, 16 cores)
.venv/bin/python -m pytest                   # tests (~1 s)
```

`main.py` writes `results_train.parquet` (one row per strategy, with every
metric) and `dataset/cleaning_report.csv` (removed series and why).

## The code

The pipeline runs top to bottom in this order:

| Step | File | What it does | Paper |
|---|---|---|---|
| 1 | [thesis/data.py](thesis/data.py) | Loads prices, removes 200 unusable series (duplicates, cash, broken or stale data) | — |
| 2 | [thesis/correlation.py](thesis/correlation.py) | One correlation matrix per year from 125-day rolling windows; optional Ledoit–Wolf shrinkage | 3.1 |
| 3 | [thesis/graph.py](thesis/graph.py) | Correlation matrix → adjacency matrix (10 graph types, threshold θ) | 3.2–3.3 |
| 4 | [thesis/centrality.py](thesis/centrality.py) | Degree, Katz, Katz-min, exponential, their subgraph versions, NBTW, betweenness | 2.1, 3.4 |
| 5 | [thesis/backtest.py](thesis/backtest.py) | Holds the chosen bonds for a year, produces daily portfolio returns | 2.2, 4 |
| | [thesis/metrics.py](thesis/metrics.py) | Sharpe, Sortino, drawdown, VaR, … — one function per metric | 2.3 |
| 6 | [thesis/grid_search.py](thesis/grid_search.py) | Runs steps 2–5 for every combination of settings | 4 |

Supporting files:

- [thesis/config.py](thesis/config.py) — **every setting of the experiment**: file paths, cleaning
  rules, thresholds, centrality parameters, portfolio sizes, and the metric
  that ranks strategies (`RANK_BY`).
- [thesis/parallel.py](thesis/parallel.py) — runs the grid search on all CPU cores.
- [thesis/plotting.py](thesis/plotting.py) — draws a graph, for figures.
- [tests/](tests/) — checks the code against the paper (e.g. its toy example in
  Figure 2) and against brute-force computations.

Two conventions keep the pieces interchangeable:

- every centrality is `centrality(graph, parameter) -> scores`
- every metric is `metric(portfolio_returns, risk_free_returns) -> float`

## The grid

| Setting | Values |
|---|---|
| Window weighting | uniform, paper (older windows weigh more, as printed), recent |
| Shrinkage | no, yes |
| Graph type | sign (positive / negative / absolute) × loops × weighted = 10 |
| Threshold θ | 0.0, 0.1, …, 0.9 |
| Centrality | 59 variants (measure × parameter) |
| Selection | central, peripheral |
| Portfolio size m | 5, 10, 15, 20, 25, 30, 40, 50, 75, 100 |

In total 708,000 strategies, evaluated walk-forward: the correlations of
year *y* choose the bonds held in year *y + 1*, with equal weights.

## Differences from the paper

- **Bonds, not stocks**: the indices are in their own currencies and trade in
  different time zones, which lowers measured correlations for, e.g., Asian
  markets.
- **Held-out test set** (2024–2025), untouched while choosing strategies.
- **Graph types**: the paper's 4 transforms × 8 adjacency options reduce to
  10 distinct graphs, because every θ ≥ 0.
- **Ties** between equally central bonds (e.g. isolated nodes) are broken
  randomly with a fixed seed, not by column order.
- **Window weighting** is a parameter, because the printed formula gives older
  windows more weight, the opposite of Pozzi et al. (2013).
- **Portfolio sizes**: 10 values of *m* instead of *m* = 10.
- **Only equal weights** so far; the paper's minimum-variance and
  mean-variance portfolios are optional future work.
