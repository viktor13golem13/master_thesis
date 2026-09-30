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
| 0 | [thesis/data.py](thesis/data.py) | Rebuilds `train.csv` / `test.csv` from the raw Datastream downloads (`build_train_test`) | — |
| 1 | [thesis/data.py](thesis/data.py) | Loads prices, removes 200 unusable series (duplicates, cash, broken or stale data) | — |
| 2 | [thesis/correlation.py](thesis/correlation.py) | One correlation matrix per year from 125-day rolling windows; optional Ledoit–Wolf shrinkage | 3.1 |
| 3 | [thesis/graph.py](thesis/graph.py) | Correlation matrix → adjacency matrix (8 graph types, threshold θ) | 3.2–3.3 |
| 4 | [thesis/centrality.py](thesis/centrality.py) | Degree, Katz, Katz-min, exponential, their subgraph versions, NBTW (resolvent and exponential), betweenness | 2.1, 3.4 |
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

## How train.csv and test.csv were built

`thesis.data.build_train_test()` recreates both files from `dataset/raw_files/`,
and a test checks that the result is identical to the files on disk:

1. Join the four Datastream exports (`datase3`, `dataset1`, `dataset2`,
   `dataset4`), dropping the 11 `#ERROR` columns (series the licence could
   not download).
2. Add `new_bonds.csv` by date.
3. Keep only series with data on every day of 2017–2025 (this removes 31
   series that start later), minus two BPAM series removed by hand.
4. Split by year: 2017–2023 is train, 2024–2025 is test.

Constant and duplicate series were *not* removed at this stage; step 1
(`load_dataset`) removes them, together with other unusable series.

## The grid

| Setting | Values |
|---|---|
| Window weighting | uniform, paper (older windows weigh more, as printed), recent |
| Shrinkage | no, yes |
| Graph type | sign (positive / absolute) × loops × weighted = 8 |
| Threshold θ | 0.0, 0.1, …, 0.9 |
| Centrality | 59 variants (measure × parameter), plus 20 NBTW exponential variants on unweighted graphs |
| Selection | central, peripheral |
| Portfolio size m | 5, 10, 15, 20, 25, 30, 40, 50, 75, 100 |

In total 662,400 strategies, evaluated walk-forward: the correlations of
year *y* choose the bonds held in year *y + 1*, with equal weights.

## Differences from the paper

- **Bonds, not stocks**: the indices are in their own currencies and trade in
  different time zones, which lowers measured correlations for, e.g., Asian
  markets.
- **Held-out test set** (2024–2025), untouched while choosing strategies.
- **Graph types**: the paper's 4 transforms × 8 adjacency options reduce to
  8 distinct graphs, because every θ ≥ 0. The paper describes transform A3
  as "the positive part of the negative elements", but its A3 results
  (Tables 6–7) are identical to its |C| results (Tables 2–3), so A3 is
  treated as |C|. Read literally, A3 would connect only negatively
  correlated bonds; on bond data that graph is almost empty above θ = 0.2.
- **Ties** between equally central bonds (e.g. isolated nodes) are broken
  randomly with a fixed seed, not by column order.
- **Window weighting** is a parameter, because the printed formula gives older
  windows more weight, the opposite of Pozzi et al. (2013).
- **Portfolio sizes**: 10 values of *m* instead of *m* = 10.
- **NBTW exponential centralities only on unweighted graphs.** For weighted
  graphs the only known formula (Arrigo, Higham, Noferini & Wood 2024,
  Thm 4.6) works on an m × m edge matrix, with m up to ~550,000 directed
  edges here, which is not feasible. Unweighted graphs use the 2n × 2n
  formula of Arrigo, Grindrod, Higham & Noferini (2018, Thm 2.1).
- **Benchmark**: the Vanguard Total Bond Market ETF instead of the S&P 500
  index, plus the equal-weight portfolio of all bonds. The Vanguard series
  is a price series (it excludes coupons), so it understates the ETF's
  total return.
- **Only equal weights** so far; the paper's minimum-variance and
  mean-variance portfolios are optional future work.
