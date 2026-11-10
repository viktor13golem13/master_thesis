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

.venv/bin/python main.py                     # grid search on the training years (~80 min, 10 workers)
.venv/bin/python -m pytest                   # tests (~1 s)
```

`main.py` writes `results_train.parquet` (one row per strategy, with every
metric), `strategy_returns_train.npy` (daily returns of each of the ~156,000
distinct portfolios; the `series` column of a row points to its column) and
`dataset/cleaning_report.csv` (removed series and why).

## The code

The pipeline runs top to bottom in this order:

| Step | File | What it does | Paper |
|---|---|---|---|
| 0 | [thesis/data.py](thesis/data.py) | Rebuilds `train.csv` / `test.csv` from the raw Datastream downloads (`build_train_test`) | — |
| 1 | [thesis/data.py](thesis/data.py) | Loads prices, removes 200 unusable series (duplicates, cash, broken or stale data) | — |
| 2 | [thesis/correlation.py](thesis/correlation.py) | One correlation matrix per year from all 125-day rolling windows in the year; optional Ledoit–Wolf shrinkage | 3.1 |
| 3 | [thesis/graph.py](thesis/graph.py) | Correlation (or covariance) matrix → adjacency matrix (8 graph types, threshold θ) | 3.2–3.3 |
| 4 | [thesis/centrality.py](thesis/centrality.py) | Degree, Katz, Katz-min, exponential, their subgraph versions, NBTW (resolvent and exponential), betweenness | 2.1, 3.4 |
| 5 | [thesis/backtest.py](thesis/backtest.py) | Holds the chosen bonds for a year, produces daily portfolio returns | 2.2, 4 |
| | [thesis/metrics.py](thesis/metrics.py) | Sharpe, Sortino, drawdown, VaR, … — one function per metric | 2.3 |
| 6 | [thesis/grid_search.py](thesis/grid_search.py) | Runs steps 2–5 for every combination of settings | 4 |

Supporting files:

- [thesis/config.py](thesis/config.py) — **every setting of the experiment**: file paths, cleaning
  rules, thresholds, centrality parameters, portfolio sizes, and the metric
  that ranks strategies (`RANK_BY`).
- [thesis/parallel.py](thesis/parallel.py) — runs the grid search on all CPU cores.
- [thesis/simple_rules.py](thesis/simple_rules.py) — ranks bonds without a graph (average correlation,
  average covariance, volatility), evaluated alongside the grid: does the graph add anything?
- [thesis/plotting.py](thesis/plotting.py) — draws a graph, for figures.
- [thesis/random_baseline.py](thesis/random_baseline.py) and [random_baseline.py](random_baseline.py) — 10,000
  random portfolios per size m, walked forward like the graph strategies: what luck alone achieves.
- [thesis/seed_check.py](thesis/seed_check.py) and [seed_check.py](seed_check.py) — reruns the best
  strategies with random tie-breaking, to show how much luck that would bring in.
- [thesis/overfitting.py](thesis/overfitting.py) and [overfitting.py](overfitting.py) — is the best strategy
  just the luckiest of many? Deflated Sharpe Ratio (Bailey & López de Prado 2014), Probability of
  Backtest Overfitting (Bailey, Borwein, López de Prado & Zhu 2017) and the expected best of n
  random portfolios. The trials are the distinct peripheral graph strategies (the paper's hypothesis,
  fixed before seeing the bond data); all graph strategies are shown as a sensitivity check. Training
  years only.
- [thesis/miqp.py](thesis/miqp.py) and [miqp_baseline.py](miqp_baseline.py) — the MIQP baseline from the
  literature study: the m bonds with the lowest equal-weight variance, chosen by
  mixed-integer optimisation with Gurobi. Run `.venv/bin/python miqp_baseline.py`; it needs a
  Gurobi licence (students get a free named-user academic licence at gurobi.com,
  activated with `grbgetkey` from the university network or VPN).
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
| Matrix | correlation (paper), covariance divided by the largest variance |
| Window weighting | uniform, paper (older windows weigh more, as printed), recent |
| Shrinkage | no, yes |
| Graph type | sign (positive / absolute) × loops × weighted = 8 |
| Threshold θ | 0.0, 0.1, …, 0.9 |
| Centrality | 59 variants (measure × parameter), plus 20 NBTW exponential variants on unweighted graphs |
| Selection | central, peripheral |
| Portfolio size m | 10, 15, 20, 25, 30, 40, 50, 75, 100 |

In total 1,192,320 graph strategies, plus 234 from the simple rules,
evaluated walk-forward: the correlations of year *y* choose the bonds held
in year *y + 1*, with equal weights.

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
- **Ties** between equally central bonds (e.g. isolated nodes) for the last
  places of a portfolio are not broken at all: those places are shared
  evenly by all tied bonds (`backtest.select_weights`), the average portfolio
  over every random tie-break. With random tie-breaking, the best training
  strategies turned out to be mostly lucky draws: rechecked with 20 other
  seeds, the median Sharpe ratio (SR_Lo) of the top 100 fell from 0.66 to
  −0.13 (`seed_check.py`). Even 2% isolated bonds is about 15 tied bonds,
  so "the m most peripheral bonds" is often a choice among tied bonds; such
  a portfolio then holds more than m bonds (column `bonds_held`).
- **Correlation windows**: every 125-day window that fits in the year
  (about 137) is averaged, not only the last 125.
- **Volatility**: daily bond-index returns are autocorrelated, so the
  daily SD understates risk. Besides the paper's daily SD and Sharpe ratio,
  every strategy also gets a Lo (2002) version (SD_Lo / SR_Lo: q = 261,
  autocorrelations beyond lag 20 taken as zero), a monthly-block and a
  yearly-block version (SD_M / SR_M, SD_Y / SR_Y). Strategies are ranked by
  SR_Lo; `choose_volatility.py` shows why (training years only).
- **Covariance graphs** in addition to correlation graphs. Weighted degree in
  a covariance graph is proportional to a bond's contribution to the variance
  of the equal-weight portfolio, which links the graph to minimum variance.
  The covariance matrix is divided by the largest variance (entries stay in
  [−1, 1]), but covariances are so much smaller than correlations that a
  fixed θ ≥ 0.1 leaves the graph almost empty. A covariance graph therefore
  keeps as many edges as the correlation graph at the same θ, the strongest
  ones, so both are compared at equal density.
- **Simple rules without a graph** (lowest average correlation, lowest
  contribution to equal-weight variance, lowest volatility) as a baseline.
- **Window weighting** is a parameter, because the printed formula gives older
  windows more weight, the opposite of Pozzi et al. (2013).
- **Portfolio sizes**: 9 values of *m* from 10 (the paper's only value) to 100.
- **NBTW exponential centralities only on unweighted graphs.** For weighted
  graphs the only known formula (Arrigo, Higham, Noferini & Wood 2024,
  Thm 4.6) works on an m × m edge matrix, with m up to ~550,000 directed
  edges here, which is not feasible. Unweighted graphs use the 2n × 2n
  formula of Arrigo, Grindrod, Higham & Noferini (2018, Thm 2.1).
- **NBTW's α** is a fraction of 1/ρ(A), as for Katz, instead of a fraction of the NBTW
  convergence radius μ (which needs a numerical search); every such α is valid because μ > 1/ρ(A).
- **Benchmark**: the Vanguard Total Bond Market ETF instead of the S&P 500
  index, plus the equal-weight portfolio of all bonds. The Vanguard series
  is a price series (it excludes coupons), so it understates the ETF's
  total return.
- **Only equal weights** so far; the paper's minimum-variance and
  mean-variance portfolios are optional future work.
