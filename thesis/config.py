"""
Every setting of the experiment, in pipeline order.

Change a value here and rerun main.py; no other file needs to change.
"""
from thesis.centrality import (
    betweenness, degree, exponential, exponential_subgraph, katz, katz_min,
    katz_subgraph, nbtw, nbtw_exponential, nbtw_exponential_subgraph, nbtw_subgraph,
)
from thesis.graph import GraphType
from thesis.metrics import sharpe_ratio

# --------------------------------------------------------------------- files
TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
BENCHMARK_PATH = "dataset/raw_files/vanguard.csv"   # Vanguard Total Bond Market ETF, the paper's S&P 500 analogue
CLEANING_REPORT_PATH = "dataset/cleaning_report.csv"
TRAIN_RESULTS_PATH = "results_train.parquet"

# ------------------------------------------------ 0. building train and test
# Datastream exports that share the same dates, in the order they are joined.
DATASTREAM_EXPORTS = [
    "dataset/raw_files/datase3.csv",
    "dataset/raw_files/dataset1.csv",
    "dataset/raw_files/dataset2.csv",
    "dataset/raw_files/dataset4.csv",
]
# A later, larger download with slightly different dates, joined by date.
LATER_EXPORT = "dataset/raw_files/new_bonds.csv"

TRAIN_YEARS = (2017, 2023)
TEST_YEARS = (2024, 2025)

# Removed by hand when the files were first built. They pass every other rule,
# but, like the BPAM BNM BOND index that was kept, they stopped moving in
# September 2016 (the stale-series rule in step 1 removes all three anyway).
REMOVED_BY_HAND = [
    "FTSE BPAM BNM MIXED IDX - TOT RETURN IND",
    "FTSE BPAM BNM SUKUK IDX - TOT RETURN IND",
]

# ------------------------------------------------------------------ 1. data
RISK_FREE_SERIES = "FTSE 3-Month Treasury Bill Index - Total Return"

# Money-market series: not bonds, so never investable.
CASH_PATTERN = r"(?i)treasury bill|eurodeposit|monetary rate"

# A daily move larger than this is treated as a data glitch ...
GLITCH_THRESHOLD = 0.20
# ... except during genuine market events, here the September 2022 UK gilt
# (LDI) crisis, when long gilts really did move 20-35% in a day.
REAL_EVENT_WINDOWS = [("2022-09-23", "2022-10-14")]

# A series with more zero daily returns than this in any year is stale.
STALE_THRESHOLD = 0.20

# ----------------------------------------------------------- 2. correlation
WINDOW_LENGTH = 125                                  # trading days (paper, Sec. 3.1)
WINDOW_WEIGHTINGS = ("uniform", "paper", "recent")   # see correlation.window_weights
SHRINKAGE_OPTIONS = (False, True)                    # Ledoit-Wolf shrinkage (paper, Sec. 3.1)

# ----------------------------------------------------------------- 3. graph
THRESHOLDS = tuple(k / 10 for k in range(10))        # θ = 0.0, 0.1, ..., 0.9 (paper, Sec. 3.3)

# The paper's 4 transforms x 8 adjacency options give only these 10
# distinct graphs, because every threshold is >= 0 (see graph.GraphType).
GRAPH_TYPES = [
    GraphType(sign, loops, weighted)
    for sign in ("positive", "negative", "absolute")
    for loops in (False, True)
    for weighted in (True, False)
    if not (sign == "negative" and loops)            # (-C)_+ has a zero diagonal
]

# ------------------------------------------------------------ 4. centrality
KATZ_FRACTIONS = tuple(k / 10 for k in range(1, 10))      # α = f / ρ(A)   (paper, Sec. 3.4)
NBTW_FRACTIONS = tuple(k / 10 for k in range(1, 10))      # α = f · μ
EXPONENTIAL_BETAS = tuple(k / 10 for k in range(1, 11))   # β in e^{βA}, maximum 1

# Every (centrality function, parameter) pair tried on each graph.
# The order matters only for reproducibility: ties between bonds are broken
# with random numbers drawn in this order, so new measures go at the end.
# The NBTW exponential measures exist only for unweighted graphs; on
# weighted graphs they are skipped.
CENTRALITIES = [
    (degree, None),
    *[(measure, f) for f in KATZ_FRACTIONS for measure in (katz, katz_subgraph)],
    (katz_min, None),
    *[(measure, b) for b in EXPONENTIAL_BETAS for measure in (exponential, exponential_subgraph)],
    *[(measure, f) for f in NBTW_FRACTIONS for measure in (nbtw, nbtw_subgraph)],
    (betweenness, None),
    *[(measure, b) for b in EXPONENTIAL_BETAS for measure in (nbtw_exponential, nbtw_exponential_subgraph)],
]

# ------------------------------------------------------------- 5. backtest
PORTFOLIO_SIZES = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100)   # m, number of bonds held
SELECTIONS = ("central", "peripheral")

# ---------------------------------------------------------- 6. grid search
# Metric used to sort the results; any function from thesis.metrics.METRICS.
RANK_BY = sharpe_ratio
