"""
Graph centrality portfolios on bond indices — a replication of
Arslan, Noferini & Vrontos (2024), "Portfolio Management Using Graph
Centralities: Review and Comparison", on FTSE and iBoxx bond indices.

The pipeline, in order:

    data.py         1. load prices, remove unusable series
    correlation.py  2. one correlation matrix per year
    graph.py        3. correlation matrix -> graph
    centrality.py   4. rank bonds by how central they are in the graph
    backtest.py     5. hold the chosen bonds for a year, measure the result
    metrics.py         (the performance measures used in step 5)
    grid_search.py  6. repeat for every combination of settings

    config.py       every setting of the experiment
    parallel.py     runs step 6 on all CPU cores
    plotting.py     draws a graph (for figures only)
"""
