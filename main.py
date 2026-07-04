import pandas as pd

from centrality import katz_centrality, exponential_centrality
from search import hyperparameter_search

TRAIN_PATH = "train.csv"
CACHE_PATH = "cached_averages.pkl"

PARAM_GRID = {
    "weighted":       [False, True],
    "relu_threshold": [0.0, 0.3, 0.5, 0.7],
    "centrality_fn":  [katz_centrality, exponential_centrality],
    "top_n":          [5, 10, 20],
    "use_top":        [True, False],
}

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_PATH)
    best_params, results = hyperparameter_search(df, PARAM_GRID, cache_path=CACHE_PATH)
