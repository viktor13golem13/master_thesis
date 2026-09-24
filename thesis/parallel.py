"""
Running a function on many items using all CPU cores.

This file only makes the grid search fast; it contains no thesis logic.
"""
import multiprocessing as mp

# Data that every worker needs (large arrays). It is set just before the
# worker processes start; with the "fork" start method they inherit it from
# the parent process instead of receiving a copy per task.
_shared = None


def _call(task):
    function, item = task
    return function(item, _shared)


def parallel_map(function, items, shared, processes: int | None = None):
    """
    Yield function(item, shared) for every item, computed in parallel.

    Results arrive in completion order, not in the order of items.
    function must be defined at module level (so worker processes can find it).
    """
    global _shared
    _shared = shared
    with mp.get_context("fork").Pool(processes) as pool:
        yield from pool.imap_unordered(_call, [(function, item) for item in items])
