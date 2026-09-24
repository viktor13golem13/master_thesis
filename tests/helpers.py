import numpy as np

# Toy correlation matrix (3.2) of Arslan, Noferini & Vrontos (2024).
C_TOY = np.array([
    [1, -0.1378, 0.2025, 0.4683, -0.2583],
    [-0.1378, 1, 0.4373, 0.1050, -0.1738],
    [0.2025, 0.4373, 1, 0.4245, 0.4108],
    [0.4683, 0.1050, 0.4245, 1, -0.0465],
    [-0.2583, -0.1738, 0.4108, -0.0465, 1],
])


def random_weighted_graph(n, seed, loops=False, density=0.6):
    """A random symmetric adjacency matrix with weights in [0, 1)."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n)) * (rng.random((n, n)) < density)
    A = np.triu(A, 1)
    A = A + A.T
    if loops:
        A[np.diag_indices(n)] = rng.random(n)
    return A
