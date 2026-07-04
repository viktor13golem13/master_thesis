import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def save_graph(path: str, dates: np.ndarray, col_names: list[str], matrices: np.ndarray) -> None:
    """Save dates, col_names and matrices to a .npz file."""
    np.savez(path, dates=dates, col_names=np.array(col_names), matrices=matrices)


def load_graph(path: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load dates, col_names and matrices from a .npz file."""
    data = np.load(path, allow_pickle=False)
    return data["dates"], data["col_names"].tolist(), data["matrices"]


def visualise_graph(
    matrix: np.ndarray,
    col_names: list[str],
    title: str = "Correlation Graph",
    figsize: tuple[int, int] = (20, 20),
    display_threshold: float = 0.0,
    save_path: str | None = None,
) -> None:
    """
    Visualise a weighted correlation graph from a symmetric adjacency matrix.

    Parameters
    ----------
    matrix : np.ndarray of shape (n, n)
        Symmetric matrix where 0 means no edge and non-zero values are edge
        weights (expected in [0, 1]).
    col_names : list of str, length n
        Node labels corresponding to matrix rows/columns.
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.
    display_threshold : float
        Edges with weight below this value are hidden (does not affect the data).
    save_path : str or None
        If provided, saves the figure to this path instead of displaying it.
    """
    G = nx.Graph()
    n = len(col_names)
    G.add_nodes_from(range(n))

    rows, cols = np.triu_indices(n, k=1)
    for i, j in zip(rows, cols):
        w = matrix[i, j]
        if w > display_threshold:
            G.add_edge(i, j, weight=w)

    pos = nx.kamada_kawai_layout(G, weight="weight")

    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    edge_widths = 1 + weights * 4

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="steelblue", alpha=0.9, ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color=weights,
        edge_cmap=plt.cm.YlOrRd,
        edge_vmin=0, edge_vmax=1,
        alpha=0.7,
        ax=ax,
    )

    labels = {i: name.split(" - ")[0][:30] for i, name in enumerate(col_names)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, ax=ax, label="Correlation weight", shrink=0.6)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()
