from dataclasses import dataclass

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class GraphType:
    """
    How a correlation matrix C is turned into an adjacency matrix.

    sign
        "positive" — edges for C_ij > θ (paper transforms A1 and A2)
        "negative" — edges for -C_ij > θ (paper transform A3, (-C)_+)
        "absolute" — edges for |C_ij| > θ (paper transform A4, or A1 with |C|)
    loops
        Keep the diagonal (paper options 1, 2, 5, 6) or drop it (3, 4, 7, 8).
    weighted
        Edge weight is the correlation value (options 5-8) or 1 (options 1-4).

    Since every threshold θ is >= 0, the paper's 4 transforms x 8 options give
    only these 10 distinct graphs: the negative part has a zero diagonal, so
    it never has loops.
    """
    sign: str
    loops: bool
    weighted: bool

    @property
    def name(self) -> str:
        return f"{self.sign}{'_loops' if self.loops else ''}{'_weighted' if self.weighted else '_unweighted'}"


GRAPH_TYPES = [
    GraphType(sign, loops, weighted)
    for sign in ("positive", "negative", "absolute")
    for loops in (False, True)
    for weighted in (True, False)
    if not (sign == "negative" and loops)
]


def adjacency_matrix(C: np.ndarray, threshold: float, graph_type: GraphType) -> np.ndarray:
    """
    Adjacency matrix A = [T > θ] ∘ T (weighted) or A = [T > θ] (unweighted),
    where T is C, -C or |C| according to graph_type.sign, with its diagonal
    set to 0 unless graph_type.loops (Arslan, Noferini & Vrontos 2024, Sec. 3.2).
    """
    if graph_type.sign == "positive":
        T = C.copy()
    elif graph_type.sign == "negative":
        T = -C
    elif graph_type.sign == "absolute":
        T = np.abs(C)
    else:
        raise ValueError(f"Unknown sign {graph_type.sign!r}")
    if not graph_type.loops:
        np.fill_diagonal(T, 0.0)
    edges = T > threshold
    return np.where(edges, T, 0.0) if graph_type.weighted else edges.astype(float)


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
