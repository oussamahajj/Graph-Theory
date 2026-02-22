"""Utilities for loading and inspecting the Cora citation dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import random

import networkx as nx
import pandas as pd


def load_cora(content_path: str | Path, cites_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load Cora content/cites files.

    Returns:
        features: DataFrame indexed by paper_id
        labels: Series indexed by paper_id
        edges: DataFrame with columns [source, target]
    """
    content_path = Path(content_path)
    cites_path = Path(cites_path)

    if not content_path.exists():
        raise FileNotFoundError(f"Missing file: {content_path}")
    if not cites_path.exists():
        raise FileNotFoundError(f"Missing file: {cites_path}")

    content = pd.read_csv(content_path, sep=r"\s+", header=None)
    if content.shape[1] < 3:
        raise ValueError("cora.content format looks invalid (expected: id, features..., label)")

    paper_ids = content.iloc[:, 0].astype(str)
    labels = content.iloc[:, -1].astype(str)
    features = content.iloc[:, 1:-1]

    features.index = paper_ids
    labels.index = paper_ids

    edges = pd.read_csv(cites_path, sep=r"\s+", header=None, names=["source", "target"])
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    valid_ids = set(features.index)
    edges = edges[edges["source"].isin(valid_ids) & edges["target"].isin(valid_ids)].reset_index(drop=True)

    return features, labels, edges


def describe_cora(features: pd.DataFrame, labels: pd.Series, edges: pd.DataFrame) -> dict:
    """Return basic descriptive stats."""
    nodes = len(features)
    undirected_edges = len({tuple(sorted(e)) for e in edges[["source", "target"]].itertuples(index=False, name=None)})
    classes = labels.nunique()
    avg_degree = (2 * undirected_edges / nodes) if nodes else 0.0

    return {
        "num_nodes": nodes,
        "num_features": features.shape[1],
        "num_edges_undirected": undirected_edges,
        "num_classes": classes,
        "avg_degree": avg_degree,
    }


def train_test_split_edges(G: nx.Graph, test_frac: float = 0.1, seed: int = 42):
    """Split edges for link prediction while keeping the training graph connected.

    Fast strategy:
    - Build one spanning tree and keep all its edges in training.
    - Sample test edges only from non-tree edges, so connectivity is preserved.
    """
    random.seed(seed)

    if G.number_of_edges() == 0:
        return G.copy(), [], []

    # Keep all spanning-tree edges to guarantee train connectivity.
    tree = nx.minimum_spanning_tree(G)
    tree_edges = {tuple(sorted(e)) for e in tree.edges()}
    all_edges = [tuple(sorted(e)) for e in G.edges()]
    removable = [e for e in all_edges if e not in tree_edges]

    num_test_requested = int(len(all_edges) * test_frac)
    num_test = min(num_test_requested, len(removable))
    test_edges = random.sample(removable, k=num_test) if num_test > 0 else []

    G_train = G.copy()
    G_train.remove_edges_from(test_edges)

    # Negative samples for test: same number as positive test edges.
    test_neg = set()
    nodes = list(G.nodes())
    while len(test_neg) < len(test_edges):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and (u, v) not in test_neg and (v, u) not in test_neg:
            test_neg.add((u, v))

    return G_train, test_edges, list(test_neg)
