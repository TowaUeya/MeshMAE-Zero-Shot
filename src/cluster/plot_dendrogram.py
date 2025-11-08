"""Plot hierarchical clustering dendrograms for fossil mesh embeddings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = {"ward", "average", "single", "complete"}


def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).values
    return np.load(path)


def load_metadata(path: Optional[Path], n: int) -> pd.DataFrame:
    if path and path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame({"sample_id": np.arange(n)})
    if "sample_id" not in df.columns:
        df["sample_id"] = df.index.astype(str)
    return df


def plot_dendrogram(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    method: str,
    output_path: Path,
    truncate: Optional[int] = None,
) -> None:
    distances = pdist(embeddings, metric="euclidean")
    linkage_matrix = linkage(distances, method=method)
    ordered = optimal_leaf_ordering(linkage_matrix, distances)

    labels = metadata["sample_id"].tolist()
    plt.figure(figsize=(12, 6))
    dendrogram(
        ordered,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        truncate_mode="lastp" if truncate else None,
        p=truncate,
        color_threshold=None,
    )
    plt.title(f"Hierarchical clustering dendrogram ({method})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dendrogram for embeddings")
    parser.add_argument("--emb", type=Path, required=True, help="Embedding file (npy/csv)")
    parser.add_argument("--meta", type=Path, default=None, help="Metadata CSV with sample_id column")
    parser.add_argument("--method", type=str, default="ward", choices=sorted(METHODS), help="Linkage method")
    parser.add_argument("--out", type=Path, required=True, help="Output image path")
    parser.add_argument("--truncate", type=int, default=None, help="Number of leaf nodes to show (None for full)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    embeddings = load_embeddings(args.emb)
    metadata = load_metadata(args.meta, embeddings.shape[0])

    if args.method not in METHODS:
        raise ValueError(f"Unsupported linkage method: {args.method}")

    logging.info("Plotting dendrogram with method=%s", args.method)
    plot_dendrogram(embeddings, metadata, args.method, args.out, args.truncate)
    logging.info("Saved dendrogram to %s", args.out)


if __name__ == "__main__":
    main()
