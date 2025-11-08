"""Generate Ward-OLO dendrograms and evaluate agreement with K-Means labels."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    linkage,
    optimal_leaf_ordering,
)
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PLOT_PATH = Path("out/plots/dendrogram.png")
DEFAULT_METRICS_PATH = Path("out/cluster/hier_vs_kmeans_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ward hierarchical clustering, plot dendrogram, and compare against K-Means labels.",
    )
    parser.add_argument("--emb", type=Path, required=True, help="Path to PCA-reduced embeddings (npy).")
    parser.add_argument(
        "--kmeans",
        type=Path,
        required=True,
        help="CSV file produced by run_clustering with a 'label' column.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional metadata CSV with 'sample_id' column for leaf labels.",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help=f"Where to save the dendrogram image (default: {DEFAULT_PLOT_PATH}).",
    )
    parser.add_argument(
        "--out-labels",
        type=Path,
        default=None,
        help="Where to save the flattened hierarchical cluster labels (default: out/cluster/hier_labels_k{K}.csv).",
    )
    parser.add_argument(
        "--out-metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Where to save ARI/VI comparison metrics (default: {DEFAULT_METRICS_PATH}).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=None,
        help="Optional cophenetic distance threshold for flattening (distance criterion).",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Override the number of flat clusters (defaults to the number found by K-Means).",
    )
    parser.add_argument(
        "--leaf-font-size",
        type=float,
        default=8.0,
        help="Font size for dendrogram leaf labels (when metadata is provided).",
    )
    return parser.parse_args()


def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        raise ValueError("Embeddings must be provided as a NumPy .npy file containing PCA outputs.")
    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D array of embeddings, got shape {embeddings.shape}.")
    return embeddings


def load_metadata(path: Optional[Path], n_samples: int) -> Optional[pd.Series]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError("Metadata CSV must contain a 'sample_id' column.")
    if len(df["sample_id"]) != n_samples:
        raise ValueError(
            "Number of metadata rows does not match the number of embeddings ("
            f"{len(df['sample_id'])} vs {n_samples})."
        )
    return df["sample_id"].astype(str)


def variation_of_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    contingency = contingency_matrix(labels_a, labels_b).astype(float)
    total = contingency.sum()
    if total <= 0:
        return 0.0

    pij = contingency / total
    pi = pij.sum(axis=1)
    pj = pij.sum(axis=0)

    def entropy(probs: np.ndarray) -> float:
        mask = probs > 0
        if not np.any(mask):
            return 0.0
        return float(-np.sum(probs[mask] * np.log(probs[mask])))

    hx = entropy(pi)
    hy = entropy(pj)

    mutual_information = 0.0
    for i in range(pij.shape[0]):
        for j in range(pij.shape[1]):
            p = pij[i, j]
            if p > 0:
                mutual_information += p * np.log(p / (pi[i] * pj[j]))

    return float(hx + hy - 2.0 * mutual_information)


def run_hierarchical_clustering(
    embeddings: np.ndarray,
    distance_threshold: Optional[float],
    max_clusters: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    linkage_matrix = linkage(embeddings, method="ward", metric="euclidean")
    ordered = optimal_leaf_ordering(linkage_matrix, embeddings, metric="euclidean")

    if distance_threshold is not None:
        labels = fcluster(ordered, t=distance_threshold, criterion="distance")
    elif max_clusters is not None:
        labels = fcluster(ordered, t=max_clusters, criterion="maxclust")
    else:
        raise ValueError("Either distance_threshold or max_clusters must be provided for flattening.")

    return ordered, labels


def plot_and_save_dendrogram(
    linkage_matrix: np.ndarray,
    leaf_labels: Optional[pd.Series],
    output_path: Path,
    leaf_font_size: float,
) -> None:
    plt.figure(figsize=(12, 5))
    dendrogram(
        linkage_matrix,
        labels=None if leaf_labels is None else leaf_labels.tolist(),
        leaf_rotation=90 if leaf_labels is not None else None,
        leaf_font_size=leaf_font_size,
        color_threshold=None,
        no_labels=leaf_labels is None,
    )
    plt.title("Ward hierarchical clustering with optimal leaf ordering")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    embeddings = load_embeddings(args.emb)
    kmeans_df = pd.read_csv(args.kmeans)
    if "label" not in kmeans_df.columns:
        raise ValueError("K-Means assignments CSV must contain a 'label' column.")
    kmeans_labels = kmeans_df["label"].to_numpy()

    if len(kmeans_labels) != embeddings.shape[0]:
        raise ValueError(
            "Number of K-Means labels does not match embeddings count "
            f"({len(kmeans_labels)} vs {embeddings.shape[0]})."
        )

    max_clusters = args.max_clusters or int(len(np.unique(kmeans_labels)))
    if max_clusters <= 0:
        raise ValueError("Number of clusters must be positive.")

    logging.info(
        "Running Ward hierarchical clustering (n_samples=%d, n_features=%d, max_clusters=%d, distance_threshold=%s)",
        embeddings.shape[0],
        embeddings.shape[1],
        max_clusters,
        "auto" if args.distance_threshold is None else args.distance_threshold,
    )

    linkage_matrix, hier_labels = run_hierarchical_clustering(
        embeddings,
        distance_threshold=args.distance_threshold,
        max_clusters=max_clusters,
    )

    metadata = load_metadata(args.meta, embeddings.shape[0])
    plot_and_save_dendrogram(linkage_matrix, metadata, args.out_plot, args.leaf_font_size)

    ari = float(adjusted_rand_score(kmeans_labels, hier_labels))
    vi = variation_of_information(kmeans_labels, hier_labels)

    out_labels = args.out_labels
    if out_labels is None:
        out_labels = Path(f"out/cluster/hier_labels_k{max_clusters}.csv")
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"hier_label": hier_labels}).to_csv(out_labels, index=False)

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(
            {
                "k_star": int(max_clusters),
                "distance_threshold": args.distance_threshold,
                "ARI_hier_vs_kmeans": ari,
                "VI_hier_vs_kmeans": vi,
            },
            f,
            indent=2,
        )

    logging.info("Saved dendrogram to    %s", args.out_plot)
    logging.info("Saved hierarchical labels to %s", out_labels)
    logging.info("Saved comparison metrics to %s", args.out_metrics)


if __name__ == "__main__":
    main()
