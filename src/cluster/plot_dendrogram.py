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
DEFAULT_LABEL_MODE = "auto"
LABEL_MODES = ("auto", "all", "thin", "truncate", "none")
TRUNCATE_MODES = ("lastp", "level")
DEFAULT_AUTO_LEAF_THRESHOLD = 200
DEFAULT_TRUNCATE_P = 100
DEFAULT_THIN_STEP = 10
DEFAULT_FIGSIZE = (12.0, 5.0)
DEFAULT_MIN_FIG_WIDTH = 12.0
DEFAULT_MAX_FIG_WIDTH = 200.0
DEFAULT_FIG_WIDTH_PER_LEAF = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ward hierarchical clustering, plot dendrogram, and compare against K-Means labels.",
    )
    parser.add_argument("--emb", type=Path, required=True, help="Path to PCA-reduced embeddings (npy).")
    parser.add_argument(
        "--kmeans",
        type=Path,
        required=True,
        help="CSV file produced by run_clustering with K-Means assignments.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Optional column name to use for K-Means labels (defaults to auto-detect).",
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
    parser.add_argument(
        "--label-mode",
        choices=LABEL_MODES,
        default=DEFAULT_LABEL_MODE,
        help=(
            "Leaf label strategy: all (show every label), thin (show every Nth), "
            "truncate (use SciPy truncation), none (no labels), or auto "
            f"(default: {DEFAULT_LABEL_MODE}). Note: higher DPI does not fix dense labels."
        ),
    )
    parser.add_argument(
        "--thin-step",
        type=int,
        default=DEFAULT_THIN_STEP,
        help="Show every Nth label when --label-mode thin is selected.",
    )
    parser.add_argument(
        "--truncate-mode",
        choices=TRUNCATE_MODES,
        default="lastp",
        help="SciPy dendrogram truncate_mode (used when --label-mode truncate or auto).",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=DEFAULT_TRUNCATE_P,
        help="Number of leaves/clusters to show when --label-mode truncate is selected.",
    )
    parser.add_argument(
        "--auto-leaf-threshold",
        type=int,
        default=DEFAULT_AUTO_LEAF_THRESHOLD,
        help="Leaf count threshold to trigger automatic truncation/no-label handling.",
    )
    parser.add_argument(
        "--auto-label-mode",
        choices=("truncate", "none"),
        default="truncate",
        help="Label strategy to use when --label-mode auto and leaf count exceeds the threshold.",
    )
    parser.add_argument(
        "--label-map-out",
        type=Path,
        default=None,
        help="Optional CSV/TSV path to save leaf label visibility mapping.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output format for the dendrogram image.",
    )
    parser.add_argument(
        "--auto-vector-format",
        choices=("pdf", "svg"),
        default=None,
        help="Automatically switch to a vector format when leaf count is large.",
    )
    parser.add_argument(
        "--min-fig-width",
        type=float,
        default=DEFAULT_MIN_FIG_WIDTH,
        help="Minimum figure width for auto-scaling when label_mode=all.",
    )
    parser.add_argument(
        "--max-fig-width",
        type=float,
        default=DEFAULT_MAX_FIG_WIDTH,
        help="Maximum figure width for auto-scaling when label_mode=all.",
    )
    parser.add_argument(
        "--fig-width-per-leaf",
        type=float,
        default=DEFAULT_FIG_WIDTH_PER_LEAF,
        help="Width per leaf when auto-scaling figure size for label_mode=all.",
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


def resolve_kmeans_labels(df: pd.DataFrame, label_column: Optional[str]) -> np.ndarray:
    if label_column:
        if label_column not in df.columns:
            raise ValueError(f"Requested label column '{label_column}' not found in K-Means CSV.")
        return df[label_column].to_numpy()

    for candidate in ("label", "kmeans", "consensus"):
        if candidate in df.columns:
            return df[candidate].to_numpy()

    raise ValueError(
        "K-Means assignments CSV must contain a 'label', 'kmeans', or 'consensus' column, "
        "or specify --label-column."
    )


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
    leaf_labels: Optional[list[str]],
    output_path: Path,
    leaf_font_size: float,
    label_mode: str,
    thin_step: int,
    truncate_mode: str,
    truncate_p: int,
    figsize: tuple[float, float],
    label_map_out: Optional[Path],
    labels_for_map: list[str],
) -> None:
    shown_indices: set[int] = set()
    show_labels = leaf_labels is not None and label_mode != "none"

    leaf_label_func = None
    if show_labels:
        if label_mode == "all":
            leaf_label_func = _make_leaf_label_func_all(leaf_labels, shown_indices)
        elif label_mode == "thin":
            leaf_label_func = _make_leaf_label_func_thin(leaf_labels, thin_step, shown_indices)
        elif label_mode == "truncate":
            leaf_label_func = _make_leaf_label_func_truncate(leaf_labels, shown_indices)
        else:
            raise ValueError(f"Unsupported label_mode '{label_mode}'.")

    plt.figure(figsize=figsize)
    dendrogram(
        linkage_matrix,
        truncate_mode=truncate_mode if label_mode == "truncate" else None,
        p=truncate_p if label_mode == "truncate" else None,
        leaf_rotation=90 if show_labels else None,
        leaf_font_size=leaf_font_size if show_labels else None,
        leaf_label_func=leaf_label_func,
        color_threshold=None,
        no_labels=not show_labels,
    )
    plt.title("Ward hierarchical clustering with optimal leaf ordering")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    if label_map_out is not None:
        label_map_out.parent.mkdir(parents=True, exist_ok=True)
        _write_label_map(label_map_out, labels_for_map, shown_indices)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    embeddings = load_embeddings(args.emb)
    kmeans_df = pd.read_csv(args.kmeans)
    kmeans_labels = resolve_kmeans_labels(kmeans_df, args.label_column)

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
    leaf_labels = metadata.astype(str).tolist() if metadata is not None else None
    labels_for_map = (
        metadata.astype(str).tolist()
        if metadata is not None
        else [str(index) for index in range(embeddings.shape[0])]
    )
    n_leaves = linkage_matrix.shape[0] + 1
    label_mode = resolve_label_mode(
        args.label_mode,
        n_leaves,
        args.auto_leaf_threshold,
        args.auto_label_mode,
        has_labels=leaf_labels is not None,
    )
    if label_mode == "all" and n_leaves > args.auto_leaf_threshold:
        logging.warning(
            "Leaf count (%d) exceeds threshold (%d) while label_mode=all; output may be enormous.",
            n_leaves,
            args.auto_leaf_threshold,
        )
    if label_mode == "thin" and args.thin_step <= 0:
        raise ValueError("--thin-step must be a positive integer.")
    if label_mode == "truncate" and args.p <= 0:
        raise ValueError("--p must be a positive integer for truncate mode.")

    output_path = resolve_output_path(args.out_plot, args.format)
    if (
        n_leaves > args.auto_leaf_threshold
        and args.format == "png"
        and args.auto_vector_format is not None
    ):
        output_path = output_path.with_suffix(f".{args.auto_vector_format}")
        logging.info(
            "Switching dendrogram output to vector format (%s) for large leaf counts.",
            args.auto_vector_format,
        )
    elif n_leaves > args.auto_leaf_threshold and args.format == "png":
        logging.warning(
            "Large leaf counts produce dense plots in PNG; consider --format pdf/svg or --auto-vector-format.",
        )

    figsize = resolve_figsize(
        label_mode=label_mode,
        n_leaves=n_leaves,
        base_size=DEFAULT_FIGSIZE,
        min_width=args.min_fig_width,
        max_width=args.max_fig_width,
        width_per_leaf=args.fig_width_per_leaf,
    )
    plot_and_save_dendrogram(
        linkage_matrix,
        leaf_labels,
        output_path,
        args.leaf_font_size,
        label_mode,
        args.thin_step,
        args.truncate_mode,
        args.p,
        figsize,
        args.label_map_out,
        labels_for_map,
    )

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

    logging.info("Saved dendrogram to    %s", output_path)
    logging.info("Saved hierarchical labels to %s", out_labels)
    logging.info("Saved comparison metrics to %s", args.out_metrics)


def resolve_label_mode(
    label_mode: str,
    n_leaves: int,
    threshold: int,
    auto_label_mode: str,
    has_labels: bool,
) -> str:
    if label_mode != "auto":
        return label_mode
    if n_leaves > threshold:
        logging.info(
            "Leaf count %d exceeds %d; switching label_mode from auto to %s.",
            n_leaves,
            threshold,
            auto_label_mode,
        )
        return auto_label_mode
    if has_labels:
        return "all"
    return "none"


def resolve_output_path(output_path: Path, fmt: str) -> Path:
    fmt = fmt.lower()
    desired_suffix = f".{fmt}"
    if output_path.suffix.lower() != desired_suffix:
        return output_path.with_suffix(desired_suffix)
    return output_path


def resolve_figsize(
    label_mode: str,
    n_leaves: int,
    base_size: tuple[float, float],
    min_width: float,
    max_width: float,
    width_per_leaf: float,
) -> tuple[float, float]:
    if label_mode != "all":
        return base_size
    width = min(max(min_width, n_leaves * width_per_leaf), max_width)
    if width > base_size[0]:
        logging.warning(
            "Auto-scaling figure width to %.1f for %d leaves (min=%.1f, max=%.1f).",
            width,
            n_leaves,
            min_width,
            max_width,
        )
    return (width, base_size[1])


def _make_leaf_label_func_all(labels: list[str], shown_indices: set[int]):
    n_samples = len(labels)

    def _label(leaf_id: int) -> str:
        if leaf_id < n_samples:
            shown_indices.add(leaf_id)
            return labels[leaf_id]
        return ""

    return _label


def _make_leaf_label_func_thin(labels: list[str], thin_step: int, shown_indices: set[int]):
    n_samples = len(labels)

    def _label(leaf_id: int) -> str:
        if leaf_id < n_samples and leaf_id % thin_step == 0:
            shown_indices.add(leaf_id)
            return labels[leaf_id]
        return ""

    return _label


def _make_leaf_label_func_truncate(labels: list[str], shown_indices: set[int]):
    n_samples = len(labels)

    def _label(leaf_id: int) -> str:
        if leaf_id < n_samples:
            shown_indices.add(leaf_id)
            return labels[leaf_id]
        return ""

    return _label


def _write_label_map(output_path: Path, labels: list[str], shown_indices: set[int]) -> None:
    rows = []
    for index, label in enumerate(labels):
        shown = int(index in shown_indices)
        rows.append(
            {
                "leaf_index": index,
                "sample_id": label,
                "shown_label": label if shown else "",
                "shown": shown,
            }
        )
    df = pd.DataFrame(rows)
    sep = "\t" if output_path.suffix.lower() == ".tsv" else ","
    df.to_csv(output_path, index=False, sep=sep)


if __name__ == "__main__":
    main()
