"""Run automatic clustering and produce reports for MeshMAE embeddings."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize

import hdbscan

from .auto_k import AutoKResult, MetricCurves, auto_select_k

UMAP = None
if importlib.util.find_spec("umap") is not None:
    from umap import UMAP  # type: ignore


@dataclass
class PipelineConfig:
    scale: bool
    pca_components: Optional[float]
    whiten: bool
    umap_cfg: Dict
    k_min: int
    k_max: int
    gap_reference: int
    covariance_type: str
    majority_threshold: float
    kmeans_init: str
    kmeans_n_init: int
    hdbscan_cfg: Dict
    silhouette_threshold: float
    posterior_threshold: float
    distance_quantile: float
    html_template: Path
    report_title: str


@dataclass
class PlotEntry:
    title: str
    filename: str
    caption: str
    kind: str = "image"


def load_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preprocessing = cfg.get("preprocessing", {})
    consensus = cfg.get("consensus", {})
    clustering = cfg.get("clustering", {})
    unknown = cfg.get("unknown_detection", {})
    report = cfg.get("report", {})

    return PipelineConfig(
        scale=bool(preprocessing.get("scale", True)),
        pca_components=preprocessing.get("pca_components", None),
        whiten=bool(preprocessing.get("whiten", False)),
        umap_cfg=preprocessing.get("umap", {}),
        k_min=int(consensus.get("k_min", 2)),
        k_max=int(consensus.get("k_max", 15)),
        gap_reference=int(consensus.get("gap_reference", 5)),
        covariance_type=consensus.get("gmm_covariance_type", "full"),
        majority_threshold=float(consensus.get("majority_threshold", 0.5)),
        kmeans_init=clustering.get("kmeans_init", "k-means++"),
        kmeans_n_init=int(clustering.get("kmeans_n_init", 10)),
        hdbscan_cfg=clustering.get("hdbscan", {}),
        silhouette_threshold=float(unknown.get("silhouette_threshold", 0.05)),
        posterior_threshold=float(unknown.get("posterior_threshold", 0.5)),
        distance_quantile=float(unknown.get("distance_quantile", 0.95)),
        html_template=Path(report.get("html_template", "./src/cluster/templates/report.html")),
        report_title=report.get("title", "Mesh Clustering Report"),
    )


def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.values
    return np.load(path)


def apply_scaling(embeddings: np.ndarray, enabled: bool) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    if not enabled:
        return embeddings, None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    return scaled, scaler


def apply_pca(embeddings: np.ndarray, components: Optional[float], whiten: bool) -> Optional[np.ndarray]:
    if components in (None, 0):
        return None
    if isinstance(components, float) and 0 < components < 1:
        pca = PCA(n_components=components, svd_solver="full", whiten=whiten, random_state=42)
    else:
        pca = PCA(n_components=int(components), whiten=whiten, random_state=42)
    transformed = pca.fit_transform(embeddings)
    return transformed


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    init: str,
    n_init: int,
    distance_metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=42)
    labels = model.fit_predict(embeddings)
    distances = cdist(embeddings, model.cluster_centers_, metric=distance_metric)
    min_dist = distances[np.arange(len(labels)), labels]
    return labels, min_dist


def log_feature_statistics(embeddings: np.ndarray, label: str) -> None:
    means = embeddings.mean(axis=0)
    variances = embeddings.var(axis=0)
    unique_counts = np.apply_along_axis(lambda col: np.unique(col).size, 0, embeddings)
    stats_df = pd.DataFrame(
        {
            "mean": means,
            "variance": variances,
            "unique": unique_counts,
        }
    )
    logging.info("%s feature stats (mean/variance/unique per dimension):\n%s", label, stats_df.to_string(index=True))


def ensure_non_degenerate_embeddings(embeddings: np.ndarray, label: str) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"{label} embeddings must be a 2D array, got shape {embeddings.shape}.")
    if embeddings.shape[1] == 0:
        raise ValueError(f"{label} embeddings have zero feature dimensions.")
    if embeddings.shape[0] == 0:
        raise ValueError(f"{label} embeddings have zero samples.")
    if np.allclose(embeddings, 0):
        raise ValueError(f"{label} embeddings are all zeros. Check embedding extraction outputs.")
    if np.allclose(embeddings, embeddings[0]):
        raise ValueError(f"{label} embeddings are identical across samples. Check embedding extraction outputs.")


def run_hdbscan(embeddings: np.ndarray, cfg: Dict) -> hdbscan.HDBSCAN:
    params = {
        "min_cluster_size": cfg.get("min_cluster_size", 8),
        "min_samples": cfg.get("min_samples"),
        "cluster_selection_method": cfg.get("cluster_selection_method", "eom"),
        "allow_single_cluster": cfg.get("allow_single_cluster", False),
        "metric": cfg.get("metric", "euclidean"),
        "prediction_data": True,
    }
    model = hdbscan.HDBSCAN(**params)
    model.fit(embeddings)
    return model


def is_scale_inappropriate(embeddings: np.ndarray, ratio_threshold: float = 100.0) -> bool:
    stds = embeddings.std(axis=0)
    nonzero = stds[stds > 0]
    if nonzero.size == 0:
        return False
    return (nonzero.max() / nonzero.min()) > ratio_threshold


def relax_hdbscan_params(cfg: Dict) -> Dict:
    relaxed = dict(cfg)
    min_cluster_size = relaxed.get("min_cluster_size", 8)
    if min_cluster_size is None:
        min_cluster_size = 8
    relaxed["min_cluster_size"] = max(2, int(min_cluster_size * 0.5))
    min_samples = relaxed.get("min_samples")
    if min_samples is None:
        relaxed["min_samples"] = max(1, int(relaxed["min_cluster_size"] / 2))
    else:
        relaxed["min_samples"] = max(1, int(min_samples * 0.5))
    return relaxed


def compute_ambiguity(
    embeddings: np.ndarray,
    kmeans_labels: np.ndarray,
    kmeans_distances: np.ndarray,
    hdbscan_labels: np.ndarray,
    silhouette_threshold: float,
    posterior_threshold: float,
    distance_quantile: float,
    gmm_model: GaussianMixture,
    distance_metric: str,
) -> np.ndarray:
    unique_labels = np.unique(kmeans_labels)
    if unique_labels.size < 2:
        silhouette_vals = np.zeros(len(kmeans_labels))
    else:
        silhouette_vals = silhouette_samples(embeddings, kmeans_labels, metric=distance_metric)
    posterior = gmm_model.predict_proba(embeddings)
    max_posterior = posterior.max(axis=1)
    distance_cutoff = np.quantile(kmeans_distances, distance_quantile)
    mask = (
        (hdbscan_labels == -1)
        | (silhouette_vals < silhouette_threshold)
        | (kmeans_distances > distance_cutoff)
        | (max_posterior < posterior_threshold)
    )
    return mask


def render_report(
    output_dir: Path,
    template_path: Path,
    title: str,
    assignments: pd.DataFrame,
    auto_result: AutoKResult,
    plots: List[PlotEntry],
) -> None:
    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(template_path.name)

    curves = auto_result.curves
    metrics = [
        {"name": "Consensus K", "value": int(auto_result.consensus_k), "description": "Majority vote across elbow/silhouette/gap/BIC."},
        {"name": "Elbow suggestion", "value": int(auto_result.elbow_k), "description": "Elbow via Kneedle."},
        {"name": "Best silhouette", "value": int(auto_result.silhouette_k), "description": "k with highest silhouette."},
        {"name": "Gap statistic", "value": int(auto_result.gap_k), "description": "k selected by gap statistic."},
        {"name": "GMM BIC", "value": int(auto_result.bic_k), "description": "k with lowest BIC."},
    ]

    rendered = template.render(
        title=title,
        generated_at=datetime.utcnow().isoformat(),
        assignments=assignments.to_dict("records"),
        consensus_metrics=metrics,
        plots=[dataclasses.asdict(plot) for plot in plots],
    )

    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "index.html").write_text(rendered, encoding="utf-8")


def save_assignments(output_dir: Path, df: pd.DataFrame) -> None:
    cluster_dir = output_dir / "cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cluster_dir / "consensus.csv", index=False)


def plot_curves(output_dir: Path, curves: MetricCurves) -> List[PlotEntry]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    entries: List[PlotEntry] = []

    if curves.ks:
        plt.figure()
        sns.lineplot(x=curves.ks, y=curves.inertia, marker="o")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.title("Elbow curve")
        path = plots_dir / "elbow.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        entries.append(PlotEntry("Elbow curve", path.name, "Within-cluster sum of squares.", kind="image"))

        plt.figure()
        sns.lineplot(x=curves.ks, y=curves.silhouette, marker="o")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.title("Silhouette vs k")
        path = plots_dir / "silhouette_vs_k.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        entries.append(PlotEntry("Silhouette vs k", path.name, "Mean silhouette over all samples.", kind="image"))

        plt.figure()
        sns.lineplot(x=curves.ks, y=curves.gap, marker="o")
        plt.fill_between(curves.ks, np.array(curves.gap) - np.array(curves.gap_std), np.array(curves.gap) + np.array(curves.gap_std), alpha=0.2)
        plt.xlabel("k")
        plt.ylabel("Gap statistic")
        plt.title("Gap statistic")
        path = plots_dir / "gap_stat.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        entries.append(PlotEntry("Gap statistic", path.name, "Gap statistic with 1-sigma ribbon.", kind="image"))

        plt.figure()
        sns.lineplot(x=curves.ks, y=curves.bic, marker="o")
        plt.xlabel("k")
        plt.ylabel("BIC")
        plt.title("Gaussian mixture BIC")
        path = plots_dir / "bic_gmm.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        entries.append(PlotEntry("GMM BIC", path.name, "Bayesian Information Criterion of GMM.", kind="image"))

    return entries


def plot_umap(output_dir: Path, embeddings: np.ndarray, labels: np.ndarray) -> Optional[PlotEntry]:
    if UMAP is None:
        logging.warning("UMAP not installed; skipping UMAP plot.")
        return None
    reducer = UMAP(n_neighbors=25, min_dist=0.1, metric="cosine", random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    df = pd.DataFrame({"x": umap_embeddings[:, 0], "y": umap_embeddings[:, 1], "label": labels})
    plt.figure()
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab20", s=40)
    plt.title("UMAP projection (consensus)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "umap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return PlotEntry("UMAP projection", path.name, "2D UMAP projection colored by consensus labels.", kind="image")


def plot_umap_3d_html(output_dir: Path, embeddings: np.ndarray, labels: np.ndarray) -> Optional[PlotEntry]:
    if UMAP is None:
        logging.warning("UMAP not installed; skipping 3D UMAP plot.")
        return None

    reducer = UMAP(n_neighbors=25, min_dist=0.1, metric="cosine", random_state=42, n_components=3)
    umap_embeddings = reducer.fit_transform(embeddings)
    df = pd.DataFrame(
        {
            "x": umap_embeddings[:, 0],
            "y": umap_embeddings[:, 1],
            "z": umap_embeddings[:, 2],
            "label": labels.astype(int),
        }
    )
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "umap_3d.html"

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]

    traces = []
    for label, group in df.groupby("label"):
        color = palette[int(label) % len(palette)]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"Cluster {label}",
                "x": group["x"].tolist(),
                "y": group["y"].tolist(),
                "z": group["z"].tolist(),
                "marker": {"size": 4, "color": color, "opacity": 0.85},
                "text": [f"cluster={label}"] * len(group),
                "hovertemplate": "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>%{text}<extra></extra>",
            }
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>3D UMAP projection</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
      font-family: "Inter", "Helvetica", sans-serif;
    }}
    #plot {{
      width: 100vw;
      height: 100vh;
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const data = {json.dumps(traces)};
    const layout = {{
      title: "3D UMAP projection (consensus)",
      paper_bgcolor: "#0f172a",
      plot_bgcolor: "#0f172a",
      font: {{color: "#e2e8f0"}},
      scene: {{
        xaxis: {{title: "UMAP-1", gridcolor: "#1e293b"}},
        yaxis: {{title: "UMAP-2", gridcolor: "#1e293b"}},
        zaxis: {{title: "UMAP-3", gridcolor: "#1e293b"}},
      }},
      margin: {{l: 0, r: 0, b: 0, t: 40}},
    }};
    Plotly.newPlot("plot", data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return PlotEntry(
        "UMAP projection (3D)",
        path.name,
        "Interactive 3D UMAP projection colored by consensus labels.",
        kind="html",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clustering pipeline")
    parser.add_argument("--emb", type=Path, required=True, help="Embeddings .npy/.csv")
    parser.add_argument("--meta", type=Path, default=None, help="Metadata CSV")
    parser.add_argument("--config", type=Path, required=True, help="Clustering YAML config")
    parser.add_argument("--out-dir", type=Path, default=Path("./out"), help="Output directory")
    parser.add_argument(
        "--kmeans-k",
        default="auto",
        help="Fix K-Means clusters (integer) or use auto selection (default: auto).",
    )
    parser.add_argument(
        "--distance-metric",
        choices=("euclidean", "cosine"),
        default="euclidean",
        help="Distance metric for K-Means/HDBSCAN ambiguity checks (default: euclidean).",
    )
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        help="Apply L2 normalization before clustering (recommended for cosine distance).",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Metadata column containing class labels for kNN accuracy checks.",
    )
    return parser.parse_args()


def parse_kmeans_k(raw: str) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.lower() == "auto":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid --kmeans-k value '{raw}'; use an integer or 'auto'.") from exc
    if value <= 0:
        raise ValueError("--kmeans-k must be a positive integer.")
    return value


def apply_l2_normalization(embeddings: np.ndarray) -> np.ndarray:
    return normalize(embeddings, norm="l2")


def compute_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: List[int],
    metric: str,
) -> Dict[int, Optional[float]]:
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Embeddings and labels must have the same number of samples for kNN evaluation.")
    n_samples = embeddings.shape[0]
    label_codes, _ = pd.factorize(labels)
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)
    results: Dict[int, Optional[float]] = {}
    for k in ks:
        if k <= 0 or k >= n_samples:
            results[k] = None
            continue
        neighbor_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        preds = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            votes = label_codes[neighbor_idx[i]]
            preds[i] = np.bincount(votes, minlength=label_codes.max() + 1).argmax()
        results[k] = float(np.mean(preds == label_codes))
    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = load_config(args.config)
    embeddings = load_embeddings(args.emb)
    logging.info(
        "Preprocessing config: scale=%s, pca_components=%s, whiten=%s",
        cfg.scale,
        cfg.pca_components,
        cfg.whiten,
    )
    if args.meta and args.meta.exists():
        metadata = pd.read_csv(args.meta)
    else:
        metadata = pd.DataFrame({"sample_id": np.arange(len(embeddings))})
    if "sample_id" not in metadata.columns:
        metadata["sample_id"] = metadata.index.astype(str)

    scaled, _ = apply_scaling(embeddings, cfg.scale)
    pca_embeddings = apply_pca(scaled, cfg.pca_components, cfg.whiten)
    working_embeddings = pca_embeddings if pca_embeddings is not None else scaled
    distance_metric = args.distance_metric
    use_l2_norm = args.l2_normalize or distance_metric == "cosine"
    clustering_embeddings = working_embeddings
    if use_l2_norm:
        clustering_embeddings = apply_l2_normalization(working_embeddings)
        logging.info("Applied L2 normalization to embeddings for clustering.")
    ensure_non_degenerate_embeddings(clustering_embeddings, "KMeans input")
    log_feature_statistics(clustering_embeddings, "Clustering input")

    auto_k_result = auto_select_k(
        clustering_embeddings,
        k_min=cfg.k_min,
        k_max=cfg.k_max,
        reference_samples=cfg.gap_reference,
        covariance_type=cfg.covariance_type,
    )
    fixed_k = parse_kmeans_k(args.kmeans_k)
    if fixed_k is not None:
        logging.info("Overriding consensus k with fixed K-Means k=%d", fixed_k)
        auto_k_result = dataclasses.replace(auto_k_result, consensus_k=fixed_k)
        k_source = "fixed"
    else:
        k_source = "auto"
        logging.info("Consensus k determined as %d", auto_k_result.consensus_k)

    kmeans_labels, kmeans_distances = run_kmeans(
        clustering_embeddings,
        auto_k_result.consensus_k,
        cfg.kmeans_init,
        cfg.kmeans_n_init,
        distance_metric,
    )
    hdbscan_embeddings = clustering_embeddings
    additional_hdbscan_scaling = False
    if not cfg.scale and is_scale_inappropriate(clustering_embeddings):
        logging.warning(
            "HDBSCAN features show wide scale variance; applying StandardScaler before HDBSCAN."
        )
        hdbscan_embeddings, _ = apply_scaling(clustering_embeddings, True)
        additional_hdbscan_scaling = True
    hdbscan_cfg = cfg.hdbscan_cfg
    hdbscan_cfg = dict(hdbscan_cfg)
    if args.distance_metric:
        hdbscan_cfg["metric"] = args.distance_metric
    logging.info(
        "HDBSCAN config: min_cluster_size=%s, min_samples=%s, metric=%s, preprocessing_scale=%s, extra_hdbscan_scale=%s",
        hdbscan_cfg.get("min_cluster_size", 8),
        hdbscan_cfg.get("min_samples"),
        hdbscan_cfg.get("metric", "euclidean"),
        cfg.scale,
        additional_hdbscan_scaling,
    )
    hdbscan_model = run_hdbscan(hdbscan_embeddings, hdbscan_cfg)
    hdbscan_labels = hdbscan_model.labels_
    noise_rate = float(np.mean(hdbscan_labels == -1))
    logging.info("HDBSCAN noise (-1) rate: %.2f", noise_rate)
    if noise_rate > 0.5:
        relaxed_cfg = relax_hdbscan_params(hdbscan_cfg)
        logging.warning("High HDBSCAN noise rate detected; relaxing params to %s", relaxed_cfg)
        hdbscan_model = run_hdbscan(hdbscan_embeddings, relaxed_cfg)
        hdbscan_labels = hdbscan_model.labels_
        noise_rate = float(np.mean(hdbscan_labels == -1))
        logging.info("HDBSCAN noise (-1) rate after relaxation: %.2f", noise_rate)

    gmm = GaussianMixture(n_components=auto_k_result.consensus_k, covariance_type=cfg.covariance_type, random_state=42)
    gmm.fit(clustering_embeddings)

    ambiguity_mask = compute_ambiguity(
        clustering_embeddings,
        kmeans_labels,
        kmeans_distances,
        hdbscan_labels,
        cfg.silhouette_threshold,
        cfg.posterior_threshold,
        cfg.distance_quantile,
        gmm,
        distance_metric,
    )

    consensus_labels = kmeans_labels.copy()

    df = metadata.copy()
    df["kmeans"] = kmeans_labels
    df["hdbscan"] = hdbscan_labels
    df["consensus"] = consensus_labels
    df["ambiguous"] = ambiguity_mask

    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_assignments(output_dir, df)

    plot_entries = plot_curves(output_dir, auto_k_result.curves)
    umap_entry = plot_umap(output_dir, clustering_embeddings, consensus_labels)
    if umap_entry:
        plot_entries.append(umap_entry)
    umap_3d_entry = plot_umap_3d_html(output_dir, clustering_embeddings, consensus_labels)
    if umap_3d_entry:
        plot_entries.append(umap_3d_entry)

    cluster_dir = output_dir / "cluster"
    df[["sample_id", "kmeans", "ambiguous"]].to_csv(cluster_dir / "kmeans_assignments.csv", index=False)
    df[["sample_id", "hdbscan", "ambiguous"]].to_csv(cluster_dir / "hdbscan_assignments.csv", index=False)

    render_report(
        output_dir,
        cfg.html_template,
        cfg.report_title,
        df,
        auto_k_result,
        plot_entries,
    )

    summary = {
        "kmeans_k": int(auto_k_result.consensus_k),
        "kmeans_k_source": k_source,
        "consensus_k": int(auto_k_result.consensus_k),
        "elbow_k": int(auto_k_result.elbow_k),
        "silhouette_k": int(auto_k_result.silhouette_k),
        "gap_k": int(auto_k_result.gap_k),
        "bic_k": int(auto_k_result.bic_k),
    }
    if args.label_column and args.label_column in metadata.columns:
        knn_metrics = compute_knn_accuracy(
            clustering_embeddings,
            metadata[args.label_column].to_numpy(),
            ks=[1, 5, 10],
            metric=distance_metric,
        )
        summary["knn_accuracy"] = {str(k): v for k, v in knn_metrics.items()}
        (output_dir / "cluster" / "knn_metrics.json").write_text(
            json.dumps(summary["knn_accuracy"], indent=2),
            encoding="utf-8",
        )
        logging.info("kNN accuracy (k=1,5,10): %s", summary["knn_accuracy"])
    elif args.label_column:
        logging.warning("Label column '%s' not found in metadata; skipping kNN accuracy.", args.label_column)
    (output_dir / "cluster" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
