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
from typing import Dict, List, Optional

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
from sklearn.preprocessing import StandardScaler

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


def run_kmeans(embeddings: np.ndarray, n_clusters: int, init: str, n_init: int) -> Tuple[np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=42)
    labels = model.fit_predict(embeddings)
    distances = cdist(embeddings, model.cluster_centers_, metric="euclidean")
    min_dist = distances[np.arange(len(labels)), labels]
    return labels, min_dist


def run_hdbscan(embeddings: np.ndarray, cfg: Dict) -> hdbscan.HDBSCAN:
    params = {
        "min_cluster_size": cfg.get("min_cluster_size", 8),
        "min_samples": cfg.get("min_samples"),
        "cluster_selection_method": cfg.get("cluster_selection_method", "eom"),
        "allow_single_cluster": cfg.get("allow_single_cluster", False),
        "prediction_data": True,
    }
    model = hdbscan.HDBSCAN(**params)
    model.fit(embeddings)
    return model


def compute_ambiguity(
    embeddings: np.ndarray,
    kmeans_labels: np.ndarray,
    kmeans_distances: np.ndarray,
    hdbscan_labels: np.ndarray,
    silhouette_threshold: float,
    posterior_threshold: float,
    distance_quantile: float,
    gmm_model: GaussianMixture,
) -> np.ndarray:
    unique_labels = np.unique(kmeans_labels)
    if unique_labels.size < 2:
        silhouette_vals = np.zeros(len(kmeans_labels))
    else:
        silhouette_vals = silhouette_samples(embeddings, kmeans_labels)
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
        entries.append(PlotEntry("Elbow curve", path.name, "Within-cluster sum of squares."))

        plt.figure()
        sns.lineplot(x=curves.ks, y=curves.silhouette, marker="o")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.title("Silhouette vs k")
        path = plots_dir / "silhouette_vs_k.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        entries.append(PlotEntry("Silhouette vs k", path.name, "Mean silhouette over all samples."))

        plt.figure()
        sns.lineplot(x=curves.ks, y=curves.gap, marker="o")
        plt.fill_between(curves.ks, np.array(curves.gap) - np.array(curves.gap_std), np.array(curves.gap) + np.array(curves.gap_std), alpha=0.2)
        plt.xlabel("k")
        plt.ylabel("Gap statistic")
        plt.title("Gap statistic")
        path = plots_dir / "gap_stat.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        entries.append(PlotEntry("Gap statistic", path.name, "Gap statistic with 1-sigma ribbon."))

        plt.figure()
        sns.lineplot(x=curves.ks, y=curves.bic, marker="o")
        plt.xlabel("k")
        plt.ylabel("BIC")
        plt.title("Gaussian mixture BIC")
        path = plots_dir / "bic_gmm.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        entries.append(PlotEntry("GMM BIC", path.name, "Bayesian Information Criterion of GMM."))

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
    return PlotEntry("UMAP projection", path.name, "2D UMAP projection colored by consensus labels.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clustering pipeline")
    parser.add_argument("--emb", type=Path, required=True, help="Embeddings .npy/.csv")
    parser.add_argument("--meta", type=Path, default=None, help="Metadata CSV")
    parser.add_argument("--config", type=Path, required=True, help="Clustering YAML config")
    parser.add_argument("--out-dir", type=Path, default=Path("./out"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = load_config(args.config)
    embeddings = load_embeddings(args.emb)
    if args.meta and args.meta.exists():
        metadata = pd.read_csv(args.meta)
    else:
        metadata = pd.DataFrame({"sample_id": np.arange(len(embeddings))})
    if "sample_id" not in metadata.columns:
        metadata["sample_id"] = metadata.index.astype(str)

    scaled, _ = apply_scaling(embeddings, cfg.scale)
    pca_embeddings = apply_pca(scaled, cfg.pca_components, cfg.whiten)
    working_embeddings = pca_embeddings if pca_embeddings is not None else scaled

    auto_k_result = auto_select_k(
        working_embeddings,
        k_min=cfg.k_min,
        k_max=cfg.k_max,
        reference_samples=cfg.gap_reference,
        covariance_type=cfg.covariance_type,
    )
    logging.info("Consensus k determined as %d", auto_k_result.consensus_k)

    kmeans_labels, kmeans_distances = run_kmeans(working_embeddings, auto_k_result.consensus_k, cfg.kmeans_init, cfg.kmeans_n_init)
    hdbscan_model = run_hdbscan(working_embeddings, cfg.hdbscan_cfg)
    hdbscan_labels = hdbscan_model.labels_

    gmm = GaussianMixture(n_components=auto_k_result.consensus_k, covariance_type=cfg.covariance_type, random_state=42)
    gmm.fit(working_embeddings)

    ambiguity_mask = compute_ambiguity(
        working_embeddings,
        kmeans_labels,
        kmeans_distances,
        hdbscan_labels,
        cfg.silhouette_threshold,
        cfg.posterior_threshold,
        cfg.distance_quantile,
        gmm,
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
    umap_entry = plot_umap(output_dir, working_embeddings, consensus_labels)
    if umap_entry:
        plot_entries.append(umap_entry)

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
        "consensus_k": int(auto_k_result.consensus_k),
        "elbow_k": int(auto_k_result.elbow_k),
        "silhouette_k": int(auto_k_result.silhouette_k),
        "gap_k": int(auto_k_result.gap_k),
        "bic_k": int(auto_k_result.bic_k),
    }
    (output_dir / "cluster" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
