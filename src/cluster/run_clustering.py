"""Run automatic clustering and produce reports for MeshMAE embeddings.

Example:
    python -m src.cluster.run_clustering \
      --emb embeddings/raw_embeddings.npy \
      --meta embeddings/meta.csv \
      --label-col category \
      --out-dir out \
      --scale \
      --l2-normalize \
      --distance-metric cosine \
      --kmeans-k 40 \
      --run-hdbscan \
      --hdbscan-sweep --hdbscan-sweep-min 5 --hdbscan-sweep-max 80 --hdbscan-sweep-step 5
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples
from sklearn.metrics.cluster import contingency_matrix
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


PRESET_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "kmeans40_labelmatch": {
        "description": "KMeans(k=40) label matching evaluation.",
        "settings": {
            "pca_dim": 64,
            "whiten": False,
            "scale": False,
            "l2_normalize": False,
            "distance_metric": "cosine",
            "kmeans_k": "40",
            "run_hdbscan": False,
            "hdbscan_sweep": False,
        },
    },
    "hdbscan_core": {
        "description": "HDBSCAN core extraction with sweep scoring.",
        "settings": {
            "pca_dim": 64,
            "whiten": False,
            "scale": False,
            "l2_normalize": True,
            "distance_metric": "cosine",
            "run_hdbscan": True,
            "hdbscan_sweep": True,
            "hdbscan_min_samples_mode": "fixed",
            "hdbscan_min_samples_sweep": "1,2,5,10",
        },
    },
    "retrieval_knn": {
        "description": "kNN retrieval evaluation (acc@1/5/10).",
        "settings": {
            "pca_dim": 64,
            "whiten": False,
            "scale": False,
            "l2_normalize": True,
            "distance_metric": "cosine",
            "run_hdbscan": False,
            "hdbscan_sweep": False,
        },
    },
}


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


def log_feature_summary(embeddings: np.ndarray, label: str) -> None:
    means = embeddings.mean(axis=0)
    stds = embeddings.std(axis=0)
    summary = {
        "dim": int(embeddings.shape[1]),
        "mean_min": float(means.min()),
        "mean_max": float(means.max()),
        "std_min": float(stds.min()),
        "std_max": float(stds.max()),
    }
    logging.info("%s feature summary: %s", label, summary)


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
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{label} embeddings contain NaN or Inf values.")


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


def build_preprocess_variant_id(
    scale: bool, pca_dim: Optional[int], whiten: bool, l2_normalize: bool
) -> str:
    pca_label = "none" if pca_dim is None else str(pca_dim)
    return f"scale-{int(scale)}_pca-{pca_label}_whiten-{int(whiten)}_l2-{int(l2_normalize)}"


def parse_int_list(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    values: List[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid integer value '{part}' in list '{raw}'.") from exc
        values.append(value)
    return values


def resolve_value(cli_value: Optional[object], preset_value: Optional[object], yaml_value: Optional[object], default: object) -> object:
    if cli_value is not None:
        return cli_value
    if preset_value is not None:
        return preset_value
    if yaml_value is not None:
        return yaml_value
    return default


def resolve_pca_components_with_precedence(
    args: argparse.Namespace,
    cfg: PipelineConfig,
    preset_settings: Dict[str, object],
) -> Optional[float]:
    if args.pca_var is not None:
        return args.pca_var
    if args.pca_dim is not None:
        return float(args.pca_dim)
    if args.pca_components is not None:
        return args.pca_components
    preset_pca_dim = preset_settings.get("pca_dim")
    preset_pca_components = preset_settings.get("pca_components")
    if preset_pca_dim is not None:
        return float(preset_pca_dim)
    if preset_pca_components is not None:
        return float(preset_pca_components)
    return cfg.pca_components


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


def select_label_column(metadata: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in metadata.columns:
            raise ValueError(
                f"Label column '{requested}' not found in metadata. Available columns: {list(metadata.columns)}"
            )
        return requested
    if "label" in metadata.columns:
        return "label"
    if "category" in metadata.columns:
        return "category"
    raise ValueError(
        "No default label column found in metadata. Provide --label-col. "
        f"Available columns: {list(metadata.columns)}"
    )


def select_path_column(metadata: pd.DataFrame) -> Optional[str]:
    for candidate in ("path", "file_path", "filepath"):
        if candidate in metadata.columns:
            return candidate
    return None


def encode_labels(labels: pd.Series) -> Tuple[np.ndarray, Dict[int, str]]:
    codes, uniques = pd.factorize(labels.astype(str))
    return codes.astype(int), {idx: str(label) for idx, label in enumerate(uniques)}


def compute_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = contingency_matrix(y_true, y_pred)
    return float(np.sum(np.max(matrix, axis=0)) / np.sum(matrix))


def compute_cluster_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None
) -> Dict[str, Optional[float]]:
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"ari": None, "nmi": None, "purity": None, "n_samples": 0}
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "purity": compute_purity(y_true, y_pred),
        "n_samples": int(y_true.size),
    }


def compute_cluster_sizes(labels: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts)}


def compute_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: Iterable[int],
    metric: str,
) -> Tuple[Dict[str, Dict[str, Optional[float]]], Dict[str, float], Dict[str, int]]:
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Embeddings and labels must have the same number of samples for kNN evaluation.")
    n_samples = embeddings.shape[0]
    label_codes, label_mapping = encode_labels(pd.Series(labels))
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)
    results: Dict[str, Dict[str, Optional[float]]] = {}
    per_class_k1: Dict[str, float] = {}
    per_class_counts: Dict[str, int] = {}
    for k in ks:
        if k <= 0 or k >= n_samples:
            results[str(k)] = {"overall": None, "macro": None}
            continue
        neighbor_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        preds = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            votes = label_codes[neighbor_idx[i]]
            preds[i] = np.bincount(votes, minlength=len(label_mapping)).argmax()
        overall = float(np.mean(preds == label_codes))
        per_class = []
        for class_id in np.unique(label_codes):
            class_mask = label_codes == class_id
            per_class.append(float(np.mean(preds[class_mask] == class_id)))
        macro = float(np.mean(per_class)) if per_class else None
        results[str(k)] = {"overall": overall, "macro": macro}
        if k == 1:
            for class_id, class_name in label_mapping.items():
                class_mask = label_codes == class_id
                if not np.any(class_mask):
                    continue
                per_class_k1[class_name] = float(np.mean(preds[class_mask] == class_id))
                per_class_counts[class_name] = int(np.sum(class_mask))
    return results, per_class_k1, per_class_counts


def compute_hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = contingency_matrix(y_true, y_pred)
    if matrix.size == 0:
        return 0.0
    max_value = matrix.max()
    cost = max_value - matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = matrix[row_ind, col_ind].sum()
    return float(matched / matrix.sum())


def compute_hungarian_mapping(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, str]:
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    if unique_true.size == 0 or unique_pred.size == 0:
        return {}
    matrix = contingency_matrix(y_true, y_pred)
    cost = matrix.max() - matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    return {int(unique_pred[col]): str(unique_true[row]) for row, col in zip(row_ind, col_ind)}


def compute_per_class_purity(true_labels: pd.Series, pred_labels: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"true_label": true_labels.astype(str), "pred_label": pred_labels})
    rows = []
    for label, group in df.groupby("true_label"):
        counts = group["pred_label"].value_counts()
        top_pred = counts.index[0]
        purity = float(counts.iloc[0] / counts.sum()) if counts.sum() else 0.0
        rows.append(
            {
                "true_label": label,
                "purity": purity,
                "dominant_cluster": int(top_pred),
                "count": int(counts.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("purity", ascending=False)


def compute_top_confusion_pairs(
    true_labels: pd.Series,
    pred_labels: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    mapping = compute_hungarian_mapping(true_labels.to_numpy(), pred_labels)
    mapped_preds = [mapping.get(int(pred), "unmatched") for pred in pred_labels]
    df = pd.DataFrame({"true_label": true_labels.astype(str), "pred_label": mapped_preds})
    confusion = pd.crosstab(df["true_label"], df["pred_label"])
    pairs = []
    for true_label in confusion.index:
        for pred_label in confusion.columns:
            if true_label == pred_label or pred_label == "unmatched":
                continue
            count = int(confusion.loc[true_label, pred_label])
            if count > 0:
                pairs.append({"true_label": true_label, "pred_label": pred_label, "count": count})
    pairs_sorted = sorted(pairs, key=lambda item: item["count"], reverse=True)[:top_k]
    return pd.DataFrame(pairs_sorted)


def compute_knn_predictions(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    label_codes, label_mapping = encode_labels(pd.Series(labels))
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)
    neighbor_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    preds = np.empty(len(label_codes), dtype=int)
    for i in range(len(label_codes)):
        votes = label_codes[neighbor_idx[i]]
        preds[i] = np.bincount(votes, minlength=len(label_mapping)).argmax()
    return preds, label_codes, label_mapping


def compute_knn_confusion_matrix(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str,
) -> pd.DataFrame:
    if embeddings.shape[0] <= 1:
        return pd.DataFrame()
    preds, label_codes, label_mapping = compute_knn_predictions(embeddings, labels, k=1, metric=metric)
    class_names = [label_mapping[idx] for idx in range(len(label_mapping))]
    matrix = pd.crosstab(
        pd.Series(label_codes, name="true"),
        pd.Series(preds, name="pred"),
        rownames=["true"],
        colnames=["pred"],
        dropna=False,
    )
    matrix = matrix.reindex(index=range(len(class_names)), columns=range(len(class_names)), fill_value=0)
    matrix.index = class_names
    matrix.columns = class_names
    return matrix


def compute_cluster_medoids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_ids: Iterable[str],
    paths: Iterable[str],
    metric: str,
) -> pd.DataFrame:
    ids = list(sample_ids)
    path_list = list(paths)
    rows = []
    for cluster_id in sorted({int(label) for label in labels if label != -1}):
        idx = np.where(labels == cluster_id)[0]
        if idx.size == 0:
            continue
        if idx.size == 1:
            medoid_idx = idx[0]
        else:
            distances = cdist(embeddings[idx], embeddings[idx], metric=metric)
            medoid_idx = idx[int(np.argmin(distances.mean(axis=1)))]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "sample_id": ids[medoid_idx],
                "path": path_list[medoid_idx],
            }
        )
    return pd.DataFrame(rows)


def render_report(
    output_dir: Path,
    template_path: Path,
    title: str,
    assignments: pd.DataFrame,
    auto_result: AutoKResult,
    plots: List[PlotEntry],
    recommended_settings: Dict[str, str],
    run_settings: Dict[str, str],
    settings_diff: List[Dict[str, str]],
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
        recommended_settings=recommended_settings,
        run_settings=run_settings,
        settings_diff=settings_diff,
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
    parser.add_argument("--meta", type=Path, required=True, help="Metadata CSV")
    parser.add_argument("--config", type=Path, default=Path("configs/cluster.yaml"), help="Clustering YAML config")
    parser.add_argument("--out-dir", type=Path, default=Path("./out"), help="Output directory")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_DEFINITIONS.keys()),
        default=None,
        help="Apply preset defaults (CLI > preset > YAML > code defaults).",
    )
    parser.add_argument(
        "--kmeans-k",
        default=None,
        help="Fix K-Means clusters (integer) or use auto selection (default: auto).",
    )
    parser.add_argument(
        "--distance-metric",
        choices=("euclidean", "cosine"),
        default=None,
        help="Distance metric for kNN and distance computations (default: euclidean).",
    )
    parser.add_argument(
        "--scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply StandardScaler before clustering.",
    )
    parser.add_argument(
        "--preproc-order",
        choices=("raw", "l2", "scale->pca->l2", "l2->pca", "l2->pca->scale"),
        default=None,
        help="Preprocessing order override for clustering embeddings.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="Apply PCA with a fixed integer dimension. 0 disables.",
    )
    parser.add_argument(
        "--pca-var",
        type=float,
        default=None,
        help="Apply PCA using variance ratio (0-1). Overrides --pca-dim.",
    )
    parser.add_argument(
        "--pca-components",
        type=float,
        default=None,
        help="Apply PCA with N components (int) or variance ratio (0-1). 0/None disables.",
    )
    parser.add_argument(
        "--whiten",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whiten PCA components.",
    )
    parser.add_argument(
        "--l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply L2 normalization before clustering (recommended for cosine distance).",
    )
    parser.add_argument(
        "--preprocess-sweep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run preprocessing ablation sweep and save preprocess_sweep.csv (default: enabled).",
    )
    parser.add_argument(
        "--label-col",
        "--label-column",
        dest="label_col",
        default=None,
        help="Metadata column containing class labels.",
    )
    parser.add_argument(
        "--run-hdbscan",
        action="store_true",
        default=None,
        help="Run HDBSCAN clustering (default: enabled).",
    )
    parser.add_argument(
        "--no-hdbscan",
        action="store_true",
        help="Disable HDBSCAN clustering.",
    )
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=None, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--hdbscan-min-samples", type=int, default=None, help="HDBSCAN min_samples.")
    parser.add_argument(
        "--hdbscan-min-samples-sweep",
        default=None,
        help="Comma-separated HDBSCAN min_samples values to sweep (default: 1,2,5,10).",
    )
    parser.add_argument(
        "--hdbscan-min-samples-mode",
        choices=("fixed", "auto"),
        default=None,
        help="When sweeping, keep min_samples fixed or tie it to min_cluster_size.",
    )
    parser.add_argument(
        "--hdbscan-metric",
        default=None,
        help="HDBSCAN metric (default: euclidean).",
    )
    parser.add_argument("--hdbscan-sweep", action="store_true", default=None, help="Sweep min_cluster_size values.")
    parser.add_argument("--hdbscan-sweep-min", type=int, default=None, help="HDBSCAN sweep min.")
    parser.add_argument("--hdbscan-sweep-max", type=int, default=None, help="HDBSCAN sweep max.")
    parser.add_argument("--hdbscan-sweep-step", type=int, default=None, help="HDBSCAN sweep step.")
    return parser.parse_args()


def parse_kmeans_k(raw: Union[str, int, None]) -> Optional[int]:
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


def apply_preprocessing(
    embeddings: np.ndarray,
    order: Optional[str],
    scale_enabled: bool,
    pca_components: Optional[float],
    whiten: bool,
    l2_normalize: bool,
) -> Tuple[np.ndarray, Dict[str, Optional[float]], List[Dict[str, object]]]:
    working = embeddings
    applied = {"scale": False, "pca_components": None, "whiten": whiten, "l2_normalize": False}
    steps: List[Dict[str, object]] = []

    def do_scale() -> None:
        nonlocal working
        working, _ = apply_scaling(working, True)
        applied["scale"] = True
        steps.append({"step": "scale", "params": {"with_mean": True, "with_std": True}})

    def do_pca() -> None:
        nonlocal working
        if pca_components in (None, 0):
            logging.warning("PCA requested in preprocessing order but pca_components is disabled; skipping PCA.")
            return
        pca_result = apply_pca(working, pca_components, whiten)
        if pca_result is None:
            logging.warning("PCA returned None; skipping PCA step.")
            return
        working = pca_result
        applied["pca_components"] = pca_components
        steps.append(
            {
                "step": "pca",
                "params": {
                    "components": pca_components,
                    "whiten": whiten,
                    "random_state": 42,
                },
            }
        )

    def do_l2() -> None:
        nonlocal working
        working = apply_l2_normalization(working)
        applied["l2_normalize"] = True
        steps.append({"step": "l2_normalize", "params": {"norm": "l2"}})

    if order is None:
        if scale_enabled:
            do_scale()
        if pca_components not in (None, 0):
            do_pca()
        if l2_normalize:
            do_l2()
    elif order == "raw":
        pass
    elif order == "l2":
        do_l2()
    elif order == "scale->pca->l2":
        do_scale()
        do_pca()
        do_l2()
    elif order == "l2->pca":
        do_l2()
        do_pca()
    elif order == "l2->pca->scale":
        do_l2()
        do_pca()
        do_scale()
    else:
        raise ValueError(f"Unknown preprocessing order: {order}")

    return working, applied, steps


def run_preprocess_sweep(
    embeddings: np.ndarray,
    true_labels_raw: pd.Series,
    true_labels_encoded: np.ndarray,
    cfg: PipelineConfig,
    output_dir: Path,
    distance_metric: str,
    hdbscan_min_samples: Optional[int],
) -> None:
    sweep_rows: List[Dict[str, Optional[float]]] = []
    summary_paths: List[Path] = []
    pca_dims = [None, 64, 128]
    for scale in (False, True):
        for pca_dim in pca_dims:
            for whiten in (False, True):
                if whiten and pca_dim is None:
                    continue
                for l2_normalize in (False, True):
                    preprocessed, applied, steps = apply_preprocessing(
                        embeddings=embeddings,
                        order=None,
                        scale_enabled=scale,
                        pca_components=None if pca_dim is None else float(pca_dim),
                        whiten=whiten,
                        l2_normalize=l2_normalize,
                    )
                    ensure_non_degenerate_embeddings(preprocessed, "Preprocess sweep input")
                    kmeans_labels, _ = run_kmeans(
                        preprocessed,
                        40,
                        cfg.kmeans_init,
                        cfg.kmeans_n_init,
                        distance_metric,
                    )
                    kmeans_metrics = compute_cluster_metrics(true_labels_encoded, kmeans_labels)
                    kmeans_hungarian = compute_hungarian_accuracy(true_labels_encoded, kmeans_labels)
                    knn_metrics, _, _ = compute_knn_accuracy(
                        preprocessed,
                        true_labels_raw.to_numpy(),
                        ks=[1, 5, 10],
                        metric=distance_metric,
                    )
                    hdbscan_embeddings = preprocessed
                    additional_hdbscan_scaling = False
                    if not scale and is_scale_inappropriate(preprocessed):
                        hdbscan_embeddings, _ = apply_scaling(preprocessed, True)
                        additional_hdbscan_scaling = True
                    hdbscan_cfg = dict(cfg.hdbscan_cfg)
                    hdbscan_cfg.update(
                        {
                            "min_cluster_size": 5,
                            "min_samples": hdbscan_min_samples,
                            "metric": "euclidean",
                        }
                    )
                    hdbscan_model = run_hdbscan(hdbscan_embeddings, hdbscan_cfg)
                    hdbscan_labels = hdbscan_model.labels_
                    hdbscan_noise_rate = float(np.mean(hdbscan_labels == -1))
                    hdbscan_metrics = compute_cluster_metrics(true_labels_encoded, hdbscan_labels)
                    hdbscan_metrics_no_noise = compute_cluster_metrics(
                        true_labels_encoded, hdbscan_labels, mask=hdbscan_labels != -1
                    )
                    variant_id = build_preprocess_variant_id(scale, pca_dim, whiten, l2_normalize)
                    summary = {
                        "generated_at": datetime.utcnow().isoformat(),
                        "preprocess": build_preprocess_summary(
                            steps=steps,
                            applied=applied,
                            distance_metric=distance_metric,
                            order_override=None,
                        ),
                        "kmeans": {
                            "k": 40,
                            "k_source": "fixed",
                            "metrics": kmeans_metrics,
                            "hungarian_accuracy": kmeans_hungarian,
                        },
                        "knn_accuracy": knn_metrics,
                        "hdbscan": {
                            "config": {
                                "min_cluster_size": 5,
                                "min_samples": hdbscan_cfg.get("min_samples"),
                                "metric": hdbscan_cfg.get("metric", "euclidean"),
                                "extra_hdbscan_scale": additional_hdbscan_scaling,
                            },
                            "noise_rate": hdbscan_noise_rate,
                            "coverage": 1.0 - hdbscan_noise_rate,
                            "metrics": hdbscan_metrics,
                            "metrics_no_noise": hdbscan_metrics_no_noise,
                        },
                    }
                    summary_path = output_dir / f"summary_{variant_id}.json"
                    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    summary_paths.append(summary_path)
                    row: Dict[str, Optional[float]] = {
                        "scale": scale,
                        "pca_dim": pca_dim,
                        "whiten": whiten,
                        "l2_normalize": l2_normalize,
                        "distance_metric": distance_metric,
                        "knn_overall_k1": knn_metrics["1"]["overall"],
                        "knn_macro_k1": knn_metrics["1"]["macro"],
                        "knn_overall_k5": knn_metrics["5"]["overall"],
                        "knn_macro_k5": knn_metrics["5"]["macro"],
                        "knn_overall_k10": knn_metrics["10"]["overall"],
                        "knn_macro_k10": knn_metrics["10"]["macro"],
                        "kmeans_ari": kmeans_metrics["ari"],
                        "kmeans_nmi": kmeans_metrics["nmi"],
                        "kmeans_purity": kmeans_metrics["purity"],
                        "kmeans_hungarian_accuracy": kmeans_hungarian,
                        "hdbscan_noise_rate": hdbscan_noise_rate,
                        "hdbscan_coverage": 1.0 - hdbscan_noise_rate,
                        "hdbscan_ari_no_noise": hdbscan_metrics_no_noise["ari"],
                        "hdbscan_nmi_no_noise": hdbscan_metrics_no_noise["nmi"],
                        "hdbscan_purity_no_noise": hdbscan_metrics_no_noise["purity"],
                        "summary_path": str(summary_path),
                    }
                    sweep_rows.append(row)

    if sweep_rows:
        sweep_path = output_dir / "preprocess_sweep.csv"
        pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
        logging.info("Saved preprocess sweep results to %s", sweep_path)


def build_fixed_k_result(k: int) -> AutoKResult:
    curves = MetricCurves(ks=[], inertia=[], silhouette=[], gap=[], gap_std=[], bic=[])
    return AutoKResult(consensus_k=k, elbow_k=k, silhouette_k=k, gap_k=k, bic_k=k, curves=curves)


def sweep_values(min_value: int, max_value: int, step: int) -> List[int]:
    if min_value <= 0 or max_value < min_value or step <= 0:
        raise ValueError("Invalid HDBSCAN sweep range. Ensure min <= max and step > 0.")
    return list(range(min_value, max_value + 1, step))


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def build_preprocess_summary(
    steps: List[Dict[str, object]],
    applied: Dict[str, Optional[float]],
    distance_metric: str,
    order_override: Optional[str],
) -> Dict[str, object]:
    if steps:
        order_label = "->".join(step["step"] for step in steps)
    else:
        order_label = "raw"
    return {
        "order": order_label,
        "order_requested": order_override,
        "steps": steps,
        "scale": applied["scale"],
        "pca_components": applied["pca_components"],
        "whiten": applied["whiten"],
        "l2_normalize": applied["l2_normalize"],
        "distance_metric": distance_metric,
    }


def build_settings_diff(recommended: Dict[str, str], actual: Dict[str, str]) -> List[Dict[str, str]]:
    diffs = []
    for key in sorted(set(recommended.keys()) | set(actual.keys())):
        rec = recommended.get(key, "")
        act = actual.get(key, "")
        if rec != act:
            diffs.append({"key": key, "recommended": rec, "actual": act})
    return diffs


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    warnings.filterwarnings(
        "ignore",
        message=".*'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.*",
        category=FutureWarning,
    )
    random.seed(42)
    np.random.seed(42)

    cfg = load_config(args.config)
    embeddings = load_embeddings(args.emb)
    if not args.meta.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.meta}")
    metadata = pd.read_csv(args.meta)
    if len(metadata) != len(embeddings):
        raise ValueError(
            "Metadata row count does not match embeddings. "
            f"meta={len(metadata)} embeddings={len(embeddings)}"
        )
    if "sample_id" not in metadata.columns:
        metadata["sample_id"] = metadata.index.astype(str)

    label_column = select_label_column(metadata, args.label_col)
    true_labels_raw = metadata[label_column]
    true_labels_encoded, label_mapping = encode_labels(true_labels_raw)
    logging.info("Using label column '%s' with %d classes.", label_column, len(label_mapping))

    preset = PRESET_DEFINITIONS.get(args.preset) if args.preset else None
    preset_settings = preset.get("settings", {}) if preset else {}

    distance_metric = str(
        resolve_value(args.distance_metric, preset_settings.get("distance_metric"), None, "euclidean")
    )
    scale_enabled = bool(resolve_value(args.scale, preset_settings.get("scale"), cfg.scale, False))
    pca_components = resolve_pca_components_with_precedence(args, cfg, preset_settings)
    whiten = bool(resolve_value(args.whiten, preset_settings.get("whiten"), cfg.whiten, False))
    use_l2_norm = bool(
        resolve_value(
            args.l2_normalize,
            preset_settings.get("l2_normalize"),
            None,
            distance_metric == "cosine",
        )
    )
    preproc_order = args.preproc_order or preset_settings.get("preproc_order")
    kmeans_k_raw = resolve_value(args.kmeans_k, preset_settings.get("kmeans_k"), None, "auto")
    preprocess_sweep_enabled = bool(
        resolve_value(args.preprocess_sweep, preset_settings.get("preprocess_sweep"), None, True)
    )

    if args.no_hdbscan:
        hdbscan_enabled = False
    else:
        hdbscan_enabled = bool(
            resolve_value(args.run_hdbscan, preset_settings.get("run_hdbscan"), None, True)
        )
    hdbscan_sweep = bool(resolve_value(args.hdbscan_sweep, preset_settings.get("hdbscan_sweep"), None, False))
    hdbscan_min_cluster_size = int(
        resolve_value(
            args.hdbscan_min_cluster_size,
            preset_settings.get("hdbscan_min_cluster_size"),
            cfg.hdbscan_cfg.get("min_cluster_size"),
            8,
        )
    )
    hdbscan_min_samples = resolve_value(
        args.hdbscan_min_samples,
        preset_settings.get("hdbscan_min_samples"),
        cfg.hdbscan_cfg.get("min_samples"),
        None,
    )
    hdbscan_metric = str(
        resolve_value(
            args.hdbscan_metric,
            preset_settings.get("hdbscan_metric"),
            cfg.hdbscan_cfg.get("metric"),
            "euclidean",
        )
    )
    hdbscan_min_samples_sweep_raw = str(
        resolve_value(
            args.hdbscan_min_samples_sweep,
            preset_settings.get("hdbscan_min_samples_sweep"),
            None,
            "1,2,5,10",
        )
    )
    hdbscan_min_samples_mode = str(
        resolve_value(
            args.hdbscan_min_samples_mode,
            preset_settings.get("hdbscan_min_samples_mode"),
            None,
            "fixed",
        )
    )
    hdbscan_sweep_min = int(
        resolve_value(
            args.hdbscan_sweep_min,
            preset_settings.get("hdbscan_sweep_min"),
            None,
            5,
        )
    )
    hdbscan_sweep_max = int(
        resolve_value(
            args.hdbscan_sweep_max,
            preset_settings.get("hdbscan_sweep_max"),
            None,
            80,
        )
    )
    hdbscan_sweep_step = int(
        resolve_value(
            args.hdbscan_sweep_step,
            preset_settings.get("hdbscan_sweep_step"),
            None,
            5,
        )
    )

    clustering_embeddings, applied_preproc, preproc_steps = apply_preprocessing(
        embeddings=embeddings,
        order=preproc_order,
        scale_enabled=scale_enabled,
        pca_components=pca_components,
        whiten=whiten,
        l2_normalize=use_l2_norm,
    )
    preprocess_summary = build_preprocess_summary(
        steps=preproc_steps,
        applied=applied_preproc,
        distance_metric=distance_metric,
        order_override=preproc_order,
    )
    logging.info(
        "Preprocessing config: order=%s, scale=%s, pca_components=%s, whiten=%s, l2_normalize=%s",
        preprocess_summary["order"],
        applied_preproc["scale"],
        applied_preproc["pca_components"],
        applied_preproc["whiten"],
        applied_preproc["l2_normalize"],
    )
    if applied_preproc["l2_normalize"] and distance_metric == "cosine":
        logging.info("Cosine distance with L2-normalized embeddings is equivalent to Euclidean distance for KMeans.")
    elif distance_metric == "cosine" and not applied_preproc["l2_normalize"]:
        logging.warning("Cosine distance selected without L2 normalization; consider --l2-normalize for KMeans.")

    ensure_non_degenerate_embeddings(clustering_embeddings, "KMeans input")
    log_feature_summary(clustering_embeddings, "Clustering input")
    log_feature_statistics(clustering_embeddings, "Clustering input")

    fixed_k = parse_kmeans_k(kmeans_k_raw)
    if fixed_k is not None:
        logging.info("Overriding consensus k with fixed K-Means k=%d", fixed_k)
        auto_k_result = build_fixed_k_result(fixed_k)
        k_source = "fixed"
    else:
        auto_k_result = auto_select_k(
            clustering_embeddings,
            k_min=cfg.k_min,
            k_max=cfg.k_max,
            reference_samples=cfg.gap_reference,
            covariance_type=cfg.covariance_type,
        )
        k_source = "auto"
        logging.info("Consensus k determined as %d", auto_k_result.consensus_k)

    kmeans_labels, kmeans_distances = run_kmeans(
        clustering_embeddings,
        auto_k_result.consensus_k,
        cfg.kmeans_init,
        cfg.kmeans_n_init,
        distance_metric,
    )
    hdbscan_labels = None
    hdbscan_noise_rate = None
    hdbscan_metrics = None
    hdbscan_metrics_no_noise = None
    hdbscan_best_score = None
    hdbscan_sweep_rows: List[Dict[str, Optional[float]]] = []
    hdbscan_embeddings = None
    if hdbscan_enabled:
        hdbscan_embeddings = clustering_embeddings
        additional_hdbscan_scaling = False
        if not scale_enabled and is_scale_inappropriate(clustering_embeddings):
            logging.warning(
                "HDBSCAN features show wide scale variance; applying StandardScaler before HDBSCAN."
            )
            hdbscan_embeddings, _ = apply_scaling(clustering_embeddings, True)
            additional_hdbscan_scaling = True
        hdbscan_cfg = dict(cfg.hdbscan_cfg)
        hdbscan_cfg.update(
            {
                "min_cluster_size": hdbscan_min_cluster_size,
                "min_samples": hdbscan_min_samples,
                "metric": hdbscan_metric,
            }
        )
        ensure_non_degenerate_embeddings(hdbscan_embeddings, "HDBSCAN input")
        logging.info(
            "HDBSCAN config: min_cluster_size=%s, min_samples=%s, metric=%s, preprocessing_scale=%s, extra_hdbscan_scale=%s",
            hdbscan_cfg.get("min_cluster_size", 8),
            hdbscan_cfg.get("min_samples"),
            hdbscan_cfg.get("metric", "euclidean"),
            scale_enabled,
            additional_hdbscan_scaling,
        )

        if hdbscan_sweep:
            sweep_values_list = sweep_values(
                hdbscan_sweep_min, hdbscan_sweep_max, hdbscan_sweep_step
            )
            min_samples_candidates = parse_int_list(hdbscan_min_samples_sweep_raw)
            if hdbscan_min_samples_mode == "auto":
                min_samples_candidates = []
            if not min_samples_candidates:
                min_samples_candidates = [hdbscan_cfg.get("min_samples")]
            best_score = float("-inf")
            best_noise = float("inf")
            best_labels = None
            best_cfg = None
            for min_cluster_size in sweep_values_list:
                for min_samples in min_samples_candidates:
                    if hdbscan_min_samples_mode == "auto":
                        min_samples = min_cluster_size
                    sweep_cfg = dict(
                        hdbscan_cfg,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                    )
                    model = run_hdbscan(hdbscan_embeddings, sweep_cfg)
                    labels = model.labels_
                    noise_rate = float(np.mean(labels == -1))
                    clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                    metrics_all = compute_cluster_metrics(true_labels_encoded, labels)
                    metrics_no_noise = compute_cluster_metrics(
                        true_labels_encoded, labels, mask=labels != -1
                    )
                    coverage = 1.0 - noise_rate
                    purity_no_noise = metrics_no_noise["purity"]
                    score = (purity_no_noise * coverage) if purity_no_noise is not None else float("-inf")
                    hdbscan_sweep_rows.append(
                        {
                            "min_cluster_size": min_cluster_size,
                            "min_samples": sweep_cfg.get("min_samples"),
                            "clusters_found": clusters_found,
                            "noise_rate": noise_rate,
                            "coverage": coverage,
                            "ari": metrics_all["ari"],
                            "nmi": metrics_all["nmi"],
                            "purity": metrics_all["purity"],
                            "ari_no_noise": metrics_no_noise["ari"],
                            "nmi_no_noise": metrics_no_noise["nmi"],
                            "purity_no_noise": metrics_no_noise["purity"],
                            "score": score,
                            "n_samples": metrics_all["n_samples"],
                            "n_samples_no_noise": metrics_no_noise["n_samples"],
                        }
                    )
                    if score > best_score or (score == best_score and noise_rate < best_noise):
                        best_score = score
                        best_noise = noise_rate
                        best_labels = labels
                        best_cfg = sweep_cfg
            if best_labels is None or best_cfg is None:
                raise ValueError("HDBSCAN sweep failed to produce valid clustering results.")
            hdbscan_labels = best_labels
            hdbscan_cfg = best_cfg
            hdbscan_noise_rate = float(np.mean(hdbscan_labels == -1))
            hdbscan_best_score = best_score
        else:
            hdbscan_model = run_hdbscan(hdbscan_embeddings, hdbscan_cfg)
            hdbscan_labels = hdbscan_model.labels_
            hdbscan_noise_rate = float(np.mean(hdbscan_labels == -1))
            if hdbscan_noise_rate > 0.5:
                relaxed_cfg = relax_hdbscan_params(hdbscan_cfg)
                logging.warning("High HDBSCAN noise rate detected; relaxing params to %s", relaxed_cfg)
                hdbscan_model = run_hdbscan(hdbscan_embeddings, relaxed_cfg)
                hdbscan_labels = hdbscan_model.labels_
                hdbscan_noise_rate = float(np.mean(hdbscan_labels == -1))
                hdbscan_cfg = relaxed_cfg

        hdbscan_metrics = compute_cluster_metrics(true_labels_encoded, hdbscan_labels)
        hdbscan_metrics_no_noise = compute_cluster_metrics(
            true_labels_encoded, hdbscan_labels, mask=hdbscan_labels != -1
        )
        logging.info("HDBSCAN noise (-1) rate: %.2f", hdbscan_noise_rate)

    gmm = GaussianMixture(n_components=auto_k_result.consensus_k, covariance_type=cfg.covariance_type, random_state=42)
    gmm.fit(clustering_embeddings)

    ambiguity_mask = compute_ambiguity(
        clustering_embeddings,
        kmeans_labels,
        kmeans_distances,
        hdbscan_labels if hdbscan_labels is not None else np.full(len(kmeans_labels), -1),
        cfg.silhouette_threshold,
        cfg.posterior_threshold,
        cfg.distance_quantile,
        gmm,
        distance_metric,
    )

    consensus_labels = kmeans_labels.copy()

    df = metadata.copy()
    df["kmeans"] = kmeans_labels
    df["hdbscan"] = hdbscan_labels if hdbscan_labels is not None else -1
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

    path_column = select_path_column(metadata)
    if path_column:
        path_values = metadata[path_column].astype(str)
    else:
        path_values = metadata.index.astype(str)

    kmeans_assignments = pd.DataFrame(
        {
            "path": path_values,
            "true_label": true_labels_raw.astype(str),
            "pred_label": kmeans_labels,
        }
    )
    kmeans_assignments.to_csv(output_dir / "kmeans_assignments.csv", index=False)

    if hdbscan_labels is not None:
        hdbscan_assignments = pd.DataFrame(
            {
                "path": path_values,
                "true_label": true_labels_raw.astype(str),
                "pred_label": hdbscan_labels,
            }
        )
        hdbscan_assignments.to_csv(output_dir / "hdbscan_assignments.csv", index=False)

    kmeans_class_purity_path = None
    kmeans_confusion_pairs_path = None
    hdbscan_core_assignments_path = None
    hdbscan_medoids_path = None
    knn_confusion_path = None

    if args.preset == "kmeans40_labelmatch":
        per_class_purity_df = compute_per_class_purity(true_labels_raw, kmeans_labels)
        kmeans_class_purity_path = output_dir / "kmeans_per_class_purity.csv"
        per_class_purity_df.to_csv(kmeans_class_purity_path, index=False)
        confusion_df = compute_top_confusion_pairs(true_labels_raw, kmeans_labels)
        kmeans_confusion_pairs_path = output_dir / "kmeans_top_confusions.csv"
        confusion_df.to_csv(kmeans_confusion_pairs_path, index=False)

    if args.preset == "hdbscan_core" and hdbscan_labels is not None:
        hdbscan_core_df = pd.DataFrame(
            {
                "sample_id": metadata["sample_id"],
                "path": path_values,
                "true_label": true_labels_raw.astype(str),
                "core_label": hdbscan_labels,
            }
        )
        hdbscan_core_assignments_path = output_dir / "hdbscan_core_assignments.csv"
        hdbscan_core_df.to_csv(hdbscan_core_assignments_path, index=False)
        medoid_embeddings = hdbscan_embeddings if hdbscan_embeddings is not None else clustering_embeddings
        medoids_df = compute_cluster_medoids(
            medoid_embeddings,
            hdbscan_labels,
            metadata["sample_id"].astype(str),
            path_values,
            metric=distance_metric,
        )
        hdbscan_medoids_path = output_dir / "hdbscan_core_medoids.csv"
        medoids_df.to_csv(hdbscan_medoids_path, index=False)

    if preset_settings:
        recommended_settings = {
            "preset": args.preset or "none",
            "pca_dim": str(preset_settings.get("pca_dim", "none")),
            "scale": str(preset_settings.get("scale", "none")),
            "whiten": str(preset_settings.get("whiten", "none")),
            "l2_normalize": str(preset_settings.get("l2_normalize", "none")),
            "distance_metric": str(preset_settings.get("distance_metric", "none")),
            "kmeans_k": str(preset_settings.get("kmeans_k", "auto")),
            "run_hdbscan": str(preset_settings.get("run_hdbscan", "auto")),
            "hdbscan_sweep": str(preset_settings.get("hdbscan_sweep", "auto")),
        }
    else:
        recommended_settings = {
            "preset": "none",
            "pca_dim": "64 or 128",
            "scale": "False",
            "whiten": "False",
            "l2_normalize": "True",
            "distance_metric": "cosine",
        }
    run_settings = {
        "preprocess_order": preprocess_summary["order"],
        "scale": str(applied_preproc["scale"]),
        "pca_components": str(applied_preproc["pca_components"]),
        "whiten": str(applied_preproc["whiten"]),
        "l2_normalize": str(applied_preproc["l2_normalize"]),
        "distance_metric": distance_metric,
        "kmeans_k": str(kmeans_k_raw),
        "run_hdbscan": str(hdbscan_enabled),
        "hdbscan_sweep": str(hdbscan_sweep),
    }
    settings_diff = build_settings_diff(recommended_settings, run_settings)
    logging.info("Preset recommended settings: %s", recommended_settings)
    logging.info("Run settings: %s", run_settings)
    if settings_diff:
        logging.info("Settings diff: %s", settings_diff)
    else:
        logging.info("Run settings match preset recommendations.")
    render_report(
        output_dir,
        cfg.html_template,
        cfg.report_title,
        df,
        auto_k_result,
        plot_entries,
        recommended_settings,
        run_settings,
        settings_diff,
    )

    kmeans_metrics = compute_cluster_metrics(true_labels_encoded, kmeans_labels)
    kmeans_hungarian = compute_hungarian_accuracy(true_labels_encoded, kmeans_labels)
    knn_metrics, knn_per_class_k1, knn_per_class_counts = compute_knn_accuracy(
        clustering_embeddings,
        true_labels_raw.to_numpy(),
        ks=[1, 5, 10],
        metric=distance_metric,
    )
    if args.preset == "retrieval_knn":
        knn_confusion = compute_knn_confusion_matrix(
            clustering_embeddings,
            true_labels_raw.to_numpy(),
            metric=distance_metric,
        )
        knn_confusion_path = output_dir / "knn_confusion_matrix_k1.csv"
        knn_confusion.to_csv(knn_confusion_path)

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "preprocess": preprocess_summary,
        "label_column": label_column,
        "preset": {
            "name": args.preset,
            "recommended_settings": recommended_settings,
            "run_settings": run_settings,
            "diff": settings_diff,
        },
        "kmeans": {
            "k": int(auto_k_result.consensus_k),
            "k_source": k_source,
            "metrics": kmeans_metrics,
            "hungarian_accuracy": kmeans_hungarian,
            "cluster_sizes": compute_cluster_sizes(kmeans_labels),
        },
        "knn_accuracy": knn_metrics,
    }

    if k_source == "auto":
        summary["auto_k"] = {
            "consensus_k": int(auto_k_result.consensus_k),
            "elbow_k": int(auto_k_result.elbow_k),
            "silhouette_k": int(auto_k_result.silhouette_k),
            "gap_k": int(auto_k_result.gap_k),
            "bic_k": int(auto_k_result.bic_k),
        }

    if hdbscan_labels is not None:
        summary["hdbscan"] = {
            "config": {
                "min_cluster_size": int(hdbscan_cfg.get("min_cluster_size", 0)),
                "min_samples": hdbscan_cfg.get("min_samples"),
                "metric": hdbscan_cfg.get("metric", "euclidean"),
            },
            "clusters_found": len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0),
            "noise_rate": hdbscan_noise_rate,
            "coverage": None if hdbscan_noise_rate is None else 1.0 - hdbscan_noise_rate,
            "metrics": hdbscan_metrics,
            "metrics_no_noise": hdbscan_metrics_no_noise,
            "score": hdbscan_best_score,
            "cluster_sizes": compute_cluster_sizes(hdbscan_labels),
        }
        if hdbscan_core_assignments_path is not None:
            summary["hdbscan"]["core_assignments_path"] = str(hdbscan_core_assignments_path)
        if hdbscan_medoids_path is not None:
            summary["hdbscan"]["core_medoids_path"] = str(hdbscan_medoids_path)

    if kmeans_class_purity_path is not None:
        summary["kmeans"]["per_class_purity_path"] = str(kmeans_class_purity_path)
    if kmeans_confusion_pairs_path is not None:
        summary["kmeans"]["top_confusions_path"] = str(kmeans_confusion_pairs_path)
    if knn_confusion_path is not None:
        summary["knn_confusion_matrix_k1_path"] = str(knn_confusion_path)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if hdbscan_sweep_rows:
        sweep_path = output_dir / "hdbscan_sweep.csv"
        pd.DataFrame(hdbscan_sweep_rows).to_csv(sweep_path, index=False)
        summary.setdefault("hdbscan", {})["sweep_path"] = str(sweep_path)

    if preprocess_sweep_enabled:
        run_preprocess_sweep(
            embeddings=embeddings,
            true_labels_raw=true_labels_raw,
            true_labels_encoded=true_labels_encoded,
            cfg=cfg,
            output_dir=output_dir,
            distance_metric="cosine",
            hdbscan_min_samples=hdbscan_min_samples,
        )

    if knn_per_class_k1:
        knn_df = pd.DataFrame(
            [
                {"label": label, "knn_at_1": acc, "count": knn_per_class_counts.get(label, 0)}
                for label, acc in knn_per_class_k1.items()
            ]
        )
        knn_df.to_csv(output_dir / "knn_per_class_at1.csv", index=False)

    logging.info(
        "kNN acc@1/5/10 overall: %s",
        {k: v["overall"] for k, v in knn_metrics.items()},
    )
    logging.info(
        "kNN acc@1/5/10 macro: %s",
        {k: v["macro"] for k, v in knn_metrics.items()},
    )
    logging.info(
        "KMeans(k=%d) ARI=%.4f NMI=%.4f purity=%.4f",
        auto_k_result.consensus_k,
        kmeans_metrics["ari"],
        kmeans_metrics["nmi"],
        kmeans_metrics["purity"],
    )
    logging.info("KMeans Hungarian-matched accuracy: %.4f", kmeans_hungarian)
    if hdbscan_labels is not None:
        logging.info(
            "HDBSCAN best: clusters=%d noise=%.2f ARI=%s NMI=%s purity=%s "
            "(no-noise ARI=%s NMI=%s purity=%s) score=%s",
            summary["hdbscan"]["clusters_found"],
            hdbscan_noise_rate,
            format_metric(hdbscan_metrics["ari"]),
            format_metric(hdbscan_metrics["nmi"]),
            format_metric(hdbscan_metrics["purity"]),
            format_metric(hdbscan_metrics_no_noise["ari"]),
            format_metric(hdbscan_metrics_no_noise["nmi"]),
            format_metric(hdbscan_metrics_no_noise["purity"]),
            format_metric(hdbscan_best_score),
        )


if __name__ == "__main__":
    main()
