"""Utilities for automatic selection of the number of clusters."""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


@dataclass
class MetricCurves:
    ks: List[int]
    inertia: List[float]
    silhouette: List[float]
    gap: List[float]
    gap_std: List[float]
    bic: List[float]


@dataclass
class AutoKResult:
    consensus_k: int
    elbow_k: int
    silhouette_k: int
    gap_k: int
    bic_k: int
    curves: MetricCurves


def _compute_kmeans_inertia(embeddings: np.ndarray, k_values: Iterable[int], n_init: int) -> Tuple[List[int], List[float], Dict[int, KMeans]]:
    ks: List[int] = []
    inertia: List[float] = []
    models: Dict[int, KMeans] = {}
    for k in k_values:
        if k >= embeddings.shape[0]:
            continue
        model = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=42)
        model.fit(embeddings)
        ks.append(k)
        inertia.append(model.inertia_)
        models[k] = model
    return ks, inertia, models


def _compute_silhouette(embeddings: np.ndarray, ks: List[int], models: Dict[int, KMeans]) -> List[float]:
    scores: List[float] = []
    for k in ks:
        if k <= 1:
            scores.append(float("nan"))
            continue
        labels = models[k].labels_
        scores.append(silhouette_score(embeddings, labels))
    return scores


def _compute_gap_statistic(embeddings: np.ndarray, ks: List[int], reference_samples: int) -> Tuple[List[float], List[float]]:
    if not ks:
        return [], []
    mins = embeddings.min(axis=0)
    maxs = embeddings.max(axis=0)
    gaps: List[float] = []
    s_k: List[float] = []
    for k in ks:
        ref_inertia = []
        for _ in range(reference_samples):
            reference = np.random.uniform(mins, maxs, size=embeddings.shape)
            ref_model = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=None)
            ref_model.fit(reference)
            ref_inertia.append(ref_model.inertia_)

        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        model.fit(embeddings)
        log_wk = np.log(model.inertia_)
        gap = np.mean(np.log(ref_inertia)) - log_wk
        sdk = np.sqrt(1 + 1 / reference_samples) * np.std(np.log(ref_inertia))
        gaps.append(gap)
        s_k.append(sdk)
    return gaps, s_k


def _compute_bic(embeddings: np.ndarray, ks: List[int], covariance_type: str) -> List[float]:
    scores: List[float] = []
    for k in ks:
        model = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42)
        model.fit(embeddings)
        scores.append(model.bic(embeddings))
    return scores


def _determine_elbow(ks: List[int], inertia: List[float]) -> int:
    if len(ks) < 2:
        return ks[0] if ks else 1
    kl = KneeLocator(ks, inertia, curve="convex", direction="decreasing")
    return int(kl.elbow or ks[0])


def _determine_gap(gaps: List[float], s_k: List[float], ks: List[int]) -> int:
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - s_k[i + 1]:
            return ks[i]
    return ks[-1] if ks else 1


def _determine_bic(bic: List[float], ks: List[int]) -> int:
    if not bic:
        return ks[0] if ks else 1
    idx = int(np.argmin(bic))
    return ks[idx]


def majority_vote(candidates: List[int]) -> int:
    counter = Counter(candidates)
    most_common = counter.most_common()
    best_count = most_common[0][1]
    tied = [k for k, count in most_common if count == best_count]
    return min(tied)


def auto_select_k(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    reference_samples: int = 10,
    covariance_type: str = "full",
) -> AutoKResult:
    k_values = list(range(k_min, min(k_max, embeddings.shape[0] - 1) + 1))
    if not k_values:
        raise ValueError("Not enough samples to evaluate clustering range.")
    ks, inertia, models = _compute_kmeans_inertia(embeddings, k_values, n_init=10)
    silhouette = _compute_silhouette(embeddings, ks, models)
    gap, gap_std = _compute_gap_statistic(embeddings, ks, reference_samples)
    bic = _compute_bic(embeddings, ks, covariance_type)

    elbow_k = _determine_elbow(ks, inertia)
    if any(np.isfinite(silhouette)):
        silhouette_idx = int(np.nanargmax(np.array(silhouette)))
        silhouette_k = ks[silhouette_idx]
    else:
        silhouette_k = elbow_k
    gap_k = _determine_gap(gap, gap_std, ks) if gap else elbow_k
    bic_k = _determine_bic(bic, ks)

    consensus_k = majority_vote([elbow_k, silhouette_k, gap_k, bic_k])

    curves = MetricCurves(
        ks=ks,
        inertia=inertia,
        silhouette=silhouette,
        gap=gap,
        gap_std=gap_std,
        bic=bic,
    )

    return AutoKResult(
        consensus_k=consensus_k,
        elbow_k=elbow_k,
        silhouette_k=silhouette_k,
        gap_k=gap_k,
        bic_k=bic_k,
        curves=curves,
    )

*** End Patch
