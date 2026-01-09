"""Diagnostics for MeshMAE inputs and linear probe quality checks."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .extract_embeddings import list_meshes
from .meshmae_inputs import build_meshmae_inputs


@dataclass
class TensorStats:
    shape: Tuple[int, ...]
    count: int
    mean: float
    std: float
    min: float
    max: float
    nans: int
    infs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MeshMAE input/embedding diagnostics")
    parser.add_argument("--data", type=Path, default=None, help="Root directory with MAPS meshes.")
    parser.add_argument("--mesh-ext", nargs="*", default=None, help="Mesh extensions to include.")
    parser.add_argument(
        "--only-repaired-maps",
        action="store_true",
        help="Only include meshes ending with _repaired_MAPS.obj.",
    )
    parser.add_argument("--sample-limit", type=int, default=10, help="Max meshes to sample for input stats.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mesh sampling.")
    parser.add_argument("--feature-mode", type=str, default="paper10", help="MeshMAE feature mode.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for mesh input tensors.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to save diagnostics JSON.")
    parser.add_argument("--embeddings", type=Path, default=None, help="Embeddings .npy for linear probe.")
    parser.add_argument(
        "--embeddings-no-normalize",
        type=Path,
        default=None,
        help="Embeddings .npy extracted with --no-normalize for comparison.",
    )
    parser.add_argument("--meta", type=Path, default=None, help="Metadata CSV with sample_id order.")
    parser.add_argument("--labels", type=Path, default=None, help="(Deprecated) CSV containing labels.")
    parser.add_argument("--id-column", type=str, default="sample_id", help="Column name for sample IDs.")
    parser.add_argument("--label-column", type=str, default="label", help="Column name for labels.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size for linear probe split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for linear probe split.")
    return parser.parse_args()


def _init_accumulator() -> Dict[str, float]:
    return {
        "count": 0.0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": math.inf,
        "max": -math.inf,
        "nans": 0.0,
        "infs": 0.0,
    }


def _update_accumulator(acc: Dict[str, float], tensor: torch.Tensor) -> None:
    data = tensor.detach().float().cpu()
    acc["count"] += float(data.numel())
    acc["sum"] += float(data.sum().item())
    acc["sumsq"] += float(torch.square(data).sum().item())
    acc["min"] = min(acc["min"], float(data.min().item()))
    acc["max"] = max(acc["max"], float(data.max().item()))
    acc["nans"] += float(torch.isnan(data).sum().item())
    acc["infs"] += float(torch.isinf(data).sum().item())


def _finalize_stats(shape: Tuple[int, ...], acc: Dict[str, float]) -> TensorStats:
    count = int(acc["count"])
    if count == 0:
        return TensorStats(shape=shape, count=0, mean=0.0, std=0.0, min=0.0, max=0.0, nans=0, infs=0)
    mean = acc["sum"] / acc["count"]
    var = max(acc["sumsq"] / acc["count"] - mean**2, 0.0)
    std = math.sqrt(var)
    return TensorStats(
        shape=shape,
        count=count,
        mean=float(mean),
        std=float(std),
        min=float(acc["min"]),
        max=float(acc["max"]),
        nans=int(acc["nans"]),
        infs=int(acc["infs"]),
    )


def compute_tensor_stats(tensor: torch.Tensor) -> TensorStats:
    acc = _init_accumulator()
    _update_accumulator(acc, tensor)
    return _finalize_stats(tuple(tensor.shape), acc)


def collect_mesh_input_stats(
    data_root: Path,
    mesh_ext: Optional[List[str]],
    only_repaired_maps: bool,
    sample_limit: int,
    seed: int,
    feature_mode: str,
    device: str,
) -> Dict[str, object]:
    extensions = tuple(ext.lower() for ext in (mesh_ext or ["ply", "stl", "obj"]))
    mesh_paths = list_meshes(data_root, extensions, only_repaired_maps)
    if not mesh_paths:
        raise FileNotFoundError(f"No meshes found under {data_root} with extensions {extensions}.")

    total_meshes = len(mesh_paths)
    rng = np.random.default_rng(seed)
    if sample_limit > 0 and len(mesh_paths) > sample_limit:
        mesh_paths = [mesh_paths[i] for i in rng.choice(len(mesh_paths), size=sample_limit, replace=False)]

    device_obj = torch.device(device)
    per_mesh: List[Dict[str, object]] = []
    feats_acc = _init_accumulator()
    centers_acc = _init_accumulator()

    for mesh_path in mesh_paths:
        faces, feats, centers, _, _ = build_meshmae_inputs(
            mesh_path,
            device=device_obj,
            feature_mode=feature_mode,
        )
        feats_stats = compute_tensor_stats(feats)
        centers_stats = compute_tensor_stats(centers)
        _update_accumulator(feats_acc, feats)
        _update_accumulator(centers_acc, centers)
        per_mesh.append(
            {
                "mesh_path": str(mesh_path),
                "faces_shape": list(faces.shape),
                "feats": feats_stats.__dict__,
                "centers": centers_stats.__dict__,
            }
        )

    overall = {
        "feats": _finalize_stats(tuple(per_mesh[0]["feats"]["shape"]), feats_acc).__dict__,
        "centers": _finalize_stats(tuple(per_mesh[0]["centers"]["shape"]), centers_acc).__dict__,
    }
    return {
        "mesh_count": total_meshes,
        "sampled_meshes": len(per_mesh),
        "feature_mode": feature_mode,
        "per_mesh": per_mesh,
        "overall": overall,
    }


def load_probe_labels(
    meta_path: Path,
    id_column: str,
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    meta_df = pd.read_csv(meta_path)
    if id_column not in meta_df.columns:
        raise ValueError(f"Metadata CSV must include '{id_column}'.")
    if label_column not in meta_df.columns:
        raise ValueError(f"Metadata CSV must include '{label_column}'.")

    sample_ids = meta_df[id_column].astype(str).to_numpy()
    labels = meta_df[label_column].to_numpy()
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels.astype(str))
    return sample_ids, encoded


def run_linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> Dict[str, float]:
    if np.unique(labels).size < 2:
        raise ValueError("Linear probe requires at least 2 unique classes.")
    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    classifier = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto")
    classifier.fit(x_train, y_train)
    accuracy = float(classifier.score(x_test, y_test))
    return {"accuracy": accuracy}


def maybe_run_linear_probe(args: argparse.Namespace) -> Optional[Dict[str, object]]:
    if args.embeddings is None:
        return None
    if args.meta is None:
        raise ValueError("--meta is required when running linear probe diagnostics.")
    if args.labels is not None:
        logging.warning("--labels is deprecated and ignored; labels are read from --meta only.")

    embeddings = np.load(args.embeddings)
    _, labels = load_probe_labels(args.meta, args.id_column, args.label_column)
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Embeddings count ({embeddings.shape[0]}) does not match label count ({labels.shape[0]})."
        )
    result = {
        "embeddings": str(args.embeddings),
        "metrics": run_linear_probe(embeddings, labels, args.test_size, args.random_state),
    }
    if args.embeddings_no_normalize is not None:
        raw_embeddings = np.load(args.embeddings_no_normalize)
        if raw_embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                "No-normalize embeddings count "
                f"({raw_embeddings.shape[0]}) does not match label count ({labels.shape[0]})."
            )
        result["no_normalize"] = {
            "embeddings": str(args.embeddings_no_normalize),
            "metrics": run_linear_probe(raw_embeddings, labels, args.test_size, args.random_state),
        }
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output: Dict[str, object] = {"inputs": None, "linear_probe": None}

    if args.data is not None:
        logging.info("Collecting MeshMAE input stats from %s", args.data)
        output["inputs"] = collect_mesh_input_stats(
            data_root=args.data,
            mesh_ext=args.mesh_ext,
            only_repaired_maps=args.only_repaired_maps,
            sample_limit=args.sample_limit,
            seed=args.seed,
            feature_mode=args.feature_mode,
            device=args.device,
        )

    probe_result = maybe_run_linear_probe(args)
    if probe_result is not None:
        output["linear_probe"] = probe_result

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        logging.info("Saved diagnostics to %s", args.out_json)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
