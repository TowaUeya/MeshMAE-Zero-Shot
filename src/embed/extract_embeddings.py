"""Extract embeddings from MeshMAE encoders or geometry-derived fallbacks.

The script intentionally supports two modes:
1. **MeshMAE-driven** (recommended): requires the official MeshMAE repository to be
   installed as a Python package. Users must provide the dotted path to a factory
   function or class that returns the encoder (e.g. `meshmae.models_mae.mae_vit_base_patch16`
   or `model.meshmae.Mesh_mae`).
   The mesh preprocessing follows MeshMAE assumptions (500 faces + MAPS hierarchy).
2. **Geometry fallback**: if MeshMAE is not available, the script computes handcrafted
   descriptors (surface area, volume, PCA spectrum, bounding box) so that the
   downstream clustering pipeline can still be tested end-to-end.

The fallback is useful for CI and lightweight smoke tests; real experiments should
install MeshMAE and run in mode (1).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

import trimesh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .meshmae_inputs import build_meshmae_inputs


@dataclass
class ExtractionConfig:
    dataroot: Path
    checkpoint: Optional[Path]
    mesh_extension: Tuple[str, ...]
    only_repaired_maps: bool
    pool_strategy: str
    batch_size: int
    device: str
    num_workers: int
    normalize_embeddings: bool
    embedding_path: Path
    metadata_path: Path
    pca_embedding_path: Optional[Path]
    umap_embedding_path: Optional[Path]
    model_factory: Optional[str]
    model_factory_kwargs: Dict[str, object]
    force_geometry: bool
    expected_embedding_dim: Optional[int]
    min_unique_ratio: float
    min_feature_std: float


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def load_yaml_config(path: Optional[Path]) -> Dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(args: argparse.Namespace) -> ExtractionConfig:
    config_dict = load_yaml_config(args.config)
    input_cfg = config_dict.get("input", {})
    encoder_cfg = config_dict.get("encoder", {})
    output_cfg = config_dict.get("output", {})

    dataroot = Path(args.data or input_cfg.get("dataroot", "./datasets/fossils_maps"))
    checkpoint = Path(args.ckpt) if args.ckpt else (
        Path(input_cfg["checkpoint"]) if input_cfg.get("checkpoint") else None
    )
    mesh_extension = tuple(args.mesh_ext or input_cfg.get("mesh_extension", ["ply", "stl"]))
    only_repaired_maps = args.only_repaired_maps or input_cfg.get("only_repaired_maps", False)
    pool_strategy = args.pool or encoder_cfg.get("pool_strategy", "mean")
    batch_size = args.batch_size or encoder_cfg.get("batch_size", 16)
    device = args.device or encoder_cfg.get("device", "cuda")
    num_workers = args.num_workers or encoder_cfg.get("num_workers", 8)
    normalize_embeddings = encoder_cfg.get("normalize_embeddings", True)
    embedding_path = Path(args.out or output_cfg.get("embedding_path", "./embeddings/raw_embeddings.npy"))
    metadata_path = Path(args.meta or output_cfg.get("metadata_path", "./embeddings/meta.csv"))
    pca_embedding_path = (
        Path(args.pca_out)
        if args.pca_out
        else (Path(output_cfg["pca_embedding_path"]) if output_cfg.get("pca_embedding_path") else None)
    )
    umap_embedding_path = (
        Path(args.umap_out)
        if args.umap_out
        else (Path(output_cfg["umap_embedding_path"]) if output_cfg.get("umap_embedding_path") else None)
    )
    model_factory = args.model_factory or encoder_cfg.get("model_factory")
    model_factory_kwargs = encoder_cfg.get("model_factory_kwargs", {})
    force_geometry = args.force_geometry
    expected_embedding_dim = encoder_cfg.get("embedding_dim")
    min_unique_ratio = float(encoder_cfg.get("min_unique_ratio", 0.1))
    min_feature_std = float(encoder_cfg.get("min_feature_std", 1e-3))

    return ExtractionConfig(
        dataroot=dataroot,
        checkpoint=checkpoint,
        mesh_extension=tuple(ext.lower() for ext in mesh_extension),
        only_repaired_maps=only_repaired_maps,
        pool_strategy=pool_strategy,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        normalize_embeddings=normalize_embeddings,
        embedding_path=embedding_path,
        metadata_path=metadata_path,
        pca_embedding_path=pca_embedding_path,
        umap_embedding_path=umap_embedding_path,
        model_factory=model_factory,
        model_factory_kwargs=model_factory_kwargs,
        force_geometry=force_geometry,
        expected_embedding_dim=expected_embedding_dim,
        min_unique_ratio=min_unique_ratio,
        min_feature_std=min_feature_std,
    )


def _read_maps_output_path(maps_dir: Path, extensions: Tuple[str, ...]) -> Optional[Path]:
    metadata_path = maps_dir / "maps_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        output_path = metadata.get("output_path_relative") or metadata.get("output_path")
        if output_path:
            candidate = Path(output_path)
            if not candidate.is_absolute():
                candidate = maps_dir / candidate
            if candidate.exists():
                return candidate

    for ext in extensions:
        matches = sorted(maps_dir.glob(f"*_MAPS*.{ext.lstrip('.')}"))
        if matches:
            return matches[0]

    for ext in extensions:
        matches = sorted(maps_dir.glob(f"*.{ext.lstrip('.')}"))
        if matches:
            return matches[0]
    return None


def _metadata_mesh_path(mesh_path: Path) -> str:
    if mesh_path.parent.name.endswith("_maps") and mesh_path.parent.parent.name == "success":
        return str(mesh_path.parent)
    return str(mesh_path)


def list_meshes(root: Path, extensions: Tuple[str, ...], only_repaired_maps: bool) -> List[Path]:
    maps_root = root / "success"
    if maps_root.exists():
        map_files = []
        for maps_dir in sorted(p for p in maps_root.rglob("*_maps") if p.is_dir()):
            output_path = _read_maps_output_path(maps_dir, extensions)
            if output_path is not None:
                map_files.append(output_path)
        if map_files:
            files = sorted(map_files)
        else:
            files = []
    else:
        files = []

    if not files:
        gathered = set()
        for ext in extensions:
            gathered.update(root.rglob(f"*.{ext.lstrip('.')}"))
        files = sorted(gathered)

    if only_repaired_maps:
        files = [p for p in files if p.name.endswith("_repaired_MAPS.obj")]
    return files


def compute_geometry_descriptor(mesh: trimesh.Trimesh) -> np.ndarray:
    mesh = mesh.copy()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()

    bbox = mesh.bounding_box.extents
    volume = mesh.volume if mesh.is_volume else 0.0
    surface_area = mesh.area
    inertia = mesh.moment_inertia
    eigenvalues, _ = np.linalg.eigh(inertia)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    try:
        curvature = mesh.curvature_principal
        curvature_stats = np.array([
            np.mean(curvature[0]),
            np.mean(curvature[1]),
            np.std(curvature[0]),
            np.std(curvature[1]),
        ])
    except BaseException:  # noqa: BLE001
        curvature_stats = np.zeros(4, dtype=np.float32)
    descriptor = np.concatenate([
        bbox,
        np.array([volume, surface_area]),
        eigenvalues[:3],
        curvature_stats,
    ])
    return descriptor.astype(np.float32)


def geometry_embedding_pipeline(mesh_paths: Iterable[Path]) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    embeddings: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []
    for mesh_path in mesh_paths:
        mesh = trimesh.load_mesh(mesh_path, process=False)
        vector = compute_geometry_descriptor(mesh)
        embeddings.append(vector)
        metadata.append({"sample_id": mesh_path.stem, "mesh_path": _metadata_mesh_path(mesh_path)})
    if not embeddings:
        raise RuntimeError("No embeddings were generated. Check input directory.")
    return np.vstack(embeddings), metadata


def instantiate_model(factory_path: str, factory_kwargs: Optional[Dict[str, object]] = None) -> torch.nn.Module:
    module_name, obj_name = factory_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, obj_name)
    factory_kwargs = factory_kwargs or {}
    if isinstance(target, torch.nn.Module):
        return target
    if isinstance(target, type) and issubclass(target, torch.nn.Module):
        signature = inspect.signature(target)
        required = [
            param
            for name, param in signature.parameters.items()
            if name != "self" and param.default is param.empty
        ]
        if required and not factory_kwargs:
            required_names = ", ".join(param.name for param in required)
            raise ValueError(
                "Model factory must be a callable that fully specifies MeshMAE initialization. "
                f"Class {factory_path} requires parameters ({required_names}). "
                "Provide a factory function or model_factory_kwargs in the YAML config."
            )
        return target(**factory_kwargs)
    if callable(target):
        return target(**factory_kwargs) if factory_kwargs else target()
    raise TypeError(f"{factory_path} is not a callable or torch.nn.Module.")


def meshmae_embedding_pipeline(
    mesh_paths: Iterable[Path],
    config: ExtractionConfig,
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    if config.model_factory is None:
        raise ValueError("A --model-factory must be provided when using MeshMAE mode.")
    if config.checkpoint is None:
        raise ValueError("A checkpoint path must be provided when using MeshMAE mode.")

    model = instantiate_model(config.model_factory, config.model_factory_kwargs)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    checkpoint = torch.load(config.checkpoint, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()
    mismatched: Dict[str, Tuple[torch.Size, torch.Size]] = {}
    for key, tensor in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != tensor.shape:
            mismatched[key] = (tensor.shape, model_state[key].shape)
            continue
    if mismatched:
        mismatch_details = ", ".join(
            f"{name} (ckpt={ckpt_shape}, model={model_shape})"
            for name, (ckpt_shape, model_shape) in mismatched.items()
        )
        raise RuntimeError(
            "Checkpoint tensors do not match model architecture. "
            f"Fix the model factory to align with the checkpoint. Mismatches: {mismatch_details}"
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint loading did not match the model state dict. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )

    embeddings: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []

    def call_with_mesh_inputs(fn: callable, label: str) -> torch.Tensor:
        signature = inspect.signature(fn)
        params = [
            p
            for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in signature.parameters.values())
        attempts = []
        vertex_names = {"vertices", "verts", "vert", "v", "x", "xyz"}
        face_names = {"faces", "face", "f", "triangles", "tri"}

        keyword_kwargs = {}
        for param in params:
            if param.name in vertex_names:
                keyword_kwargs[param.name] = vertices
            if param.name in face_names:
                keyword_kwargs[param.name] = faces
        if keyword_kwargs:
            attempts.append(("keyword", (), keyword_kwargs))

        if len(params) == 1 and params[0].name in {"mesh", "data", "input", "inputs"}:
            attempts.append(
                (
                    "mesh-dict",
                    (),
                    {params[0].name: {"vertices": vertices, "faces": faces}},
                )
            )
            attempts.append(("mesh-tuple", (), {params[0].name: (vertices, faces)}))

        if has_varargs or len(params) >= 2:
            attempts.append(("positional-2", (vertices, faces), {}))

        required = [p for p in params if p.default is p.empty]
        if len(required) > 2:
            extras = []
            for param in required[2:]:
                if param.name in {"mask_ratio", "masking_ratio"}:
                    extras.append(0.0)
                elif param.name in {"mask", "masked", "return_mask"}:
                    extras.append(False)
                else:
                    extras.append(None)
            attempts.append(("positional-fill", (vertices, faces, *extras), {}))

        if len(params) == 1:
            attempts.append(("positional-1", (vertices,), {}))
        if not params and not has_varargs:
            attempts.append(("positional-0", (), {}))

        last_error = None
        for attempt_label, args, kwargs in attempts:
            logging.debug(
                "Attempting MeshMAE inference using %s (%s) with signature %s",
                label,
                attempt_label,
                signature,
            )
            try:
                return fn(*args, **kwargs)
            except TypeError as exc:
                last_error = exc
                logging.debug("Skipping %s due to TypeError: %s", label, exc)
                continue

        logging.debug(
            "Falling back to (vertices, faces) call for %s with signature %s",
            label,
            signature,
        )
        try:
            return fn(vertices, faces)
        except TypeError as exc:
            last_error = exc
        if last_error is not None:
            raise last_error
        raise TypeError(f"No compatible call signature found for {label}.")

    def call_with_meshmae_inputs(fn: callable, label: str, mesh_inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        signature = inspect.signature(fn)
        params = [
            p
            for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        faces_input, feats_input, centers_input, Fs_input, cordinates_input = mesh_inputs
        keyword_kwargs = {}
        for param in params:
            if param.name == "faces":
                keyword_kwargs[param.name] = faces_input
            elif param.name == "feats":
                keyword_kwargs[param.name] = feats_input
            elif param.name == "centers":
                keyword_kwargs[param.name] = centers_input
            elif param.name in {"Fs", "fs"}:
                keyword_kwargs[param.name] = Fs_input
            elif param.name in {"cordinates", "coordinates"}:
                keyword_kwargs[param.name] = cordinates_input
            elif param.default is param.empty:
                if param.name in {"mask_ratio", "masking_ratio"}:
                    keyword_kwargs[param.name] = 0.0
                elif param.name in {"mask", "masked", "return_mask"}:
                    keyword_kwargs[param.name] = False
                else:
                    keyword_kwargs[param.name] = None
        attempts = []
        if keyword_kwargs:
            attempts.append(("keyword", (), keyword_kwargs))
        if len(params) >= 5:
            attempts.append(("positional-5", mesh_inputs, {}))
        required = [p for p in params if p.default is p.empty]
        if len(required) > 5:
            extras = []
            for param in required[5:]:
                if param.name in {"mask_ratio", "masking_ratio"}:
                    extras.append(0.0)
                elif param.name in {"mask", "masked", "return_mask"}:
                    extras.append(False)
                else:
                    extras.append(None)
            attempts.append(("positional-fill", (*mesh_inputs, *extras), {}))
        if len(params) >= 1:
            attempts.append(("positional-1", (faces_input,), {}))

        last_error = None
        for attempt_label, args, kwargs in attempts:
            logging.debug(
                "Attempting MeshMAE inference using %s (%s) with signature %s",
                label,
                attempt_label,
                signature,
            )
            try:
                return fn(*args, **kwargs)
            except TypeError as exc:
                last_error = exc
                logging.debug("Skipping %s due to TypeError: %s", label, exc)
                continue

        if last_error is not None:
            raise last_error
        raise TypeError(f"No compatible MeshMAE input signature found for {label}.")

    def extract_output_tensor(output: object) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            tensors = [value for value in output.values() if isinstance(value, torch.Tensor)]
        elif isinstance(output, (tuple, list)):
            tensors = [value for value in output if isinstance(value, torch.Tensor)]
        else:
            tensors = []
        if not tensors:
            return None
        return max(tensors, key=lambda tensor: tensor.numel())

    def extract_embedding(output: object) -> Optional[torch.Tensor]:
        latent = extract_output_tensor(output)
        if latent is None:
            return None
        if latent.dim() > 2:
            embedding = latent[:, 0, :] if config.pool_strategy == "cls" else latent.mean(dim=1)
        else:
            embedding = latent
        if embedding.numel() < 8:
            return None
        return embedding

    def select_encoder_output(model: torch.nn.Module, mesh_inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        candidates = []
        if hasattr(model, "forward_encoder"):
            candidates.append(("forward_encoder", model.forward_encoder))
        if hasattr(model, "encode"):
            candidates.append(("encode", model.encode))
        if hasattr(model, "forward_features"):
            candidates.append(("forward_features", model.forward_features))
        if hasattr(model, "forward"):
            candidates.append(("forward", model.forward))
        if hasattr(model, "encoder"):
            encoder = getattr(model, "encoder")
            if hasattr(encoder, "forward_encoder"):
                candidates.append(("encoder.forward_encoder", encoder.forward_encoder))
            if hasattr(encoder, "encode"):
                candidates.append(("encoder.encode", encoder.encode))
            if hasattr(encoder, "forward_features"):
                candidates.append(("encoder.forward_features", encoder.forward_features))
            if hasattr(encoder, "forward"):
                candidates.append(("encoder.forward", encoder.forward))

        last_shape: Optional[torch.Size] = None
        for label, fn in candidates:
            logging.debug("Attempting MeshMAE inference using %s", label)
            try:
                output = call_with_meshmae_inputs(fn, label, mesh_inputs)
            except TypeError as exc:
                logging.debug("Skipping %s due to TypeError: %s", label, exc)
            else:
                embedding = extract_embedding(output)
                if embedding is not None:
                    return embedding
                last_shape = output.shape if isinstance(output, torch.Tensor) else None
                logging.debug("Skipping %s due to invalid embedding output shape.", label)
            try:
                output = call_with_mesh_inputs(fn, label)
            except TypeError as exc:
                logging.debug("Skipping %s due to TypeError: %s", label, exc)
                continue
            embedding = extract_embedding(output)
            if embedding is not None:
                return embedding
            last_shape = output.shape if isinstance(output, torch.Tensor) else None
            logging.debug("Skipping %s due to invalid embedding output shape.", label)

        shape_message = f" Last output shape: {last_shape}." if last_shape is not None else ""
        raise AttributeError(
            "Provided model does not expose a compatible encoder output that yields embeddings."
            f"{shape_message}"
        )

    for mesh_path in mesh_paths:
        mesh = trimesh.load_mesh(mesh_path, process=True)
        mesh_inputs = build_meshmae_inputs(mesh, device)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        with torch.no_grad():
            embedding = select_encoder_output(model, mesh_inputs)
        embedding_np = embedding.detach().cpu().numpy().astype(np.float32)
        embedding_vec = embedding_np.reshape(-1)
        embeddings.append(embedding_vec)
        metadata.append({"sample_id": mesh_path.stem, "mesh_path": _metadata_mesh_path(mesh_path)})

    stacked = np.vstack(embeddings)
    return stacked, metadata


def ensure_non_degenerate_embeddings(embeddings: np.ndarray, label: str, config: ExtractionConfig) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"{label} embeddings must be a 2D array, got shape {embeddings.shape}.")
    if embeddings.shape[0] == 0:
        raise ValueError(f"{label} embeddings have zero samples.")
    if embeddings.shape[1] == 0:
        raise ValueError(f"{label} embeddings have zero feature dimensions.")
    if config.expected_embedding_dim is not None and embeddings.shape[1] != config.expected_embedding_dim:
        raise ValueError(
            f"{label} embeddings have dimension {embeddings.shape[1]}, expected {config.expected_embedding_dim}."
        )
    if np.allclose(embeddings, 0):
        raise ValueError(f"{label} embeddings are all zeros. Check MeshMAE/geometry feature extraction.")
    if np.allclose(embeddings, embeddings[0]):
        raise ValueError(f"{label} embeddings are identical across samples. Check MeshMAE/geometry feature extraction.")
    rounded = np.round(embeddings, decimals=4)
    unique_rows = np.unique(rounded, axis=0).shape[0]
    unique_ratio = unique_rows / embeddings.shape[0]
    if unique_ratio < config.min_unique_ratio:
        raise ValueError(
            f"{label} embeddings are nearly identical across samples "
            f"(unique ratio={unique_ratio:.3f} < {config.min_unique_ratio:.3f})."
        )
    feature_std = np.std(embeddings, axis=0)
    mean_feature_std = float(np.mean(feature_std))
    if mean_feature_std < config.min_feature_std:
        raise ValueError(
            f"{label} embeddings have collapsed variance "
            f"(mean feature std={mean_feature_std:.6f} < {config.min_feature_std:.6f})."
        )


def maybe_normalize(embeddings: np.ndarray, normalize: bool) -> np.ndarray:
    if not normalize:
        return embeddings
    scaler = StandardScaler()
    normalized = scaler.fit_transform(embeddings)
    return normalized


def maybe_compute_pca(embeddings: np.ndarray, output_path: Optional[Path]) -> Optional[np.ndarray]:
    if output_path is None:
        return None
    pca = PCA(n_components=min(embeddings.shape[0], embeddings.shape[1], 256))
    reduced = pca.fit_transform(embeddings)
    np.save(output_path, reduced)
    return reduced


def maybe_compute_umap(embeddings: np.ndarray, output_path: Optional[Path]) -> Optional[np.ndarray]:
    if output_path is None:
        return None
    from umap import UMAP

    umap = UMAP(n_neighbors=25, min_dist=0.1, metric="cosine", random_state=42)
    reduced = umap.fit_transform(embeddings)
    np.save(output_path, reduced)
    return reduced


def write_outputs(
    embeddings: np.ndarray,
    metadata: List[Dict[str, str]],
    config: ExtractionConfig,
) -> None:
    config.embedding_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.embedding_path, embeddings)
    df = pd.DataFrame(metadata)
    df.to_csv(config.metadata_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MeshMAE or geometry embeddings")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config to override CLI flags")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--data", type=str, default=None, help="Directory with processed meshes")
    parser.add_argument("--mesh-ext", nargs="*", default=None, help="Mesh extensions to include")
    parser.add_argument(
        "--only-repaired-maps",
        action="store_true",
        help="Only keep meshes ending with _repaired_MAPS.obj",
    )
    parser.add_argument("--pool", type=str, default=None, choices=["mean", "cls"], help="Pooling strategy")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--device", type=str, default=None, help="Device identifier (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=None, help="Data loader workers (MeshMAE mode)")
    parser.add_argument("--normalize", action="store_true", help="Normalize embeddings with StandardScaler")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization explicitly")
    parser.add_argument("--out", type=str, default=None, help="Path to save raw embeddings (.npy)")
    parser.add_argument("--meta", type=str, default=None, help="Path to save metadata (.csv)")
    parser.add_argument("--pca-out", type=str, default=None, help="Optional PCA output path (.npy)")
    parser.add_argument("--umap-out", type=str, default=None, help="Optional UMAP output path (.npy)")
    parser.add_argument(
        "--model-factory",
        type=str,
        default=None,
        help="Dotted path to a MeshMAE model factory or class (e.g. meshmae.models_mae.mae_vit_base_patch16)",
    )
    parser.add_argument("--force-geometry", action="store_true", help="Force handcrafted geometry descriptors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.normalize and args.no_normalize:
        raise ValueError("--normalize and --no-normalize are mutually exclusive")

    config = resolve_config(args)
    if args.normalize:
        config.normalize_embeddings = True
    if args.no_normalize:
        config.normalize_embeddings = False

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Embedding normalization (StandardScaler) enabled: %s", config.normalize_embeddings)
    mesh_paths = list_meshes(config.dataroot, config.mesh_extension, config.only_repaired_maps)
    if not mesh_paths:
        raise FileNotFoundError(f"No meshes with extensions {config.mesh_extension} found in {config.dataroot}")

    logging.info("Found %d meshes for embedding extraction", len(mesh_paths))

    embeddings: np.ndarray
    metadata: List[Dict[str, str]]

    meshmae_detected = module_available("meshmae")
    model_factory_provided = config.model_factory is not None

    if not config.force_geometry and model_factory_provided:
        logging.info("Model factory provided. Attempting encoder-based feature extraction.")
        embeddings, metadata = meshmae_embedding_pipeline(mesh_paths, config)
    else:
        if not config.force_geometry and not model_factory_provided:
            logging.warning(
                "No --model-factory provided. Falling back to geometry descriptors."
            )
        elif not meshmae_detected and not model_factory_provided:
            logging.warning(
                "MeshMAE package not detected. Falling back to geometry descriptors."
            )
        else:
            logging.info("Using geometry descriptors as requested.")
        embeddings, metadata = geometry_embedding_pipeline(mesh_paths)

    embeddings = maybe_normalize(embeddings, config.normalize_embeddings)
    ensure_non_degenerate_embeddings(embeddings, "Extracted", config)
    write_outputs(embeddings, metadata, config)
    logging.info("Saved embeddings to %s", config.embedding_path)
    logging.info("Saved metadata to %s", config.metadata_path)

    maybe_compute_pca(embeddings, config.pca_embedding_path)
    maybe_compute_umap(embeddings, config.umap_embedding_path)


if __name__ == "__main__":
    main()
