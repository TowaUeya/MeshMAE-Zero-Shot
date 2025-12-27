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
    force_geometry: bool


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
    force_geometry = args.force_geometry

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
        force_geometry=force_geometry,
    )


def list_meshes(root: Path, extensions: Tuple[str, ...], only_repaired_maps: bool) -> List[Path]:
    files = set()
    for ext in extensions:
        files.update(root.rglob(f"*.{ext.lstrip('.')}"))
    files = sorted(files)
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
        mesh = trimesh.load_mesh(mesh_path, process=True)
        vector = compute_geometry_descriptor(mesh)
        embeddings.append(vector)
        metadata.append({"sample_id": mesh_path.stem, "mesh_path": str(mesh_path)})
    if not embeddings:
        raise RuntimeError("No embeddings were generated. Check input directory.")
    return np.vstack(embeddings), metadata


def instantiate_model(factory_path: str) -> torch.nn.Module:
    module_name, obj_name = factory_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, obj_name)
    if isinstance(target, torch.nn.Module):
        return target
    if isinstance(target, type) and issubclass(target, torch.nn.Module):
        return target()
    if callable(target):
        return target()
    raise TypeError(f"{factory_path} is not a callable or torch.nn.Module.")


def meshmae_embedding_pipeline(
    mesh_paths: Iterable[Path],
    config: ExtractionConfig,
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    if config.model_factory is None:
        raise ValueError("A --model-factory must be provided when using MeshMAE mode.")
    if config.checkpoint is None:
        raise ValueError("A checkpoint path must be provided when using MeshMAE mode.")

    model = instantiate_model(config.model_factory)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    checkpoint = torch.load(config.checkpoint, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()
    filtered_state: Dict[str, torch.Tensor] = {}
    mismatched: Dict[str, Tuple[torch.Size, torch.Size]] = {}
    for key, tensor in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != tensor.shape:
            mismatched[key] = (tensor.shape, model_state[key].shape)
            continue
        filtered_state[key] = tensor
    if mismatched:
        logging.warning(
            "Skipping %d checkpoint tensors due to shape mismatch: %s",
            len(mismatched),
            ", ".join(
                f"{name} (ckpt={ckpt_shape}, model={model_shape})"
                for name, (ckpt_shape, model_shape) in mismatched.items()
            ),
        )
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        logging.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    embeddings: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []

    for mesh_path in mesh_paths:
        mesh = trimesh.load_mesh(mesh_path, process=True)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        with torch.no_grad():
            if hasattr(model, "forward_encoder"):
                output = model.forward_encoder(vertices, faces)
            elif hasattr(model, "encode"):
                output = model.encode(vertices, faces)
            else:
                raise AttributeError("Provided model does not expose forward_encoder/encode methods.")
        if isinstance(output, tuple):
            latent = output[0]
        else:
            latent = output
        if latent.dim() > 2:
            if config.pool_strategy == "cls":
                embedding = latent[:, 0, :]
            else:
                embedding = latent.mean(dim=1)
        else:
            embedding = latent
        embedding_np = embedding.detach().cpu().numpy().astype(np.float32)
        embeddings.append(embedding_np.squeeze())
        metadata.append({"sample_id": mesh_path.stem, "mesh_path": str(mesh_path)})

    stacked = np.vstack(embeddings)
    return stacked, metadata


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
    write_outputs(embeddings, metadata, config)
    logging.info("Saved embeddings to %s", config.embedding_path)
    logging.info("Saved metadata to %s", config.metadata_path)

    maybe_compute_pca(embeddings, config.pca_embedding_path)
    maybe_compute_umap(embeddings, config.umap_embedding_path)


if __name__ == "__main__":
    main()
