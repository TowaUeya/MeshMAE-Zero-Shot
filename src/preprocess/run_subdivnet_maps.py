"""Safe wrapper around SubdivNet MAPS generation for a single mesh.

This module keeps the SubdivNet dependency isolated from the main
MeshMAE preprocessing flow by importing `datagen_maps.py` inside a
separate Python process. It avoids triggering the demo entrypoints in
`datagen_maps.py` and passes absolute paths for both input and output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import trimesh
from src.preprocess.clean_mesh import clean_mesh


def resolve_subdivnet(subdivnet_root: Path):
    subdivnet_root = subdivnet_root.resolve()
    if not subdivnet_root.exists():
        raise FileNotFoundError(f"SubdivNet root does not exist: {subdivnet_root}")
    if str(subdivnet_root) not in sys.path:
        sys.path.insert(0, str(subdivnet_root))

    try:
        datagen_maps = importlib.import_module("datagen_maps")
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time failure
        raise ModuleNotFoundError(
            "Could not import datagen_maps from SubdivNet. Ensure --subdivnet_root"
            " points to the repository root containing datagen_maps.py"
        ) from exc

    try:
        maps_module = importlib.import_module("maps")
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time failure
        raise ModuleNotFoundError(
            "Could not import the SubdivNet 'maps' package. Install SubdivNet's"
            " dependencies (e.g., triangle, sortedcollections) and ensure the"
            " repository root is on PYTHONPATH."
        ) from exc

    if not hasattr(maps_module, "MAPS"):
        raise ImportError("SubdivNet maps.MAPS class is unavailable; cannot generate MAPS")

    return datagen_maps, maps_module.MAPS


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SubdivNet MAPS generation for a single mesh")
    parser.add_argument("--subdivnet_root", required=True, type=Path, help="Path to the SubdivNet repository root")
    parser.add_argument("--input", required=True, type=Path, help="Input mesh file (absolute path recommended)")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory to place MAPS outputs")
    parser.add_argument("--output-path", type=Path, default=None, help="Exact MAPS output mesh path (with extension)")
    parser.add_argument("--base_size", type=int, default=96, help="Base size passed to MAPS")
    parser.add_argument("--depth", type=int, default=3, help="Subdivision depth")
    parser.add_argument("--max_base_size", type=int, default=None, help="Abort if computed base_size exceeds this value")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose MAPS logging")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path to write MAPS run metadata as JSON",
    )
    parser.add_argument(
        "--clean-input",
        action="store_true",
        help="Clean the mesh before running MAPS (remove duplicate vertices/faces, drop small faces, recompute normals)",
    )
    parser.add_argument(
        "--clean-min-face-area",
        type=float,
        default=0.0,
        help="Minimum area threshold when removing tiny faces during cleaning",
    )
    parser.add_argument(
        "--skip-failed-maps",
        action="store_true",
        help="Do not raise on MAPS failures; write metadata and exit successfully.",
    )
    parser.add_argument(
        "--dump-failed-cases",
        type=Path,
        default=None,
        help="Optional directory to dump failed inputs and simple reports.",
    )
    return parser.parse_args(argv)


def _compute_mesh_stats(mesh: trimesh.Trimesh, min_face_area: float) -> dict:
    faces = np.array(mesh.faces)
    vertices = np.array(mesh.vertices)
    try:
        areas = mesh.area_faces
        min_area = float(np.min(areas)) if len(areas) else 0.0
        degenerate = int(np.count_nonzero(areas <= max(min_face_area, 0.0)))
    except Exception:
        min_area = 0.0
        degenerate = 0
    try:
        components = len(mesh.split(only_watertight=False))
    except Exception:
        components = 1
    return {
        "faces": int(len(faces)),
        "vertices": int(len(vertices)),
        "min_face_area": float(min_area),
        "degenerate_faces": int(degenerate),
        "components": int(components),
    }


def _build_local_adjacency(faces: np.ndarray, vertex_count: int) -> list[dict[int, set[int]]]:
    local_adjacency: list[dict[int, set[int]]] = [
        {} for _ in range(vertex_count)
    ]
    for tri in faces:
        if len(tri) != 3:
            continue
        a, b, c = tri
        local_adjacency[a].setdefault(b, set()).add(c)
        local_adjacency[a].setdefault(c, set()).add(b)
        local_adjacency[b].setdefault(a, set()).add(c)
        local_adjacency[b].setdefault(c, set()).add(a)
        local_adjacency[c].setdefault(a, set()).add(b)
        local_adjacency[c].setdefault(b, set()).add(a)
    return local_adjacency


def _topology_check_fast(mesh: trimesh.Trimesh) -> dict:
    """Check if local vertex neighborhoods form cycles.

    SubdivNet's MAPS code expects subdivision connectivity. If the induced
    neighbor graph around a vertex has no cycle, ``cycle_basis`` returns an
    empty list (hence ``[0]`` would fail). We treat this as a likely MAPS failure.
    """

    faces = np.array(mesh.faces)
    vertex_count = len(mesh.vertices)
    local_adjacency = _build_local_adjacency(faces, vertex_count)
    failed_vertices = []
    reasons = []

    for vid, adj in enumerate(local_adjacency):
        if len(adj) < 3:
            failed_vertices.append(vid)
            reasons.append("valence<3")
            continue

        degrees = {node: len(neighbors) for node, neighbors in adj.items()}
        if any(degree != 2 for degree in degrees.values()):
            failed_vertices.append(vid)
            reasons.append("neighbor_degree!=2")
            continue

        nodes = list(adj.keys())
        visited = set()
        stack = [nodes[0]]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adj[node] - visited)
        if len(visited) != len(nodes):
            failed_vertices.append(vid)
            reasons.append("neighbor_disconnected")
            continue

        edge_count = sum(degrees.values()) // 2
        if edge_count != len(nodes):
            failed_vertices.append(vid)
            reasons.append("edge_count_mismatch")

    return {
        "checked_vertices": int(vertex_count),
        "failed_vertices": int(len(failed_vertices)),
        "failed_vertex_indices": failed_vertices[:10],
        "failure_reasons_sample": reasons[:10],
    }


def _dump_failed_case(
    mesh: trimesh.Trimesh,
    input_path: Path,
    dump_dir: Path,
    report: dict,
) -> dict:
    dump_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{input_path.stem}_failed"
    mesh_path = dump_dir / f"{stem}.ply"
    report_path = dump_dir / f"{stem}.json"
    mesh.export(mesh_path)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return {"mesh_path": str(mesh_path), "report_path": str(report_path)}


def run_maps(
    subdivnet_root: Path,
    input_path: Path,
    out_dir: Path,
    output_path: Optional[Path],
    base_size: int,
    depth: int,
    max_base_size: Optional[int],
    verbose: bool,
    metadata: Optional[Path] = None,
    clean_input: bool = False,
    clean_min_face_area: float = 0.0,
    skip_failed_maps: bool = False,
    dump_failed_cases: Optional[Path] = None,
) -> Optional[Path]:
    datagen_maps, maps_cls = resolve_subdivnet(subdivnet_root)

    input_path = input_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = out_dir / f"{input_path.stem}_MAPS{input_path.suffix}"
    else:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_payload = {
        "input_faces": None,
        "input_vertices": None,
        "attempted_base_sizes": [],
        "attempt_errors": [],
        "chosen_base_size": None,
        "actual_base_size": None,
        "success": False,
        "output_path": str(output_path),
        "output_path_relative": output_path.name,
        "cleaning": None,
        "cleaned_input_path": None,
        "cleaned_input_relative": None,
        "failed_mesh_path": None,
        "failed_mesh_relative": None,
        "mesh_stats": None,
        "topology_check": None,
        "failure": None,
        "error": None,
    }

    def _numpy_to_builtin(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _write_metadata():
        if metadata is None:
            return
        metadata_path = metadata.resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata_payload, handle, indent=2, default=_numpy_to_builtin)

    mesh = trimesh.load_mesh(input_path, process=False)
    metadata_payload["input_faces"] = len(mesh.faces)
    metadata_payload["input_vertices"] = len(mesh.vertices)
    metadata_payload["mesh_stats"] = _compute_mesh_stats(mesh, min_face_area=clean_min_face_area)

    if clean_input:
        try:
            cleaning_report = clean_mesh(mesh, min_face_area=clean_min_face_area)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            metadata_payload["error"] = f"Cleaning failed: {exc}"
            _write_metadata()
            raise
        metadata_payload["cleaning"] = cleaning_report.to_dict()
        cleaned_path = out_dir / f"{input_path.stem}_cleaned{input_path.suffix}"
        mesh.export(cleaned_path)
        metadata_payload["cleaned_input_path"] = str(cleaned_path.resolve())
        metadata_payload["cleaned_input_relative"] = cleaned_path.name
        metadata_payload["mesh_stats"] = _compute_mesh_stats(mesh, min_face_area=clean_min_face_area)
        logging.info(
            "Cleaned MAPS input %s: faces %s -> %s, vertices %s -> %s (removed %s faces, %s vertices)",
            input_path.name,
            cleaning_report.original_faces,
            cleaning_report.cleaned_faces,
            cleaning_report.original_vertices,
            cleaning_report.cleaned_vertices,
            cleaning_report.removed_small_or_zero_faces + cleaning_report.removed_duplicate_faces,
            cleaning_report.removed_duplicate_vertices,
        )
    face_count = len(mesh.faces)
    if face_count == 0:
        metadata_payload["error"] = f"Mesh {input_path} has no faces; cannot generate MAPS"
        metadata_payload["failed_mesh_path"] = metadata_payload["cleaned_input_path"] or str(input_path)
        metadata_payload["failed_mesh_relative"] = metadata_payload["cleaned_input_relative"]
        if metadata_payload["failed_mesh_relative"] is None:
            metadata_payload["failed_mesh_relative"] = input_path.name
        _write_metadata()
        if dump_failed_cases is not None:
            _dump_failed_case(mesh, input_path, dump_failed_cases, metadata_payload)
        if skip_failed_maps:
            return None
        raise ValueError(metadata_payload["error"])

    attempted_sizes: list[int] = []
    last_error: Optional[Exception] = None
    topology_needed = False

    def _candidate_base_sizes() -> list[int]:
        """Return a descending list of base sizes, always reaching 4 faces."""

        sizes: list[int] = []
        size = max(min(base_size, face_count), 4)

        while True:
            if size not in sizes:
                sizes.append(size)

            if size <= 4:
                break

            next_size = max(size // 2, 4)
            if next_size == size:
                break

            size = next_size

        return sizes

    candidate_sizes = _candidate_base_sizes()
    if verbose:
        logging.info("MAPS base_size candidates: %s", candidate_sizes)

    for candidate_size in candidate_sizes:
        if verbose:
            logging.info("MAPS attempt start: base_size=%s for %s", candidate_size, input_path.name)
        attempted_sizes.append(candidate_size)
        metadata_payload["attempted_base_sizes"] = attempted_sizes.copy()
        try:
            if verbose:
                logging.info("MAPS build start: base_size=%s for %s", candidate_size, input_path.name)
            maps = maps_cls(
                mesh.vertices,
                mesh.faces,
                base_size=candidate_size,
                verbose=verbose,
            )
            if verbose:
                logging.info("MAPS build success: base_size=%s for %s", candidate_size, input_path.name)
            actual_base_size = getattr(maps, "base_size", candidate_size)
            if max_base_size is not None and actual_base_size > max_base_size:
                raise ValueError(
                    f"Computed base_size {actual_base_size} exceeds max_base_size {max_base_size}"
                )
            sub_mesh = maps.mesh_upsampling(depth=depth)
            sub_mesh.export(output_path)
            metadata_payload["chosen_base_size"] = candidate_size
            metadata_payload["actual_base_size"] = actual_base_size
            metadata_payload["success"] = True
            metadata_payload["error"] = None
            _write_metadata()
            return output_path
        except Exception as exc:  # pragma: no cover - SubdivNet failures are external
            last_error = exc
            if isinstance(exc, IndexError):
                topology_needed = True
            logging.info(
                "MAPS attempt failed for base_size=%s on %s: %s",
                candidate_size,
                input_path.name,
                exc,
            )
            metadata_payload["error"] = str(exc)
            metadata_payload["attempt_errors"].append(
                {
                    "base_size": int(candidate_size),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            metadata_payload["failed_mesh_path"] = metadata_payload["cleaned_input_path"] or str(input_path)
            metadata_payload["failed_mesh_relative"] = metadata_payload["cleaned_input_relative"]
            if metadata_payload["failed_mesh_relative"] is None:
                metadata_payload["failed_mesh_relative"] = input_path.name
            print(
                f"MAPS failed for base_size={candidate_size} on {input_path}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            continue

    metadata_payload["error"] = str(last_error) if last_error is not None else "Unknown MAPS failure"
    metadata_payload["failure"] = {
        "type": type(last_error).__name__ if last_error is not None else "Unknown",
        "message": metadata_payload["error"],
    }
    if topology_needed:
        if verbose:
            logging.info("MAPS topology check start for %s", input_path.name)
        topology_report = _topology_check_fast(mesh)
        if verbose:
            logging.info("MAPS topology check complete for %s", input_path.name)
        metadata_payload["topology_check"] = topology_report
    _write_metadata()
    if dump_failed_cases is not None:
        _dump_failed_case(mesh, input_path, dump_failed_cases, metadata_payload)
    if skip_failed_maps:
        return None
    raise RuntimeError(
        "MAPS generation failed after trying base_size candidates "
        f"{attempted_sizes} for {input_path}"
    ) from last_error


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_maps(
        subdivnet_root=args.subdivnet_root,
        input_path=args.input,
        out_dir=args.out_dir,
        output_path=args.output_path,
        base_size=args.base_size,
        depth=args.depth,
        max_base_size=args.max_base_size,
        verbose=args.verbose,
        metadata=args.metadata,
        clean_input=args.clean_input,
        clean_min_face_area=args.clean_min_face_area,
        skip_failed_maps=args.skip_failed_maps,
        dump_failed_cases=args.dump_failed_cases,
    )


if __name__ == "__main__":
    main()
