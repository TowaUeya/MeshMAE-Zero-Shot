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
import sys
import traceback
from pathlib import Path
from typing import Optional

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
    return parser.parse_args(argv)


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
) -> Path:
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
        "error": None,
    }

    mesh = trimesh.load_mesh(input_path, process=False)
    metadata_payload["input_faces"] = len(mesh.faces)
    metadata_payload["input_vertices"] = len(mesh.vertices)

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
        raise ValueError(metadata_payload["error"])

    attempted_sizes: list[int] = []
    last_error: Optional[Exception] = None

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

    def _write_metadata():
        if metadata is None:
            return
        metadata_path = metadata.resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata_payload, handle, indent=2)

    for candidate_size in _candidate_base_sizes():
        attempted_sizes.append(candidate_size)
        metadata_payload["attempted_base_sizes"] = attempted_sizes.copy()
        try:
            maps = maps_cls(
                mesh.vertices,
                mesh.faces,
                base_size=candidate_size,
                verbose=verbose,
            )
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
            metadata_payload["error"] = str(exc)
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
    _write_metadata()
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
    )


if __name__ == "__main__":
    main()
