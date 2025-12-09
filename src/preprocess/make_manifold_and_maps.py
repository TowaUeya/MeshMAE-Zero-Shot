"""Utilities to convert raw fossil meshes into MeshMAE-ready assets.

The script performs the following steps for each mesh inside the input directory:
1. Loads .ply/.stl meshes using `trimesh`.
2. Repairs non-manifold artifacts (merging duplicate vertices, fixing normals).
3. Simplifies the mesh to the requested number of faces (default 500).
4. Saves the processed mesh into the output directory, keeping the folder structure.
5. (Optional) launches an external MAPS generation script for hierarchical subdivision.

The MAPS generation is intentionally kept external. MeshMAE-compatible MAPS can be
produced with the official SubdivNet `datagen_maps.py` script. When `--make-maps` is
set, provide `--maps-script` pointing to the SubdivNet executable (the script is
auto-detected from a sibling `../SubdivNet/datagen_maps.py` when available). Any
additional arguments for the MAPS script can be passed via `--maps-extra-args`.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import trimesh


SUPPORTED_EXTENSIONS = {".ply", ".stl", ".obj"}


@dataclass
class MeshProcessRecord:
    """Metadata for a processed mesh."""

    source: str
    destination: str
    original_faces: int
    processed_faces: int
    scale: float
    maps_generated: bool
    notes: str = ""


def iter_mesh_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: Optional[int]) -> trimesh.Trimesh:
    """Simplify a mesh to approximately the given face count."""

    if target_faces is None:
        return mesh

    target_faces_int = int(target_faces)
    if target_faces_int <= 0:
        return mesh

    current_faces = len(mesh.faces)
    if current_faces <= target_faces_int:
        return mesh

    if hasattr(mesh, "simplify_quadric_decimation"):
        try:
            # Use face_count keyword; positional args are treated as percent (0-1).
            simplified = mesh.simplify_quadric_decimation(face_count=target_faces_int)
            return simplified
        except Exception as exc:
            logging.warning(
                "simplify_quadric_decimation failed; falling back to clustering: %s", exc
            )

    if hasattr(mesh, "simplify_vertex_clustering"):
        bbox_extent = mesh.bounding_box.extents
        # Approximate voxel size to end near the requested face count while
        # staying conservative to avoid degenerate output.
        max_extent = float(bbox_extent.max()) or 1.0
        voxel_size = max(target_faces_int, 1)
        voxel_size = max_extent / voxel_size ** (1 / 3)
        simplified = mesh.simplify_vertex_clustering(voxel_size=voxel_size)
        return simplified

    # Fallback: perform iterative midpoint decimation by clustering vertices.
    logging.warning("Quadratic decimation unavailable; using voxel downsampling fallback.")
    bbox_extent = mesh.bounding_box.extents
    max_extent = float(bbox_extent.max()) or 1.0
    voxel_size = (max_extent / max(target_faces_int, 1)) ** (1 / 3)
    simplified = mesh.voxelized(voxel_size).as_boxes()
    return simplified


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Run a tolerant set of repair ops across trimesh versions."""

    # Newer trimesh versions dropped some instance helpers; prefer methods when
    # available and fall back to trimesh.repair functions otherwise.
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(trimesh.repair, "remove_duplicate_faces"):
        trimesh.repair.remove_duplicate_faces(mesh)

    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()
    elif hasattr(trimesh.repair, "remove_unreferenced_vertices"):
        trimesh.repair.remove_unreferenced_vertices(mesh)

    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(trimesh.repair, "remove_degenerate_faces"):
        trimesh.repair.remove_degenerate_faces(mesh)

    if hasattr(mesh, "fill_holes"):
        mesh.fill_holes()
    elif hasattr(trimesh.repair, "fill_holes"):
        trimesh.repair.fill_holes(mesh)

    try:
        mesh.process(validate=True)
    except IndexError as exc:
        # Some trimesh versions can raise IndexError inside fix_normals when
        # winding repair encounters inconsistent adjacency. Retry with a less
        # strict pass to keep processing moving.
        logging.warning("trimesh.process(validate=True) failed (%s); retrying with validate=False", exc)
        mesh.process(validate=False)
    mesh.process(validate=True)
    return mesh


def run_maps_generation(script: Path, mesh_path: Path, output_dir: Path, extra_args: List[str]) -> bool:
    # Allow passing either the MAPS executable directly or a Python interpreter with the
    # actual MAPS script in `extra_args`. Place `extra_args` before mesh/output so the
    # invocation works for both styles:
    #   - maps_script = datagen_maps.py, extra_args = []
    #       => datagen_maps.py <mesh> <output>
    #   - maps_script = python, extra_args = [datagen_maps.py]
    #       => python datagen_maps.py <mesh> <output>
    command = [str(script)] + list(extra_args) + [str(mesh_path), str(output_dir)]
    logging.info("Running MAPS command: %s", " ".join(command))
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        logging.error("MAPS generation failed for %s: %s", mesh_path.name, completed.stderr)
        return False
    logging.debug("MAPS generation stdout for %s: %s", mesh_path.name, completed.stdout)
    return True


def process_mesh(
    source_path: Path,
    source_root: Path,
    destination_root: Path,
    target_faces: int,
    generate_maps: bool,
    maps_script: Optional[Path],
    maps_extra_args: List[str],
) -> MeshProcessRecord:
    logging.info("Processing %s", source_path)
    mesh = trimesh.load_mesh(source_path, process=False)
    original_faces = len(mesh.faces)
    mesh = repair_mesh(mesh)
    mesh = simplify_mesh(mesh, target_faces)

    relative_path = source_path.relative_to(source_root)
    destination_path = destination_root / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination_path)

    maps_generated = False
    notes = ""
    if generate_maps:
        if maps_script is None:
            notes = "--make-maps was requested but no script was provided."
            logging.warning(notes)
        else:
            maps_output_dir = destination_path.parent / f"{destination_path.stem}_maps"
            maps_output_dir.mkdir(parents=True, exist_ok=True)
            maps_generated = run_maps_generation(maps_script, destination_path, maps_output_dir, maps_extra_args)
            if not maps_generated:
                notes = "MAPS command failed; see logs."

    scale = float(np.linalg.norm(mesh.bounding_box.extents))
    return MeshProcessRecord(
        source=str(source_path),
        destination=str(destination_path),
        original_faces=original_faces,
        processed_faces=len(mesh.faces),
        scale=scale,
        maps_generated=maps_generated,
        notes=notes,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fossil meshes for MeshMAE pipelines")
    parser.add_argument("--in", dest="input_dir", required=True, type=Path, help="Directory with raw meshes")
    parser.add_argument("--out", dest="output_dir", required=True, type=Path, help="Directory to save processed meshes")
    parser.add_argument("--target_faces", type=int, default=500, help="Target number of faces after simplification")
    parser.add_argument("--make_maps", action="store_true", help="Run external MAPS generation tool")
    parser.add_argument(
        "--maps_script",
        type=Path,
        default=None,
        help=(
            "Path to the SubdivNet datagen_maps.py executable. When omitted, the script "
            "tries ../SubdivNet/datagen_maps.py automatically."
        ),
    )
    parser.add_argument(
        "--maps_extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Additional arguments passed to MAPS script. Use `--maps_extra_args -- "
            "<args...>` to forward flags verbatim; this option should be the last one "
            "on the command line."
        ),
    )
    parser.add_argument("--metadata", type=Path, default=None, help="Optional path to save processing metadata JSON")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Explicitly allow a standalone `--` after --maps_extra_args by accepting unknown
    # tokens and attaching them to maps_extra_args when present. Also tolerate
    # invocations where the separator itself ends up in "unknown" by checking for it
    # explicitly.
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        seen_separator = "--" in unknown or "--" in args.maps_extra_args
        if args.maps_extra_args or seen_separator:
            args.maps_extra_args.extend(arg for arg in unknown if arg != "--")
            unknown = []
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    target_faces: int = args.target_faces
    generate_maps: bool = args.make_maps
    maps_script: Optional[Path] = args.maps_script
    maps_extra_args: List[str] = [arg for arg in args.maps_extra_args if arg != "--"]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if generate_maps:
        if maps_script is None:
            candidate = Path(__file__).resolve().parents[2] / "SubdivNet" / "datagen_maps.py"
            if candidate.exists():
                maps_script = candidate
                logging.info("Auto-detected SubdivNet MAPS script at %s", candidate)
            else:
                raise FileNotFoundError(
                    "MAPS generation requested but no --maps_script was provided and "
                    "../SubdivNet/datagen_maps.py was not found. Clone https://github.com/"
                    "lzhengning/SubdivNet next to this repository or pass --maps_script."
                )
        elif not maps_script.exists():
            raise FileNotFoundError(f"Provided MAPS script does not exist: {maps_script}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[MeshProcessRecord] = []

    for mesh_file in iter_mesh_files(input_dir):
        record = process_mesh(
            mesh_file,
            input_dir,
            output_dir,
            target_faces,
            generate_maps,
            maps_script,
            maps_extra_args,
        )
        records.append(record)

    metadata_path = args.metadata or (output_dir / "processing_metadata.json")
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_faces": target_faces,
        "records": [asdict(record) for record in records],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
