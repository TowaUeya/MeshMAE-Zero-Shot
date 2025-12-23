"""Utilities to convert raw fossil meshes into MeshMAE-ready assets.

The script performs the following steps for each mesh inside the input directory:
1. Loads .ply/.stl meshes using `trimesh`.
2. Repairs non-manifold artifacts (merging duplicate vertices, fixing normals).
3. Optionally generates MAPS using SubdivNet *before* simplification to avoid
   topology changes that can break MAPS assumptions.
4. Simplifies the mesh to the requested number of faces (default 500).
5. Saves the processed mesh into the output directory, keeping the folder structure.

The MAPS generation is kept in a separate process via the
`src.preprocess.run_subdivnet_maps` wrapper to avoid triggering SubdivNet's demo
entrypoints. Pass `--subdivnet_root` to point at the SubdivNet repository and
forward MAPS arguments through `--maps_extra_args`.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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
    maps_output_path: Optional[str] = None
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
        logging.debug(
            "Skipping simplification: current_faces=%s is already <= target_faces=%s",
            current_faces,
            target_faces_int,
        )
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


def _parse_maps_args(extra_args: List[str]) -> Tuple[int, int, Optional[int], List[str]]:
    """Extract MAPS parameters while preserving passthrough flags."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base_size", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max_base_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    known, unknown = parser.parse_known_args(extra_args)
    passthrough = []
    for token in unknown:
        if token == "--":
            continue
        passthrough.append(token)

    return known.base_size, known.depth, known.max_base_size, (["--verbose"] if known.verbose else []) + passthrough


def run_maps_generation(
    subdivnet_root: Path,
    mesh_path: Path,
    output_dir: Path,
    extra_args: List[str],
    output_suffix: Optional[str] = None,
) -> bool:
    def _tail(text: str, limit: int = 40) -> str:
        lines = text.splitlines()
        if len(lines) > limit:
            lines = lines[-limit:]
        return "\n".join(lines)

    subdivnet_root = subdivnet_root.resolve()
    mesh_path = mesh_path.resolve()
    output_dir = output_dir.resolve()
    output_file = output_dir / f"{mesh_path.stem}_MAPS{output_suffix or mesh_path.suffix}"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_size, depth, max_base_size, passthrough = _parse_maps_args(extra_args)
    command = [
        sys.executable,
        "-m",
        "src.preprocess.run_subdivnet_maps",
        "--subdivnet_root",
        str(subdivnet_root),
        "--input",
        str(mesh_path),
        "--out-dir",
        str(output_dir),
        "--output-path",
        str(output_file),
        "--base_size",
        str(base_size),
        "--depth",
        str(depth),
    ]
    if max_base_size is not None:
        command.extend(["--max_base_size", str(max_base_size)])
    command.extend(passthrough)
    env = os.environ.copy()
    pythonpath_entries = [str(subdivnet_root)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    logging.info("Running MAPS command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout_tail = _tail(completed.stdout)
    stderr_tail = _tail(completed.stderr)
    success = completed.returncode == 0 and output_file.exists()

    if not success:
        logging.error(
            "MAPS generation failed for %s (return code %s, output_exists=%s).\nstdout:\n%s\nstderr:\n%s",
            mesh_path.name,
            completed.returncode,
            output_file.exists(),
            stdout_tail,
            stderr_tail,
        )
        error_log_dir = output_dir
        if error_log_dir.exists():
            shutil.rmtree(error_log_dir)
        error_log_dir.mkdir(parents=True, exist_ok=True)
        error_log = error_log_dir / "error.log"
        with error_log.open("w", encoding="utf-8") as f:
            f.write("Command: " + " ".join(command) + "\n")
            f.write(f"Return code: {completed.returncode}\n")
            f.write(f"Output path: {output_file}\n")
            f.write("\n--- stdout ---\n")
            f.write(completed.stdout or "<empty>\n")
            f.write("\n--- stderr ---\n")
            f.write(completed.stderr or "<empty>\n")

        # Clean up and avoid leaving empty *_maps directories.
        if not any(output_dir.iterdir()):
            shutil.rmtree(output_dir)
        return False

    if stdout_tail:
        logging.debug("MAPS generation stdout for %s: %s", mesh_path.name, stdout_tail)
    if stderr_tail:
        logging.debug("MAPS generation stderr for %s: %s", mesh_path.name, stderr_tail)
    return True


def process_mesh(
    source_path: Path,
    source_root: Path,
    destination_root: Path,
    target_faces: int,
    generate_maps: bool,
    subdivnet_root: Optional[Path],
    maps_extra_args: List[str],
) -> MeshProcessRecord:
    logging.info("Processing %s", source_path)
    mesh = trimesh.load_mesh(source_path, process=False)
    original_faces = len(mesh.faces)
    mesh = repair_mesh(mesh)
    relative_path = source_path.relative_to(source_root)

    maps_generated = False
    maps_output_path: Optional[str] = None
    notes = ""
    if generate_maps:
        if len(mesh.faces) == 0:
            notes = "MAPS skipped: mesh has no faces after repair."
        elif mesh.is_watertight is False:
            notes = "MAPS skipped: mesh is not watertight after repair."
        elif mesh.is_winding_consistent is False:
            notes = "MAPS skipped: mesh has inconsistent winding after repair."
        elif subdivnet_root is None:
            notes = "--make_maps was requested but no SubdivNet root was provided."
            logging.warning(notes)
        else:
            with tempfile.TemporaryDirectory(prefix=f".{source_path.stem}.") as temp_dir:
                temp_repaired_path = Path(temp_dir) / f"{source_path.stem}_repaired{source_path.suffix or '.ply'}"
                mesh.export(temp_repaired_path)
                maps_relative_dir = relative_path.parent / f"{source_path.stem}_maps"
                maps_output_dir = Path(temp_dir) / "maps_output"
                success_maps_dir = destination_root / "success" / maps_relative_dir
                failed_maps_dir = destination_root / "failed" / maps_relative_dir
                if maps_output_dir.exists():
                    shutil.rmtree(maps_output_dir)
                maps_generated = run_maps_generation(
                    subdivnet_root,
                    temp_repaired_path,
                    maps_output_dir,
                    maps_extra_args,
                    output_suffix=source_path.suffix or ".ply",
                )
                target_maps_dir = success_maps_dir if maps_generated else failed_maps_dir
                target_maps_dir.parent.mkdir(parents=True, exist_ok=True)
                if target_maps_dir.exists():
                    shutil.rmtree(target_maps_dir)
                if maps_output_dir.exists():
                    shutil.move(str(maps_output_dir), target_maps_dir)
                maps_output_path = str(target_maps_dir)
                if maps_generated:
                    notes = f"MAPS stored at {target_maps_dir}."
                else:
                    error_log_path = target_maps_dir / "error.log"
                    if not error_log_path.exists():
                        target_maps_dir.mkdir(parents=True, exist_ok=True)
                    notes = f"MAPS command failed; see {error_log_path} for details."

    mesh = simplify_mesh(mesh, target_faces)

    destination_path = destination_root / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination_path)

    scale = float(np.linalg.norm(mesh.bounding_box.extents))
    return MeshProcessRecord(
        source=str(source_path),
        destination=str(destination_path),
        original_faces=original_faces,
        processed_faces=len(mesh.faces),
        scale=scale,
        maps_generated=maps_generated,
        maps_output_path=maps_output_path,
        notes=notes,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fossil meshes for MeshMAE pipelines")
    parser.add_argument("--in", dest="input_dir", required=True, type=Path, help="Directory with raw meshes")
    parser.add_argument("--out", dest="output_dir", required=True, type=Path, help="Directory to save processed meshes")
    parser.add_argument("--target_faces", type=int, default=500, help="Target number of faces after simplification")
    parser.add_argument("--make_maps", action="store_true", help="Run external MAPS generation tool")
    parser.add_argument(
        "--subdivnet_root",
        type=Path,
        default=None,
        help=(
            "Path to the SubdivNet repository root. When omitted, ../SubdivNet is "
            "auto-detected if present."
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes for parallel mesh processing. "
            "Use 0 or 1 to run sequentially; values >1 enable parallel processing."
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
    subdivnet_root: Optional[Path] = args.subdivnet_root
    maps_extra_args: List[str] = [arg for arg in args.maps_extra_args if arg != "--"]
    num_workers: int = max(args.num_workers, 0)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if generate_maps:
        if subdivnet_root is None:
            candidate = Path(__file__).resolve().parents[2] / "SubdivNet"
            if (candidate / "datagen_maps.py").exists():
                subdivnet_root = candidate
                logging.info("Auto-detected SubdivNet repository at %s", candidate)
            else:
                raise FileNotFoundError(
                    "MAPS generation requested but no --subdivnet_root was provided and "
                    "../SubdivNet was not found. Clone https://github.com/lzhengning/SubdivNet "
                    "next to this repository or pass --subdivnet_root."
                )
        elif not subdivnet_root.exists():
            raise FileNotFoundError(f"Provided SubdivNet root does not exist: {subdivnet_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[MeshProcessRecord] = []

    mesh_files = list(iter_mesh_files(input_dir))
    if num_workers > 1:
        logging.info("Running in parallel with %s workers", num_workers)
        with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
            records = pool.starmap(
                process_mesh,
                [
                    (
                        mesh_file,
                        input_dir,
                        output_dir,
                        target_faces,
                        generate_maps,
                        subdivnet_root,
                        maps_extra_args,
                    )
                    for mesh_file in mesh_files
                ],
            )
    else:
        logging.info("Running sequentially (num_workers=%s)", num_workers)
        for mesh_file in mesh_files:
            record = process_mesh(
                mesh_file,
                input_dir,
                output_dir,
                target_faces,
                generate_maps,
                subdivnet_root,
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
