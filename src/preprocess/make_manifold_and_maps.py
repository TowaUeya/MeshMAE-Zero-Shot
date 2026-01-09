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
    repaired_faces: int
    repaired_vertices: int
    repaired_is_watertight: bool
    repaired_is_winding_consistent: bool
    repaired_non_manifold_edges: int
    repair_mode: str
    scale: float
    maps_generated: bool
    maps_metadata_path: Optional[str] = None
    maps_cleaned_input_path: Optional[str] = None
    maps_failed_mesh_path: Optional[str] = None
    maps_output_path: Optional[str] = None
    notes: str = ""


@dataclass
class RepairReport:
    faces: int
    vertices: int
    is_watertight: bool
    is_winding_consistent: bool
    non_manifold_edges: int
    mode: str
    attempts: int
    notes: str = ""


@dataclass
class MapsRunResult:
    success: bool
    output_dir: Path
    metadata_path: Optional[Path]


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


def _count_non_manifold_edges(mesh: trimesh.Trimesh) -> int:
    if mesh.faces.size == 0:
        return 0
    faces = mesh.faces
    edges = np.sort(
        np.vstack(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ]
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts > 2))


def _iterative_fill_holes(mesh: trimesh.Trimesh, max_iters: int = 1) -> None:
    if max_iters <= 0:
        return
    if hasattr(mesh, "fill_holes"):
        filler = mesh.fill_holes
    elif hasattr(trimesh.repair, "fill_holes"):
        filler = lambda: trimesh.repair.fill_holes(mesh)
    else:
        return

    previous_faces = len(mesh.faces)
    for _ in range(max_iters):
        try:
            filler()
        except Exception as exc:
            logging.debug("fill_holes failed during repair: %s", exc)
            break
        current_faces = len(mesh.faces)
        if current_faces == previous_faces:
            break
        previous_faces = current_faces


def _append_error_log(log_dir: Path, lines: List[str]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    error_log = log_dir / "error.log"
    timestamp = datetime.utcnow().isoformat()
    with error_log.open("a", encoding="utf-8") as f:
        for idx, line in enumerate(lines):
            prefix = f"[{timestamp}] " if idx == 0 else " " * (len(timestamp) + 3)
            f.write(prefix + line.rstrip() + "\n")
        f.write("\n")
    return error_log


def _mesh_stats(mesh: trimesh.Trimesh, min_face_area: float) -> dict:
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
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "min_face_area": float(min_area),
        "degenerate_faces": int(degenerate),
        "components": int(components),
    }


def _write_maps_failure_metadata(
    target_maps_dir: Path,
    reasons: List[str],
    repair_report: RepairReport,
    mesh: trimesh.Trimesh,
    clean_min_face_area: float,
) -> None:
    payload = {
        "success": False,
        "output_path": str(target_maps_dir),
        "output_path_relative": target_maps_dir.name,
        "error": "; ".join(reasons),
        "failure_reasons": reasons,
        "mesh_stats": _mesh_stats(mesh, min_face_area=clean_min_face_area),
        "repair": asdict(repair_report),
    }
    target_maps_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_maps_dir / "maps_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _quality_snapshot(mesh: trimesh.Trimesh) -> dict:
    return {
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "non_manifold_edges": _count_non_manifold_edges(mesh),
    }


def repair_mesh(mesh: trimesh.Trimesh, aggressive: bool = False) -> Tuple[trimesh.Trimesh, RepairReport]:
    """Run a tolerant set of repair ops across trimesh versions.

    Parameters
    ----------
    mesh:
        Input mesh to be repaired (mutated in-place).
    aggressive:
        Whether to immediately enable stronger cleanup. When False, a
        lightweight pass is attempted first and escalated if quality checks
        fail.
    """

    def _remove_duplicate_faces(target: trimesh.Trimesh) -> None:
        if hasattr(target, "remove_duplicate_faces"):
            target.remove_duplicate_faces()
        elif hasattr(trimesh.repair, "remove_duplicate_faces"):
            trimesh.repair.remove_duplicate_faces(target)

    def _remove_unreferenced_vertices(target: trimesh.Trimesh) -> None:
        if hasattr(target, "remove_unreferenced_vertices"):
            target.remove_unreferenced_vertices()
        elif hasattr(trimesh.repair, "remove_unreferenced_vertices"):
            trimesh.repair.remove_unreferenced_vertices(target)

    def _remove_degenerate_faces(target: trimesh.Trimesh) -> None:
        if hasattr(target, "remove_degenerate_faces"):
            target.remove_degenerate_faces()
        elif hasattr(trimesh.repair, "remove_degenerate_faces"):
            trimesh.repair.remove_degenerate_faces(target)

    def _process_safe(target: trimesh.Trimesh) -> None:
        try:
            target.process(validate=True)
        except IndexError as exc:
            logging.warning(
                "trimesh.process(validate=True) failed (%s); retrying with validate=False", exc
            )
            target.process(validate=False)
        target.process(validate=True)

    attempts = 0

    def _run_pass(mode: str, aggressive_mode: bool) -> RepairReport:
        nonlocal attempts, mesh
        note_log: List[str] = []
        attempts += 1
        _remove_duplicate_faces(mesh)
        _remove_degenerate_faces(mesh)
        _remove_unreferenced_vertices(mesh)
        _iterative_fill_holes(mesh, max_iters=3 if aggressive_mode else 1)

        if aggressive_mode:
            if hasattr(trimesh.repair, "fix_inversion"):
                try:
                    trimesh.repair.fix_inversion(mesh)
                except Exception as exc:
                    logging.debug("fix_inversion failed: %s", exc)
            if hasattr(trimesh.repair, "fix_winding"):
                try:
                    trimesh.repair.fix_winding(mesh)
                except Exception as exc:
                    logging.debug("fix_winding failed: %s", exc)
            if hasattr(trimesh.repair, "fix_normals"):
                try:
                    trimesh.repair.fix_normals(mesh)
                except Exception as exc:
                    logging.debug("fix_normals failed: %s", exc)

            if mesh.is_watertight is False or mesh.is_winding_consistent is False:
                try:
                    components = mesh.split(only_watertight=False)
                except Exception as exc:
                    logging.debug("mesh.split failed during repair: %s", exc)
                    components = []
                if len(components) > 1:
                    watertight_components = [c for c in components if c.is_watertight]
                    selected = max(watertight_components or components, key=lambda c: len(c.faces))
                    if selected is not mesh:
                        note_log.append(
                            f"Selected component with {len(selected.faces)} faces after split "
                            f"(watertight={selected.is_watertight})."
                        )
                        mesh = selected
                    _remove_unreferenced_vertices(mesh)

        _process_safe(mesh)
        quality = _quality_snapshot(mesh)
        needs_extra_pass = aggressive_mode and (
            not quality["is_watertight"]
            or not quality["is_winding_consistent"]
            or quality["non_manifold_edges"] > 0
        )
        if needs_extra_pass:
            _iterative_fill_holes(mesh, max_iters=2 if aggressive_mode else 1)
            _process_safe(mesh)
            quality = _quality_snapshot(mesh)
        return RepairReport(**quality, mode=mode, attempts=attempts, notes="\n".join(note_log))

    initial_mode = "aggressive" if aggressive else "standard"
    report = _run_pass(initial_mode, aggressive_mode=aggressive)
    if (not report.is_watertight or not report.is_winding_consistent or report.non_manifold_edges > 0) and not aggressive:
        logging.info(
            "Initial repair left issues (watertight=%s, winding=%s, non_manifold_edges=%s); retrying with aggressive pass",
            report.is_watertight,
            report.is_winding_consistent,
            report.non_manifold_edges,
        )
        report = _run_pass("aggressive_retry", aggressive_mode=True)
    return mesh, report


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
    clean_input: bool = False,
    clean_min_face_area: float = 0.0,
    skip_failed_maps: bool = False,
    dump_failed_cases: Optional[Path] = None,
) -> MapsRunResult:
    def _tail(text: str, limit: int = 40) -> str:
        lines = text.splitlines()
        if len(lines) > limit:
            lines = lines[-limit:]
        return "\n".join(lines)

    subdivnet_root = subdivnet_root.resolve()
    mesh_path = mesh_path.resolve()
    output_dir = output_dir.resolve()
    output_file = output_dir / f"{mesh_path.stem}_MAPS{output_suffix or mesh_path.suffix}"
    metadata_path = output_dir / "maps_metadata.json"

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
        "--metadata",
        str(metadata_path),
        "--clean-min-face-area",
        str(clean_min_face_area),
    ]
    if max_base_size is not None:
        command.extend(["--max_base_size", str(max_base_size)])
    command.extend(passthrough)
    if clean_input:
        command.append("--clean-input")
    if skip_failed_maps:
        command.append("--skip-failed-maps")
    if dump_failed_cases is not None:
        command.extend(["--dump-failed-cases", str(dump_failed_cases)])
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    pythonpath_entries = [str(subdivnet_root)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    logging.info("Running MAPS command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        check=False,
#       capture_output=True,
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
        error_log_dir.mkdir(parents=True, exist_ok=True)
        _append_error_log(
            error_log_dir,
            [
                "MAPS command failed.",
                "Command: " + " ".join(command),
                f"Return code: {completed.returncode}",
                f"Output path: {output_file}",
                "--- stdout ---",
                completed.stdout or "<empty>",
                "--- stderr ---",
                completed.stderr or "<empty>",
            ],
        )
        return MapsRunResult(success=False, output_dir=output_dir, metadata_path=metadata_path if metadata_path.exists() else None)

    if stdout_tail:
        logging.debug("MAPS generation stdout for %s: %s", mesh_path.name, stdout_tail)
    if stderr_tail:
        logging.debug("MAPS generation stderr for %s: %s", mesh_path.name, stderr_tail)
    return MapsRunResult(success=True, output_dir=output_dir, metadata_path=metadata_path if metadata_path.exists() else None)


def process_mesh(
    source_path: Path,
    source_root: Path,
    destination_root: Path,
    target_faces: int,
    generate_maps: bool,
    subdivnet_root: Optional[Path],
    maps_extra_args: List[str],
    aggressive_repair: bool,
    clean_maps_input: bool,
    clean_maps_min_face_area: float,
    skip_failed_maps: bool,
    dump_failed_cases: Optional[Path],
) -> MeshProcessRecord:
    logging.info("Processing %s", source_path)
    mesh = trimesh.load_mesh(source_path, process=False)
    original_faces = len(mesh.faces)
    mesh, repair_report = repair_mesh(mesh, aggressive=aggressive_repair)
    relative_path = source_path.relative_to(source_root)

    maps_generated = False
    maps_output_path: Optional[str] = None
    maps_metadata_path: Optional[str] = None
    maps_cleaned_input_path: Optional[str] = None
    maps_failed_mesh_path: Optional[str] = None
    notes_parts = [f"Repair mode={repair_report.mode}; attempts={repair_report.attempts}."]
    repair_summary_lines = [
        "Repair checkpoint before MAPS:",
        f"- Faces after repair: {repair_report.faces}",
        f"- Vertices after repair: {repair_report.vertices}",
        f"- Watertight: {repair_report.is_watertight}",
        f"- Winding consistent: {repair_report.is_winding_consistent}",
        f"- Non-manifold edges: {repair_report.non_manifold_edges}",
    ]
    if repair_report.notes:
        repair_summary_lines.append("Repair notes: " + repair_report.notes.replace("\n", "; "))

    if generate_maps:
        maps_relative_dir = relative_path.parent / f"{source_path.stem}_maps"
        success_maps_dir = destination_root / "success" / maps_relative_dir
        failed_maps_dir = destination_root / "failed" / maps_relative_dir
        maps_failure_reasons: List[str] = []
        if len(mesh.faces) == 0:
            maps_failure_reasons.append("MAPS skipped: mesh has no faces after repair.")
        if repair_report.is_watertight is False:
            maps_failure_reasons.append("MAPS skipped: mesh is not watertight after repair.")
        if repair_report.is_winding_consistent is False:
            maps_failure_reasons.append("MAPS skipped: mesh has inconsistent winding after repair.")
        if repair_report.non_manifold_edges > 0:
            maps_failure_reasons.append(
                f"MAPS skipped: {repair_report.non_manifold_edges} non-manifold edges remain after repair."
            )

        if maps_failure_reasons:
            target_maps_dir = failed_maps_dir
            if target_maps_dir.exists():
                shutil.rmtree(target_maps_dir)
            _append_error_log(target_maps_dir, maps_failure_reasons + repair_summary_lines)
            _write_maps_failure_metadata(
                target_maps_dir,
                maps_failure_reasons,
                repair_report,
                mesh,
                clean_maps_min_face_area,
            )
            maps_metadata_path = str((target_maps_dir / "maps_metadata.json").resolve())
            if dump_failed_cases is not None:
                dump_failed_cases.mkdir(parents=True, exist_ok=True)
                failed_mesh_path = dump_failed_cases / f"{source_path.stem}_precheck_failed{source_path.suffix or '.ply'}"
                report_path = dump_failed_cases / f"{source_path.stem}_precheck_failed.json"
                mesh.export(failed_mesh_path)
                with report_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "source": str(source_path),
                            "failure_reasons": maps_failure_reasons,
                            "mesh_stats": _mesh_stats(mesh, min_face_area=clean_maps_min_face_area),
                            "repair": asdict(repair_report),
                        },
                        handle,
                        indent=2,
                    )
            maps_output_path = str(target_maps_dir)
            notes_parts.extend(maps_failure_reasons)
            maps_generated = False
        elif subdivnet_root is None:
            warning_note = "--make_maps was requested but no SubdivNet root was provided."
            logging.warning(warning_note)
            notes_parts.append(warning_note)
        elif not maps_failure_reasons and subdivnet_root is not None:
            if failed_maps_dir.exists():
                shutil.rmtree(failed_maps_dir)
            if success_maps_dir.exists():
                shutil.rmtree(success_maps_dir)
            with tempfile.TemporaryDirectory(prefix=f".{source_path.stem}.") as temp_dir:
                temp_repaired_path = Path(temp_dir) / f"{source_path.stem}_repaired{source_path.suffix or '.ply'}"
                mesh.export(temp_repaired_path)
                maps_output_dir = Path(temp_dir) / "maps_output"
                if maps_output_dir.exists():
                    shutil.rmtree(maps_output_dir)
                maps_result = run_maps_generation(
                    subdivnet_root,
                    temp_repaired_path,
                    maps_output_dir,
                    maps_extra_args,
                    output_suffix=source_path.suffix or ".ply",
                    clean_input=clean_maps_input,
                    clean_min_face_area=clean_maps_min_face_area,
                    skip_failed_maps=skip_failed_maps,
                    dump_failed_cases=dump_failed_cases,
                )
                maps_generated = maps_result.success
                target_maps_dir = success_maps_dir if maps_generated else failed_maps_dir
                target_maps_dir.parent.mkdir(parents=True, exist_ok=True)
                if target_maps_dir.exists():
                    shutil.rmtree(target_maps_dir)
                if maps_output_dir.exists():
                    shutil.move(str(maps_output_dir), target_maps_dir)
                maps_output_path = str(target_maps_dir)
                if maps_generated:
                    notes_parts.append(f"MAPS stored at {target_maps_dir}.")
                else:
                    _append_error_log(target_maps_dir, ["MAPS command failed."] + repair_summary_lines)
                    error_log_path = target_maps_dir / "error.log"
                    notes_parts.append(f"MAPS command failed; see {error_log_path} for details.")
                    if not skip_failed_maps:
                        raise RuntimeError(f"MAPS generation failed for {source_path}")
                metadata_path = target_maps_dir / "maps_metadata.json"
                maps_metadata_path = str(metadata_path) if metadata_path.exists() else None
                maps_cleaned_input_path = None
                maps_failed_mesh_path = None
                if metadata_path.exists():
                    with metadata_path.open("r", encoding="utf-8") as meta_handle:
                        maps_metadata = json.load(meta_handle)
                    if maps_metadata.get("output_path_relative"):
                        maps_metadata["output_path"] = str((target_maps_dir / maps_metadata["output_path_relative"]).resolve())
                    if maps_metadata.get("cleaned_input_relative"):
                        maps_cleaned_input_path = str((target_maps_dir / maps_metadata["cleaned_input_relative"]).resolve())
                        maps_metadata["cleaned_input_path"] = maps_cleaned_input_path
                    if maps_metadata.get("failed_mesh_relative"):
                        maps_failed_mesh_path = str((target_maps_dir / maps_metadata["failed_mesh_relative"]).resolve())
                        maps_metadata["failed_mesh_path"] = maps_failed_mesh_path
                    with metadata_path.open("w", encoding="utf-8") as meta_handle:
                        json.dump(maps_metadata, meta_handle, indent=2)
                    cleaning_info = maps_metadata.get("cleaning")
                    if cleaning_info:
                        removed_faces = cleaning_info.get("removed_small_or_zero_faces", 0) + cleaning_info.get(
                            "removed_duplicate_faces", 0
                        )
                        notes_parts.append(
                            f"MAPS cleaning: faces {cleaning_info.get('original_faces')} -> {cleaning_info.get('cleaned_faces')} "
                            f"(removed {removed_faces}); vertices {cleaning_info.get('original_vertices')} "
                            f"-> {cleaning_info.get('cleaned_vertices')} (removed {cleaning_info.get('removed_duplicate_vertices', 0)})."
                        )
                    if maps_failed_mesh_path and not maps_generated:
                        notes_parts.append(f"MAPS failed; cleaned input stored at {maps_failed_mesh_path}.")
    notes = "; ".join(notes_parts)

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
        repaired_faces=repair_report.faces,
        repaired_vertices=repair_report.vertices,
        repaired_is_watertight=repair_report.is_watertight,
        repaired_is_winding_consistent=repair_report.is_winding_consistent,
        repaired_non_manifold_edges=repair_report.non_manifold_edges,
        repair_mode=repair_report.mode,
        scale=scale,
        maps_generated=maps_generated,
        maps_metadata_path=maps_metadata_path if generate_maps else None,
        maps_cleaned_input_path=maps_cleaned_input_path if generate_maps else None,
        maps_failed_mesh_path=maps_failed_mesh_path if generate_maps else None,
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
        "--clean_maps_input",
        action="store_true",
        help=(
            "Clean meshes before running MAPS (remove duplicate vertices/faces, drop tiny faces, "
            "recompute normals). Cleaning happens after repair and before MAPS."
        ),
    )
    parser.add_argument(
        "--clean_maps_min_face_area",
        type=float,
        default=0.0,
        help="Minimum face area threshold when cleaning MAPS inputs.",
    )
    parser.add_argument(
        "--skip_failed_maps",
        action="store_true",
        help="Continue the pipeline even when MAPS fails for a mesh (default behavior).",
    )
    parser.add_argument(
        "--fail_on_maps_error",
        action="store_true",
        help="Stop the pipeline when MAPS fails for a mesh.",
    )
    parser.add_argument(
        "--dump_failed_cases",
        type=Path,
        default=None,
        help="Optional directory to dump failed MAPS inputs and reports.",
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
    parser.add_argument(
        "--aggressive-repair",
        action="store_true",
        help=(
            "Enable stronger mesh repair (multiple hole-filling passes, inversion/winding fixes, "
            "component splitting). Recommended when MAPS generation fails due to self-intersections "
            "or non-manifold edges."
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
    aggressive_repair: bool = args.aggressive_repair
    clean_maps_input: bool = args.clean_maps_input
    clean_maps_min_face_area: float = args.clean_maps_min_face_area
    skip_failed_maps: bool = args.skip_failed_maps or not args.fail_on_maps_error
    dump_failed_cases: Optional[Path] = args.dump_failed_cases

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
                        aggressive_repair,
                        clean_maps_input,
                        clean_maps_min_face_area,
                        skip_failed_maps,
                        dump_failed_cases,
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
                aggressive_repair,
                clean_maps_input,
                clean_maps_min_face_area,
                skip_failed_maps,
                dump_failed_cases,
            )
            records.append(record)

    metadata_path = args.metadata or (output_dir / "processing_metadata.json")
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_faces": target_faces,
        "skip_failed_maps": skip_failed_maps,
        "records": [asdict(record) for record in records],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
