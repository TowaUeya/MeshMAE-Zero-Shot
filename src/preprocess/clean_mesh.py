"""Mesh cleaning utilities for MAPS preprocessing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import trimesh


@dataclass
class CleanReport:
    """Summary of mesh cleaning operations."""

    original_vertices: int
    original_faces: int
    cleaned_vertices: int
    cleaned_faces: int
    removed_duplicate_vertices: int
    removed_duplicate_faces: int
    removed_small_or_zero_faces: int
    removed_unreferenced_vertices: int
    removed_components: int
    components_before: int
    components_after: int
    min_face_area: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _safe_call(label: str, func) -> None:
    try:
        func()
    except Exception as exc:  # pragma: no cover - defensive logging only
        logging.debug("Skipping %s during cleaning: %s", label, exc)


def _remove_duplicate_vertices(mesh: trimesh.Trimesh, merge_tolerance: float) -> int:
    before = len(mesh.vertices)
    if hasattr(mesh, "remove_duplicate_vertices"):
        _safe_call("remove_duplicate_vertices", mesh.remove_duplicate_vertices)
    if hasattr(mesh, "merge_vertices"):
        _safe_call("merge_vertices", lambda: mesh.merge_vertices(epsilon=merge_tolerance))
    elif hasattr(trimesh.repair, "merge_vertices"):
        _safe_call(
            "trimesh.repair.merge_vertices",
            lambda: trimesh.repair.merge_vertices(mesh, merge_tex=True, merge_norm=True),
        )
    if len(mesh.vertices) == before:
        try:
            digits = max(int(abs(np.log10(merge_tolerance))), 0) if merge_tolerance > 0 else 12
        except ValueError:
            digits = 12
        try:
            rounded = np.round(mesh.vertices, decimals=digits)
            unique_vertices, inverse = np.unique(rounded, axis=0, return_inverse=True)
            if len(unique_vertices) != len(mesh.vertices):
                mesh.vertices = unique_vertices
                mesh.faces = inverse[mesh.faces]
        except Exception as exc:  # pragma: no cover - defensive fallback
            logging.debug("Fallback duplicate vertex removal failed: %s", exc)
    return before - len(mesh.vertices)


def _remove_duplicate_faces(mesh: trimesh.Trimesh) -> int:
    before = len(mesh.faces)
    if hasattr(mesh, "remove_duplicate_faces"):
        _safe_call("remove_duplicate_faces", mesh.remove_duplicate_faces)
    elif hasattr(trimesh.repair, "remove_duplicate_faces"):
        _safe_call("trimesh.repair.remove_duplicate_faces", lambda: trimesh.repair.remove_duplicate_faces(mesh))
    if len(mesh.faces) == before and len(mesh.faces) > 0:
        try:
            sorted_faces = np.sort(mesh.faces, axis=1)
            _, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
            if len(unique_indices) != len(mesh.faces):
                keep_mask = np.zeros(len(mesh.faces), dtype=bool)
                keep_mask[unique_indices] = True
                mesh.update_faces(keep_mask)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logging.debug("Fallback duplicate face removal failed: %s", exc)
    return before - len(mesh.faces)


def _remove_small_faces(mesh: trimesh.Trimesh, min_face_area: float) -> int:
    if len(mesh.faces) == 0:
        return 0
    try:
        areas = mesh.area_faces
    except Exception as exc:  # pragma: no cover - defensive logging only
        logging.debug("Failed to compute area_faces: %s", exc)
        return 0
    if not isinstance(areas, np.ndarray):
        return 0
    mask = areas <= max(min_face_area, 0.0)
    removed = int(np.count_nonzero(mask))
    if removed > 0:
        keep_mask = np.logical_not(mask)
        mesh.update_faces(keep_mask)
        if hasattr(mesh, "remove_unreferenced_vertices"):
            mesh.remove_unreferenced_vertices()
        elif hasattr(trimesh.repair, "remove_unreferenced_vertices"):
            trimesh.repair.remove_unreferenced_vertices(mesh)
    return removed


def _remove_unreferenced_vertices(mesh: trimesh.Trimesh) -> int:
    before = len(mesh.vertices)
    if hasattr(mesh, "remove_unreferenced_vertices"):
        _safe_call("remove_unreferenced_vertices", mesh.remove_unreferenced_vertices)
    elif hasattr(trimesh.repair, "remove_unreferenced_vertices"):
        _safe_call("trimesh.repair.remove_unreferenced_vertices", lambda: trimesh.repair.remove_unreferenced_vertices(mesh))
    return before - len(mesh.vertices)


def _keep_largest_component(mesh: trimesh.Trimesh) -> int:
    try:
        components = mesh.split(only_watertight=False)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logging.debug("mesh.split failed during cleaning: %s", exc)
        return 0
    if len(components) <= 1:
        return 0
    selected = max(components, key=lambda part: len(part.faces))
    removed_components = len(components) - 1
    if selected is not mesh:
        mesh.vertices = np.array(selected.vertices)
        mesh.faces = np.array(selected.faces)
    return removed_components


def _recompute_normals(mesh: trimesh.Trimesh) -> None:
    if hasattr(mesh, "fix_normals"):
        _safe_call("fix_normals", mesh.fix_normals)
    elif hasattr(trimesh.repair, "fix_normals"):
        _safe_call("trimesh.repair.fix_normals", lambda: trimesh.repair.fix_normals(mesh))
    # Accessors trigger recomputation in trimesh.
    _ = mesh.face_normals
    _ = mesh.vertex_normals


def clean_mesh(mesh: trimesh.Trimesh, min_face_area: float = 0.0, merge_tolerance: float = 1e-12) -> CleanReport:
    """Remove duplicate vertices/faces, drop tiny faces, and recompute normals."""

    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.faces)
    components_before = 1
    try:
        components_before = len(mesh.split(only_watertight=False))
    except Exception:  # pragma: no cover - defensive fallback
        components_before = 1

    removed_duplicate_vertices = _remove_duplicate_vertices(mesh, merge_tolerance=merge_tolerance)
    removed_duplicate_faces = _remove_duplicate_faces(mesh)
    removed_small_faces = _remove_small_faces(mesh, min_face_area=min_face_area)
    removed_unreferenced_vertices = _remove_unreferenced_vertices(mesh)
    removed_components = _keep_largest_component(mesh)
    components_after = 1
    try:
        components_after = len(mesh.split(only_watertight=False))
    except Exception:  # pragma: no cover - defensive fallback
        components_after = 1
    _recompute_normals(mesh)

    return CleanReport(
        original_vertices=original_vertices,
        original_faces=original_faces,
        cleaned_vertices=len(mesh.vertices),
        cleaned_faces=len(mesh.faces),
        removed_duplicate_vertices=removed_duplicate_vertices,
        removed_duplicate_faces=removed_duplicate_faces,
        removed_small_or_zero_faces=removed_small_faces,
        removed_unreferenced_vertices=removed_unreferenced_vertices,
        removed_components=removed_components,
        components_before=components_before,
        components_after=components_after,
        min_face_area=float(min_face_area),
    )
