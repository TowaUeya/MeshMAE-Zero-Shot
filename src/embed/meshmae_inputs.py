"""Utilities for building MeshMAE-compatible inputs from MAPS meshes.

The MeshMAE encoder expects per-face tensors grouped into fixed-size patches.
This module loads MAPS meshes (e.g. ``*_MAPS.obj``) and constructs the
``faces``, ``feats``, ``centers``, ``Fs`` and ``cordinates`` tensors that are
fed into ``Mesh_mae.forward``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, Union

import torch
import trimesh

MeshInput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _pad_or_trim_faces(faces: torch.Tensor, total_faces: int) -> torch.Tensor:
    if faces.shape[0] == 0:
        raise ValueError("MAPS mesh contains no faces.")
    if faces.shape[0] == total_faces:
        return faces
    if faces.shape[0] < total_faces:
        reps = math.ceil(total_faces / faces.shape[0])
        faces = faces.repeat((reps, 1))
    return faces[:total_faces]


def _compute_face_angles(edge_lengths: torch.Tensor) -> torch.Tensor:
    a = edge_lengths[:, 0]
    b = edge_lengths[:, 1]
    c = edge_lengths[:, 2]
    cos_angle0 = (a.pow(2) + b.pow(2) - c.pow(2)) / (2 * a * b).clamp_min(1e-8)
    cos_angle1 = (a.pow(2) + c.pow(2) - b.pow(2)) / (2 * a * c).clamp_min(1e-8)
    cos_angle2 = (b.pow(2) + c.pow(2) - a.pow(2)) / (2 * b * c).clamp_min(1e-8)
    cos_angles = torch.stack([cos_angle0, cos_angle1, cos_angle2], dim=1)
    cos_angles = cos_angles.clamp(-1.0, 1.0)
    return torch.acos(cos_angles)


def _compute_face_features_legacy(face_coords: torch.Tensor) -> torch.Tensor:
    """Compute 13-channel features per face.

    The feature layout is ``[center(3), normal(3), area(1), edge_lengths(3),
    corner_angles(3)]`` which matches MeshMAE's ``channels=13`` setting.
    """

    v0 = face_coords[:, 0]
    v1 = face_coords[:, 1]
    v2 = face_coords[:, 2]

    centers = (v0 + v1 + v2) / 3.0
    e0 = v1 - v0
    e1 = v2 - v0
    e2 = v2 - v1

    normals = torch.cross(e0, e1, dim=1)
    normal_norm = torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-8)
    normals = normals / normal_norm
    areas = 0.5 * normal_norm

    edge_lengths = torch.stack(
        [
            torch.linalg.norm(e0, dim=1),
            torch.linalg.norm(e1, dim=1),
            torch.linalg.norm(e2, dim=1),
        ],
        dim=1,
    )
    angles = _compute_face_angles(edge_lengths)

    return torch.cat([centers, normals, areas, edge_lengths, angles], dim=1)


def _compute_face_features_paper(
    face_coords: torch.Tensor,
    face_vertex_normals: torch.Tensor,
) -> torch.Tensor:
    """Compute 10-channel features per face following the MeshMAE paper.

    Feature layout: ``[area(1), interior_angles(3), face_normal(3), dot_products(3)]``.
    """

    v0 = face_coords[:, 0]
    v1 = face_coords[:, 1]
    v2 = face_coords[:, 2]

    e0 = v1 - v0
    e1 = v2 - v0
    e2 = v2 - v1

    normals = torch.cross(e0, e1, dim=1)
    normal_norm = torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-8)
    normals = normals / normal_norm
    areas = 0.5 * normal_norm

    edge_lengths = torch.stack(
        [
            torch.linalg.norm(e0, dim=1),
            torch.linalg.norm(e1, dim=1),
            torch.linalg.norm(e2, dim=1),
        ],
        dim=1,
    )
    angles = _compute_face_angles(edge_lengths)

    dot_products = (normals.unsqueeze(1) * face_vertex_normals).sum(dim=2)

    return torch.cat([areas, angles, normals, dot_products], dim=1)


def build_meshmae_inputs(
    mesh: Union[trimesh.Trimesh, str, Path],
    device: torch.device,
    patch_size: int = 64,
    num_patches: int = 256,
    feature_mode: str = "paper10",
) -> MeshInput:
    """Build MeshMAE inputs from a MAPS mesh.

    Parameters
    ----------
    mesh:
        A MAPS mesh (``trimesh.Trimesh`` or mesh path). The face order is
        preserved to keep MAPS patch continuity.
    device:
        Target torch device for all returned tensors.
    patch_size:
        Faces per patch. MeshMAE default is 64.
    num_patches:
        Total patches. MeshMAE default is 256.
    feature_mode:
        Feature mode for per-face descriptors. ``paper10`` uses the 10-channel
        features described in the MeshMAE paper; ``legacy13`` keeps the previous
        13-channel feature set (center/normal/area/edges/angles).

    Returns
    -------
    faces:
        Long tensor of shape ``(1, num_patches, patch_size, 3)`` on ``device``.
    feats:
        Float tensor of shape ``(1, channels, num_patches, patch_size)`` on ``device``.
    centers:
        Float tensor of shape ``(1, num_patches, patch_size, 3)`` on ``device``.
    Fs:
        Long tensor of shape ``(1,)`` filled with ``num_patches * patch_size``.
    cordinates:
        Float tensor of shape ``(1, num_patches, patch_size, 3, 3)`` on ``device``.

    Examples
    --------
    >>> inputs = build_meshmae_inputs("sample_MAPS.obj", torch.device("cpu"))
    >>> faces, feats, centers, Fs, cordinates = inputs
    >>> faces.shape
    torch.Size([1, 256, 64, 3])
    >>> feats.shape
    torch.Size([1, 10, 256, 64])
    >>> feats.dtype, feats.device
    (torch.float32, device(type='cpu'))
    """

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.load_mesh(mesh, process=False)
    faces_np = torch.tensor(mesh.faces, dtype=torch.long)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    vertex_normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)

    total_faces = patch_size * num_patches
    faces_np = _pad_or_trim_faces(faces_np, total_faces)
    face_coords = vertices[faces_np]

    face_centers = face_coords.mean(dim=1)

    if feature_mode == "paper10":
        face_vertex_normals = vertex_normals[faces_np]
        feats = _compute_face_features_paper(face_coords, face_vertex_normals)
    elif feature_mode == "legacy13":
        feats = _compute_face_features_legacy(face_coords)
    else:
        raise ValueError(f"Unsupported feature_mode '{feature_mode}'. Use 'paper10' or 'legacy13'.")

    faces_np = faces_np.view(num_patches, patch_size, 3)
    face_coords = face_coords.view(num_patches, patch_size, 3, 3)
    feats = feats.view(num_patches, patch_size, feats.shape[-1])
    feats = feats.permute(2, 0, 1)
    centers = face_centers.view(num_patches, patch_size, 3)
    Fs = torch.tensor([total_faces], dtype=torch.long)

    faces_np = faces_np.unsqueeze(0).to(device)
    feats = feats.unsqueeze(0).to(device)
    centers = centers.unsqueeze(0).to(device)
    Fs = Fs.to(device)
    face_coords = face_coords.unsqueeze(0).to(device)

    return faces_np, feats, centers, Fs, face_coords
