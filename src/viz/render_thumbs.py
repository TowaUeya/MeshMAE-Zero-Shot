"""Generate thumbnail renders for fossil meshes using trimesh + pyglet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

import trimesh


SUPPORTED_EXTENSIONS = (".ply", ".stl", ".obj")


def iter_meshes(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def render_thumbnail(mesh_path: Path, output_path: Path, resolution: Tuple[int, int]) -> None:
    mesh = trimesh.load_mesh(mesh_path, process=True)
    scene = mesh.scene()
    scene.camera.fov = (50, 50)
    scene.camera.distance = scene.scale * 2.0
    png = scene.save_image(resolution=resolution, visible=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render thumbnails for meshes")
    parser.add_argument("--input", type=Path, required=True, help="Directory with meshes")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save PNG thumbnails")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 512), help="Width height in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    for mesh_path in iter_meshes(args.input):
        rel = mesh_path.relative_to(args.input)
        output_path = args.output / rel.with_suffix(".png")
        logging.info("Rendering %s", mesh_path)
        render_thumbnail(mesh_path, output_path, tuple(args.resolution))


if __name__ == "__main__":
    main()
