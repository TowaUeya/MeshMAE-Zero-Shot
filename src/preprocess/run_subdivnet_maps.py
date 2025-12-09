"""Safe wrapper around SubdivNet MAPS generation for a single mesh.

This module keeps the SubdivNet dependency isolated from the main
MeshMAE preprocessing flow by importing `datagen_maps.py` inside a
separate Python process. It avoids triggering the demo entrypoints in
`datagen_maps.py` and passes absolute paths for both input and output.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional

import trimesh


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

    mesh = trimesh.load_mesh(input_path, process=False)
    maps = maps_cls(mesh.vertices, mesh.faces, base_size=base_size, verbose=verbose)
    if max_base_size is not None and getattr(maps, "base_size", base_size) > max_base_size:
        raise ValueError(
            f"Computed base_size {getattr(maps, 'base_size', base_size)} exceeds max_base_size {max_base_size}"
        )
    sub_mesh = maps.mesh_upsampling(depth=depth)
    sub_mesh.export(output_path)
    return output_path


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
    )


if __name__ == "__main__":
    main()
