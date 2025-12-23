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
import sys
import traceback
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
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path to write MAPS run metadata as JSON",
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
        "attempted_base_sizes": [],
        "chosen_base_size": None,
        "actual_base_size": None,
        "success": False,
        "output_path": str(output_path),
        "error": None,
    }

    mesh = trimesh.load_mesh(input_path, process=False)
    face_count = len(mesh.faces)
    if face_count == 0:
        raise ValueError(f"Mesh {input_path} has no faces; cannot generate MAPS")

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
    )


if __name__ == "__main__":
    main()
