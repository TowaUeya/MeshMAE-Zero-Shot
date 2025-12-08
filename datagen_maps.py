"""MAPS dataset/shape generator with CLI support.

This is an adapted copy of the SubdivNet `datagen_maps.py` script with
small changes so it can be called from `make_manifold_and_maps.py`.

Notes
-----
- Dataset mode assumes the SubdivNet folder layout of
  `<src_root>/<label>/<split>/*.obj`. For loosely organized meshes
  (例: 化石データがクラス分けされていない場合) は、`make_manifold_and_maps.py`
  の単一メッシュ呼び出し経由で使う方が安全です。
- `FOSSILS_CONFIG` は化石データ用の控えめな既定値です
  (`base_size=96`, `depth=3`, `max_base_size=192`, `n_variation=1`)。

Usage
-----
- Single shape:
    python datagen_maps.py <input_mesh> <output_mesh> --base_size 96 --depth 3

- Dataset (use a named config):
    python datagen_maps.py --config FOSSILS

The script retains the original demo functions for compatibility but
prefers the CLI entrypoint when arguments are provided.
"""

import argparse
import os
import traceback
from multiprocessing import Pool
from multiprocessing.context import TimeoutError as MTE
from pathlib import Path
from typing import Dict

import numpy as np
import trimesh
from maps import MAPS
from tqdm import tqdm


SHREC_CONFIG = {
    "dst_root": "./data/SHREC11-MAPS-48-4-split10",
    "src_root": "./data/shrec11-split10",
    "n_variation": 10,
    "base_size": 48,
    "depth": 4,
}

CUBES_CONFIG = {
    "dst_root": "./data/Cubes-MAPS-48-4",
    "src_root": "./data/cubes",
    "n_variation": 10,
    "base_size": 48,
    "depth": 4,
}

MANIFOLD40_CONFIG = {
    "dst_root": "./data/Manifold40-MAPS-96-3",
    "src_root": "./data/Manifold40",
    "n_variation": 10,
    "base_size": 96,
    "max_base_size": 192,
    "depth": 3,
}

FOSSILS_CONFIG = {
    "dst_root": "../MeshMAE-Zero-Shot/datasets/fossils_maps",
    "src_root": "../MeshMAE-Zero-Shot/datasets/fossils_raw",
    "n_variation": 1,
    "base_size": 96,
    "max_base_size": 192,
    "depth": 3,
}

CONFIGS: Dict[str, dict] = {
    "SHREC": SHREC_CONFIG,
    "CUBES": CUBES_CONFIG,
    "MANIFOLD40": MANIFOLD40_CONFIG,
    "FOSSILS": FOSSILS_CONFIG,
}


def maps_async(obj_path, out_path, base_size, max_base_size, depth, timeout, trial=1, verbose=False):
    if verbose:
        print("[IN]", out_path)

    for _ in range(trial):
        try:
            mesh = trimesh.load(obj_path, process=False)
            maps = MAPS(mesh.vertices, mesh.faces, base_size, timeout=timeout, verbose=verbose)

            if maps.base_size > max_base_size:
                continue

            sub_mesh = maps.mesh_upsampling(depth=depth)
            sub_mesh.export(out_path)
            break
        except Exception:
            if verbose:
                traceback.print_exc()
    else:
        if verbose:
            print("[OUT FAIL]", out_path)
        return False, out_path
    if verbose:
        print("[OUT SUCCESS]", out_path)
    return True, out_path


def make_MAPS_dataset(
    dst_root,
    src_root,
    base_size,
    depth,
    n_variation=None,
    n_worker=1,
    timeout=None,
    max_base_size=None,
    verbose=False,
):
    if max_base_size is None:
        max_base_size = base_size

    if os.path.exists("maps.log"):
        os.remove("maps.log")

    def callback(pbar, success, path):
        pbar.update()
        if not success:
            with open("maps.log", "a") as f:
                f.write(str(path) + "\n")

    for label_dir in sorted(Path(src_root).iterdir(), reverse=True):
        if label_dir.is_dir():
            for mode_dir in sorted(label_dir.iterdir()):
                if mode_dir.is_dir():
                    obj_paths = list(sorted(mode_dir.glob("*.obj")))
                    dst_dir = Path(dst_root) / label_dir.name / mode_dir.name
                    dst_dir.mkdir(parents=True, exist_ok=True)

                    pbar = tqdm(total=len(obj_paths) * n_variation)
                    pbar.set_description(f"{label_dir.name}-{mode_dir.name}")

                    if n_worker > 0:
                        pool = Pool(processes=n_worker)

                    results = []
                    for obj_path in obj_paths:
                        obj_id = str(obj_path.stem)

                        for var in range(n_variation):
                            dst_path = dst_dir / f"{obj_id}-{var}.obj"
                            if dst_path.exists():
                                continue

                            if n_worker > 0:
                                ret = pool.apply_async(
                                    maps_async,
                                    (str(obj_path), str(dst_path), base_size, max_base_size, depth, timeout),
                                    callback=lambda x: callback(pbar, x[0], x[1]),
                                )
                                results.append(ret)
                            else:
                                maps_async(
                                    str(obj_path),
                                    str(dst_path),
                                    base_size,
                                    max_base_size,
                                    depth,
                                    timeout,
                                    verbose=verbose,
                                )
                                pbar.update()

                    if n_worker > 0:
                        try:
                            [r.get(timeout + 1) for r in results]
                            pool.close()
                        except MTE:
                            pass

                    pbar.close()


def make_MAPS_shape(in_path, out_path, base_size, depth):
    mesh = trimesh.load_mesh(in_path, process=False)
    maps = MAPS(mesh.vertices, mesh.faces, base_size=base_size, verbose=True)
    sub_mesh = maps.mesh_upsampling(depth=depth)
    sub_mesh.export(out_path)


def MAPS_demo1():
    """Apply MAPS to a single 3D model"""
    make_MAPS_shape("airplane.obj", "airplane_MAPS.obj", 96, 3)


def MAPS_demo2(config_name: str = "MANIFOLD40"):
    """Apply MAPS to shapes from a dataset in parallel using a named config."""
    config = CONFIGS[config_name]
    make_MAPS_dataset(
        config["dst_root"],
        config["src_root"],
        config["base_size"],
        config["depth"],
        n_variation=config["n_variation"],
        n_worker=60,
        timeout=30,
        max_base_size=config.get("max_base_size", config["base_size"]),
        verbose=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MAPS outputs for meshes")
    parser.add_argument("input", nargs="?", help="Input mesh path for single-shape processing")
    parser.add_argument("output", nargs="?", help="Output path for single-shape processing")
    parser.add_argument("--base_size", type=int, default=96, help="Base mesh size for MAPS")
    parser.add_argument("--depth", type=int, default=3, help="Subdivision depth")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout for MAPS runs")
    parser.add_argument("--config", choices=CONFIGS.keys(), help="Run MAPS over a named dataset config")
    parser.add_argument("--n_worker", type=int, default=60, help="Parallel workers for dataset mode")
    parser.add_argument("--n_variation", type=int, default=None, help="Override variation count in config")
    parser.add_argument("--max_base_size", type=int, default=None, help="Maximum allowed base size")
    parser.add_argument("--demo1", action="store_true", help="Force running the original MAPS_demo1")
    parser.add_argument("--demo2", action="store_true", help="Force running the original MAPS_demo2")
    parser.add_argument("--verbose", action="store_true", help="Verbose MAPS logging")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Explicit demo requests keep legacy behaviour.
    if args.demo1:
        MAPS_demo1()
        return
    if args.demo2:
        MAPS_demo2()
        return

    # Single-shape mode when positional args are given.
    if args.input and args.output:
        make_MAPS_shape(args.input, args.output, args.base_size, args.depth)
        return

    # Dataset mode via named config.
    if args.config:
        config = CONFIGS[args.config]
        n_variation = args.n_variation if args.n_variation is not None else config.get("n_variation", 1)
        max_base_size = args.max_base_size if args.max_base_size is not None else config.get("max_base_size", args.base_size)
        make_MAPS_dataset(
            config["dst_root"],
            config["src_root"],
            config.get("base_size", args.base_size),
            config.get("depth", args.depth),
            n_variation=n_variation,
            n_worker=args.n_worker,
            timeout=args.timeout,
            max_base_size=max_base_size,
            verbose=args.verbose,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
