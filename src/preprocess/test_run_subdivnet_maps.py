import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.preprocess import run_subdivnet_maps


class DummyMesh:
    def __init__(self, face_count: int):
        self.vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        self.faces = [[0, 1, 2]] * face_count


class SuccessfulMaps:
    def __init__(self, vertices, faces, base_size: int, verbose: bool):
        self.base_size = base_size + 1

    def mesh_upsampling(self, depth: int):
        class UpsampledMesh:
            def export(self, path):
                Path(path).write_text("upsampled")

        return UpsampledMesh()


class FailingMaps:
    def __init__(self, vertices, faces, base_size: int, verbose: bool):
        raise RuntimeError("maps init failed")


def test_parse_args_accepts_metadata(tmp_path):
    metadata_path = tmp_path / "meta.json"
    args = run_subdivnet_maps.parse_args(
        [
            "--subdivnet_root",
            str(tmp_path / "subdivnet"),
            "--input",
            str(tmp_path / "mesh.obj"),
            "--out-dir",
            str(tmp_path / "out"),
            "--metadata",
            str(metadata_path),
        ]
    )
    assert args.metadata == metadata_path


def test_run_maps_writes_metadata_on_success(tmp_path, monkeypatch):
    metadata_path = tmp_path / "meta" / "info.json"
    output_dir = tmp_path / "out"
    input_path = tmp_path / "mesh.obj"
    input_path.write_text("mesh")

    dummy_mesh = DummyMesh(face_count=6)

    monkeypatch.setattr(run_subdivnet_maps.trimesh, "load_mesh", lambda *args, **kwargs: dummy_mesh)
    monkeypatch.setattr(run_subdivnet_maps, "resolve_subdivnet", lambda root: (object(), SuccessfulMaps))

    result_path = run_subdivnet_maps.run_maps(
        subdivnet_root=tmp_path / "subdivnet",
        input_path=input_path,
        out_dir=output_dir,
        output_path=None,
        base_size=8,
        depth=2,
        max_base_size=None,
        verbose=False,
        metadata=metadata_path,
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["input_faces"] == 6
    assert metadata["input_vertices"] == 3
    assert metadata["attempted_base_sizes"] == [6]
    assert metadata["chosen_base_size"] == 6
    assert metadata["actual_base_size"] == 7
    assert metadata["success"] is True
    assert metadata["output_path"] == str(result_path)
    assert metadata["output_path_relative"] == Path(metadata["output_path"]).name
    assert metadata["cleaning"] is None
    assert metadata["cleaned_input_path"] is None
    assert metadata["failed_mesh_path"] is None
    assert metadata["error"] is None
    assert result_path.read_text() == "upsampled"


def test_run_maps_writes_metadata_on_failure(tmp_path, monkeypatch):
    metadata_path = tmp_path / "meta.json"
    output_dir = tmp_path / "out"
    input_path = tmp_path / "mesh.obj"
    input_path.write_text("mesh")

    dummy_mesh = DummyMesh(face_count=5)

    monkeypatch.setattr(run_subdivnet_maps.trimesh, "load_mesh", lambda *args, **kwargs: dummy_mesh)
    monkeypatch.setattr(run_subdivnet_maps, "resolve_subdivnet", lambda root: (object(), FailingMaps))

    with pytest.raises(RuntimeError):
        run_subdivnet_maps.run_maps(
            subdivnet_root=tmp_path / "subdivnet",
            input_path=input_path,
            out_dir=output_dir,
            output_path=None,
            base_size=6,
            depth=2,
            max_base_size=None,
            verbose=False,
        metadata=metadata_path,
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["attempted_base_sizes"] == [5, 4]
    assert metadata["chosen_base_size"] is None
    assert metadata["actual_base_size"] is None
    assert metadata["success"] is False
    assert metadata["output_path"] == str((output_dir / "mesh_MAPS.obj").resolve())
    assert metadata["failed_mesh_path"] == str(input_path.resolve())
    assert metadata["failed_mesh_relative"] == input_path.name
    assert "maps init failed" in metadata["error"]


def test_run_maps_cleans_mesh_and_records_metadata(tmp_path, monkeypatch):
    metadata_path = tmp_path / "meta.json"
    output_dir = tmp_path / "out"
    input_path = tmp_path / "mesh.obj"
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    faces = [
        (0, 1, 2),
        (0, 1, 2),  # duplicate
        (0, 3, 2),  # zero-area because 0 == 3
    ]
    real_mesh = run_subdivnet_maps.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    input_path.write_text("mesh")

    monkeypatch.setattr(run_subdivnet_maps.trimesh, "load_mesh", lambda *args, **kwargs: real_mesh.copy())
    monkeypatch.setattr(run_subdivnet_maps, "resolve_subdivnet", lambda root: (object(), SuccessfulMaps))

    result_path = run_subdivnet_maps.run_maps(
        subdivnet_root=tmp_path / "subdivnet",
        input_path=input_path,
        out_dir=output_dir,
        output_path=None,
        base_size=4,
        depth=2,
        max_base_size=None,
        verbose=False,
        metadata=metadata_path,
        clean_input=True,
        clean_min_face_area=1e-9,
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["cleaning"]["removed_duplicate_vertices"] == 1
    assert metadata["cleaning"]["removed_duplicate_faces"] == 1
    assert metadata["cleaning"]["removed_small_or_zero_faces"] == 1
    assert metadata["cleaning"]["cleaned_faces"] == 1
    assert metadata["cleaned_input_relative"] == f"{input_path.stem}_cleaned{input_path.suffix}"
    cleaned_mesh_path = output_dir / metadata["cleaned_input_relative"]
    assert cleaned_mesh_path.exists()
    assert result_path.read_text() == "upsampled"
