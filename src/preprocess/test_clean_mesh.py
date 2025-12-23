import trimesh

from src.preprocess.clean_mesh import clean_mesh


def test_clean_mesh_removes_duplicates_and_small_faces():
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    faces = [
        (0, 1, 2),
        (0, 1, 2),  # duplicate face
        (0, 3, 2),  # zero-area because 0 == 3
    ]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    report = clean_mesh(mesh, min_face_area=1e-9)

    assert report.original_faces == 3
    assert report.cleaned_faces == 1
    assert report.removed_duplicate_faces == 1
    assert report.removed_small_or_zero_faces == 1
    assert report.removed_duplicate_vertices == 1
    assert len(mesh.faces) == 1
    assert len(mesh.vertices) == 3
