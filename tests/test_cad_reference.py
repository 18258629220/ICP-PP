"""Exact-face path and trimming checks, independent of private geometry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
cq = pytest.importorskip('cadquery')
from planning.cad_reference import paths_on_face, build_reference_paths


def test_subsampling_size_hole_is_split_by_exact_trim():
    face = cq.Workplane('XY').rect(10, 10).circle(.01).extrude(1).faces('>Z').val()
    controls = [face.paramAt((-4, 0, 1)), face.paramAt((4, 0, 1))]
    paths = paths_on_face(face, controls, spacing=.5, outward=(0, 0, 1))
    assert len(paths) == 2
    assert paths[0]['points_C'][-1, 0] == pytest.approx(-.01, abs=1e-7)
    assert paths[1]['points_C'][0, 0] == pytest.approx(.01, abs=1e-7)
    for path in paths:
        assert np.allclose(path['normals_C'], [0, 0, 1])
        assert np.max(np.linalg.norm(np.diff(path['points_C'], axis=0), axis=1)) <= .500001


def test_cylindrical_normals_are_analytic_and_do_not_flip():
    face = cq.Workplane('XY').circle(5).extrude(10).faces('%CYLINDER').val()
    u0, u1, v0, v1 = face.uvBounds()
    paths = paths_on_face(face, [[u0+.1, (v0+v1)/2], [u0+1.3, (v0+v1)/2]], .3, (1, 0, 0))
    assert len(paths) == 1
    p, n = paths[0]['points_C'], paths[0]['normals_C']
    assert np.allclose(np.linalg.norm(p[:, :2], axis=1), 5)
    assert np.allclose(n[:, :2], p[:, :2]/5, atol=1e-8)
    assert np.all(np.einsum('ij,ij->i', n[:-1], n[1:]) > .99)
    assert np.linalg.norm(n[0]-n[-1]) > .5


def test_explicit_xyz_and_uv_selection_are_reproducible(tmp_path):
    face = cq.Workplane('XY').rect(10, 8).extrude(1).faces('>Z').val()
    step = tmp_path/'face.step'
    cq.exporters.export(face, str(step))
    support = np.array([[x, y, 1] for x in np.linspace(-5, 5, 10) for y in np.linspace(-4, 4, 9)])
    cfg = {'selections': [{'face_id': 0, 'xyz_controls': [[-3, 1, 2], [3, 1, 2]]}],
           'approach_outward_C': [0, 0, 1], 'path_sampling_mm': .4}
    p1, cat = build_reference_paths(step, support, cfg)
    p2, _ = build_reference_paths(step, support, cfg)
    assert len(p1) == 1
    assert cat['selection_mode'] == 'explicit'
    assert np.allclose(p1[0]['points_C'], p2[0]['points_C'])
    assert np.allclose(p1[0]['points_C'][:, 2], 1)
    assert np.all(p1[0]['face_ids'] == 0)


def test_invalid_manual_selection_is_not_replaced_by_auto(tmp_path):
    face = cq.Workplane('XY').rect(10, 8).extrude(1).faces('>Z').val()
    step = tmp_path/'face.step'; cq.exporters.export(face, str(step))
    with pytest.raises(ValueError, match='Unknown STEP face_id'):
        build_reference_paths(step, np.zeros((1, 3)), {'selections': [{'face_id': 99, 'uv_controls': [[0, 0], [1, 1]]}]})
