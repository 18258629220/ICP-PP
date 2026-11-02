"""Exact-face path and trimming checks, independent of private geometry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
cq = pytest.importorskip('cadquery')
from planning.cad_reference import paths_on_face


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
