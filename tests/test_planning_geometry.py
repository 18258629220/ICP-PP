"""Known geometry and discontinuities, independent of the private workpiece."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from shapely.geometry import Polygon
from planning.geometry import tool_frame, ocl_frame, ball_drop, raster_lines, split_valid
from planning.validation import numerical_sign_check, physical_gate


def plane(slope=0.):
    p = np.array([[-10, -10, -10*slope], [10, -10, 10*slope], [10, 10, 10*slope], [-10, 10, -10*slope]])
    return p[[[0, 1, 2], [0, 2, 3]]]


@pytest.mark.parametrize('outward', [[0, 0, -1], [0, 0, 1], [1, 2, 3]])
def test_right_handed_frame_and_approach(outward):
    outward = np.asarray(outward)/np.linalg.norm(outward)
    frame = tool_frame(outward)
    np.testing.assert_allclose(frame.T@frame, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(frame[:, 2], -outward)
    assert np.linalg.det(frame) == pytest.approx(1)
    np.testing.assert_allclose(ocl_frame(outward)[:, 2], outward)


@pytest.mark.parametrize('slope', [0., .4])
def test_opencamlib_ball_offset_known_plane(slope):
    radius = 2.
    tip, contact, touched = ball_drop(plane(slope), np.array([[0., 0.], [1., 2.]]), 4., 20., -100.)
    assert touched.all()
    centers = tip + [0, 0, radius]
    normal = np.array([-slope, 0., 1.])/np.sqrt(1+slope*slope)
    expected_z = slope*tip[:, 0] + radius*np.sqrt(1+slope*slope)
    np.testing.assert_allclose(centers[:, 2], expected_z, atol=1e-6)
    np.testing.assert_allclose(centers-contact, np.tile(radius*normal, (2, 1)), atol=1e-6)


def test_raster_does_not_bridge_hole():
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(4, 3), (6, 3), (6, 7), (4, 7)]])
    lines = raster_lines(polygon, 1., .4)
    from shapely.geometry import LineString
    assert lines
    for points in lines:
        assert polygon.covers(LineString(points))


def test_invalid_sample_and_height_jump_split_strokes():
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 5], [6, 0, 5]])
    groups = split_valid(points, np.array([1, 1, 0, 1, 1, 1, 1], bool), 1.1)
    assert [x.tolist() for x in groups] == [[0, 1], [3, 4], [5, 6]]


def test_m7_sign_and_physical_gate_remain_distinct():
    assert numerical_sign_check()['passed']
    assert not physical_gate(False, False, None)['passed']
    assert len(physical_gate(False, False, None)['pending']) == 3
