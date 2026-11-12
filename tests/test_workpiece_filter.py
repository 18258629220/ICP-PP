"""Checks for CAD filtering that could otherwise corrupt BASE/index provenance."""
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

from pipeline_common import read_ply, transform_points
from planning.workpiece_filter import filter_scan_by_cad, write_point_subset_ply


def test_open_cad_distance_filter_preserves_base_rows_and_uncertain_faces(tmp_path):
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                                        dtype=o3d.core.Dtype.Float32),
                        o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.Dtype.UInt32))
    transform = np.array([[0., -1., 0., 4.], [1., 0., 0., -2.],
                          [0., 0., 1., 1.], [0., 0., 0., 1.]])
    cad_points = np.array([[.2, .2, .1], [.2, .2, 4.], [3., 3., 3.], [.1, .1, .15]])
    base_points = transform_points(cad_points, np.linalg.inv(transform))
    raw = np.zeros(4, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                             ('intensity', 'u1'), ('event_id', '<u2')])
    for j, axis in enumerate('xyz'):
        raw[axis] = base_points[:, j]
    raw['intensity'] = [12, 13, 14, 15]
    raw['event_id'] = [21, 22, 23, 24]
    segmenter = SimpleNamespace(scene=scene, nthreads=1, mesh_error_margin_mm=.01,
        unmeshed_face_ids=np.array([0]),
        topology=SimpleNamespace(face_bounds=np.array([[2.99, 2.99, 2.99, 3.01, 3.01, 3.01]])))
    report = filter_scan_by_cad(raw, transform, segmenter, tmp_path, .2, chunk_points=2)
    assert report['kept_points'] == 3
    assert report['protected_unmeshed_points'] == 1
    indices = np.load(tmp_path/'source_indices.npy')
    np.testing.assert_array_equal(indices, [0, 2, 3])
    np.testing.assert_array_equal(np.load(tmp_path/'removed_source_indices.npy'), [1])
    labels = np.array([7, -1, 9], dtype=np.int32)
    write_point_subset_ply(tmp_path/'segmented.ply', raw, indices, labels, chunk_points=2)
    result, _ = read_ply(tmp_path/'segmented.ply')
    for name in raw.dtype.names:
        # Includes arbitrary scalar attributes; coordinates stay in BASE.
        assert result[name].tobytes() == raw[name][indices].tobytes()
    np.testing.assert_array_equal(result['cad_face_id'], labels)
    np.testing.assert_array_equal([result['red'][1], result['green'][1], result['blue'][1]],
                                  [150, 158, 167])
    assert 'planned_surface' not in result.dtype.names
    with pytest.raises(FileExistsError):
        filter_scan_by_cad(raw, transform, segmenter, tmp_path, .2)


def test_subset_writer_rejects_reordered_indices_and_misaligned_labels(tmp_path):
    raw = np.zeros(3, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    with pytest.raises(ValueError, match='increasing'):
        write_point_subset_ply(tmp_path/'bad-order.ply', raw, np.array([2, 0]))
    with pytest.raises(ValueError, match='align'):
        write_point_subset_ply(tmp_path/'bad-label.ply', raw, np.array([0, 2]), np.array([7]))
    with pytest.raises(ValueError, match='Completed'):
        write_point_subset_ply(tmp_path/'unfinished.ply', raw, np.array([0]), np.array([-2]))
    assert not list(tmp_path.glob('*.ply'))
