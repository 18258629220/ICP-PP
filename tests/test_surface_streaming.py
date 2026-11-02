"""Full-resolution row alignment and deterministic voxel support across chunks."""
from pathlib import Path
import importlib.util
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
cq = pytest.importorskip('cadquery')
from planning.cad_topology import CadTopology
from planning.segmentation import segment_points
from pipeline_common import read_ply, xyz_of, transform_points

spec = importlib.util.spec_from_file_location('surface_cli_streaming',
    Path(__file__).resolve().parents[1]/'09_plan_surface_workflow.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def _scan(path, points):
    dtype = np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('intensity','u1')])
    raw = np.zeros(len(points), dtype=dtype)
    for i, axis in enumerate('xyz'):
        raw[axis] = points[:,i]
    raw['intensity'] = np.arange(len(points))
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(points)}\nproperty double x\nproperty double y\n'
              'property double z\nproperty uchar intensity\nend_header\n')
    path.write_bytes(header.encode()+raw.tobytes())
    return read_ply(path)[0]


def test_streaming_matches_all_rows_and_support_for_different_chunks(tmp_path):
    topology = CadTopology(cq.Workplane('XY').box(10,8,.4).val().translate((0,0,.5)).Faces())
    points = np.array([[.01,.01,.72],[.02,.02,.73],[.03,.03,.28],[100,100,100],
                       [.01,.01,.72],[1.05,.01,.72],[.03,.03,.28],[1.06,.01,.72]])
    transform = np.eye(4); transform[:3,3] = [.3,.2,.1]
    scan = tmp_path/'input.ply'
    raw = _scan(scan, points-transform[:3,3])
    original = scan.read_bytes()
    assert isinstance(raw, np.memmap)
    all_rows = segment_points(transform_points(xyz_of(raw), transform), topology)
    results = []
    for chunk in (1,3,100):
        out = tmp_path/f'chunk_{chunk}'
        streamed = workflow.stream_segment_scan(raw, transform, topology, {}, 1., out, chunk)
        frame = pd.read_csv(out/'scan_face_labels.csv')
        pd.testing.assert_frame_equal(frame, all_rows, check_dtype=False)
        labels = np.load(out/'face_labels.npy', mmap_mode='r')
        assert isinstance(labels, np.memmap)
        np.testing.assert_array_equal(labels, all_rows.face_id)
        provenance = pd.read_csv(out/'support_provenance.csv')
        np.testing.assert_array_equal(provenance.source_index, [0,2,5])
        np.testing.assert_array_equal(streamed['support_source_indices'], [0,2,5])
        assert streamed['accepted_face_points'] == 7
        assert streamed['status_counts'] == {'assigned':7,'no_candidate':1}
        assert streamed['compact_segmentation'].point_count.sum() == 7
        # Both thin-wall faces survive one spatial voxel.
        assert streamed['support_faces'][0] != streamed['support_faces'][1]
        assert np.array_equal(frame.source_index, np.arange(len(points)))
        results.append(streamed)
    for key in ('support_points','support_faces','support_source_indices'):
        np.testing.assert_array_equal(results[0][key], results[1][key])
        np.testing.assert_array_equal(results[0][key], results[2][key])
    assert scan.read_bytes() == original


def test_interrupted_scan_has_no_completed_marker_and_unprocessed_sentinel(tmp_path):
    topology = CadTopology(cq.Workplane('XY').box(2,2,2).val().Faces())
    raw = _scan(tmp_path/'input.ply', np.array([[20,0,0],[np.nan,0,0]]))
    out = tmp_path/'partial'
    with pytest.raises(ValueError, match='Nonfinite scan positions'):
        workflow.stream_segment_scan(raw,np.eye(4),topology,{},.5,out,1)
    assert np.load(out/'face_labels.npy').tolist() == [-1,-2]
    assert len(pd.read_csv(out/'scan_face_labels.csv')) == 1
    assert not (out/'summary.json').exists()
