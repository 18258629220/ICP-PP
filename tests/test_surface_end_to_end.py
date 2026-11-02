"""Independent analytic truth through STEP -> labels -> coverage -> CSV poses."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip('cadquery')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analytic_surface_case_preserves_labels_and_maps_derivative_axes(tmp_path):
    generate = _load('synthetic_surface_case', ROOT/'scripts/create_synthetic_surface_case.py')
    workflow = _load('surface_workflow_cli', ROOT/'09_plan_surface_workflow.py')
    source, output = tmp_path/'source', tmp_path/'result'
    case = generate.create_case(source)
    config = json.loads((source/'surface_config.json').read_text())
    config['segmentation_chunk_points'] = 701  # Force block boundaries.
    (source/'surface_config.json').write_text(json.dumps(config))
    workflow.run(source/'config.json', source/'surface_config.json', output, skip_visuals=True)
    labels = pd.read_csv(output/'scan_face_labels.csv')
    truth = pd.read_csv(source/'ground_truth.csv')
    np.testing.assert_array_equal(labels.source_index, truth.source_index)
    np.testing.assert_array_equal(labels.face_id, truth.face_id)
    np.testing.assert_array_equal(np.load(output/'face_labels.npy'), truth.face_id)
    poses = pd.read_csv(output/'layer_000/tool_axis_poses_CAD.csv')
    plane = poses[poses.face_id == case['plate_face_id']]
    assert len(plane) > 100
    x, y = plane.x_mm.to_numpy(), plane.y_mm.to_numpy()
    expected_z = 2+.15+.015*x+.008*y+.0008*x*x
    np.testing.assert_allclose(plane.z_mm, expected_z, atol=2e-6)
    normal = np.column_stack([-(.015+.0016*x), np.full(len(x),-.008), np.ones(len(x))])
    normal /= np.linalg.norm(normal,axis=1)[:,None]
    np.testing.assert_allclose(plane[['normal_x','normal_y','normal_z']], normal, atol=2e-6)
    np.testing.assert_allclose(plane[['tool_z_x','tool_z_y','tool_z_z']], -normal, atol=2e-6)
    summary = json.loads((output/'summary.json').read_text())
    assert summary['process_region_count'] >= 1
    assert all(check['passed'] for check in summary['checks'])
    assert summary['layers'][1]['poses'] > 0
    assert summary['layers'][2]['poses'] > 0
    for index in [1, 2]:
        target = pd.read_csv(output/f'layer_{index:03d}'/'tool_axis_poses_CAD.csv')
        assert target.process_eligible.all()
        assert not target.execution_allowed.any()
    assert not poses.execution_allowed.any()
    assert json.loads((output/'run_state.json').read_text())['status'] == 'completed'
    with pytest.raises(FileExistsError):
        workflow.run(source/'config.json', source/'surface_config.json', output, skip_visuals=True)
    # Even an interrupted archive belongs to its original invocation.
    (output/'run_state.json').write_text('{"status":"running","stage":"segmentation"}')
    before = (output/'run_state.json').read_bytes()
    with pytest.raises(FileExistsError):
        workflow.run(source/'config.json', source/'surface_config.json', output, skip_visuals=True)
    assert (output/'run_state.json').read_bytes() == before
