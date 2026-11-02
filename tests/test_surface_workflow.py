"""Known geometry and provenance checks for the non-robot surface workflow."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
cq = pytest.importorskip('cadquery')
from planning.surface_workflow import (
    assign_process_regions, build_surface_rasters, classify_surface_samples,
    target_layer_table,
)


@pytest.fixture
def cfg():
    value = json.loads((ROOT/'surface_workflow.example.json').read_text())
    value['process']['boundary_position_bound_mm'] = 0.
    return value


def _samples(offsets=(.4, .4, .4, .4, .4)):
    count = len(offsets)
    table = pd.DataFrame({'path_id': 'known_plane', 'sample_index': np.arange(count),
                          'segment_id': 0, 'face_id': 0, 'valid': True,
                          'status': 'accepted', 'u': np.arange(count)*.5, 'v': 0.,
                          'offset_mm': offsets, 'conditional_position_sigma_mm': .02,
                          'uncertainty_complete': True,
                          'max_abs_principal_curvature_per_mm': 0.})
    for axis in 'xyz':
        table[f'normal_{axis}'] = 1. if axis == 'z' else 0.
        table[f'cad_normal_{axis}'] = 1. if axis == 'z' else 0.
        table[f'cad_{axis}_mm'] = table.u if axis == 'x' else 0.
        table[f'mapped_{axis}_mm'] = table.u if axis == 'x' else (table.offset_mm if axis == 'z' else 0.)
    return table


def test_coverage_splits_exact_hole_and_never_connects_across_it(cfg):
    face = cq.Workplane('XY').rect(10, 10).circle(1.).extrude(1).faces('>Z').val()
    topology = SimpleNamespace(faces=[face])
    segmentation = pd.DataFrame({'face_id': np.zeros(20, int), 'valid': True})
    paths, records = build_surface_rasters(topology, segmentation, cfg, [0, 0, 1])
    assert records[0]['line_spacing_certified'] is False
    center_line_parts = []
    for path in paths:
        points = path['points_C']
        # Every straight segment lies outside the circular hole, including between samples.
        for a, b in zip(points[:-1, :2], points[1:, :2]):
            delta = b-a
            t = np.clip(-np.dot(a, delta)/np.dot(delta, delta), 0, 1)
            assert np.linalg.norm(a+t*delta) >= 1.-1e-7
        if np.max(np.abs(points[:, 1])) < 1e-7:
            center_line_parts.append(path)
    assert len(center_line_parts) == 2
    assert all(not (p['points_C'][:, 0].min() < 0 < p['points_C'][:, 0].max())
               for p in center_line_parts)


def test_boundary_and_explicit_protected_uv_region_override_allowance(cfg):
    cfg['process']['material_side_confirmed'] = True
    cfg['process']['protected_uv_boxes'] = [{'face_id': 0, 'bounds': [.9, 1.1, -.1, .1]}]
    table = _samples()
    result = classify_surface_samples(table, [.1, 10., 10., 10., 10.], cfg)
    assert result.iloc[0].process_status == 'boundary_protection'
    assert result.iloc[2].process_status == 'protected_uv_region'
    assert result.iloc[1].process_eligible
    assert not result.iloc[0].process_eligible and not result.iloc[2].process_eligible
    assert not result.execution_allowed.any()


def test_normal_uncertainty_cannot_substitute_for_boundary_position_bound(cfg):
    cfg['process']['material_side_confirmed'] = True
    cfg['process']['boundary_position_bound_mm'] = None
    result = classify_surface_samples(_samples(), np.full(5, 10.), cfg)
    assert not result.process_eligible.any()
    assert result.process_status.eq('boundary_uncertainty_unknown').all()


def test_positive_layers_reject_overcut_and_keep_lost_samples_as_gaps(cfg):
    cfg['process']['material_side_confirmed'] = True
    table = _samples((.4, .4, .12, .4, .4))
    table = classify_surface_samples(table, np.full(5, 10.), cfg)
    target = target_layer_table(table, .2, cfg)
    assert target.valid.tolist() == [True, True, False, True, True]
    np.testing.assert_allclose(target.loc[target.valid, 'mapped_z_mm'], .2)
    assert target.iloc[1].segment_id != target.iloc[3].segment_id
    assert target.iloc[2].segment_id == -1
    assert np.isnan(target.iloc[2].mapped_z_mm)
    # A further layer below the available nominal-CAD allowance cannot survive.
    assert not target_layer_table(table, .5, cfg).valid.any()


def _region_candidates(face, points, face_id=0, normal=(0., 0., 1.)):
    points = np.asarray(points, float)
    uv = np.asarray([face.paramAt(tuple(p)) for p in points])
    table = pd.DataFrame({'face_id': face_id, 'valid': True,
                          'process_eligible': True, 'process_status': 'allowance_candidate',
                          'u': uv[:, 0], 'v': uv[:, 1]})
    for j, axis in enumerate('xyz'):
        table[f'cad_{axis}_mm'] = points[:, j]
        table[f'normal_{axis}'] = normal[j]
    return table


def test_process_regions_do_not_connect_through_a_narrow_trim_slot(cfg):
    # Euclidean neighbors on either side of the 0.2 mm slot are only 0.4 mm
    # apart, yet there is no observed/material surface on their UV connecting line.
    face = cq.Workplane('XY').rect(10, 10).rect(.2, 8).extrude(1).faces('>Z').val()
    left = [[-.2, y, 1.] for y in [-.6, -.2, .2, .6]]
    right = [[.2, y, 1.] for y in [-.6, -.2, .2, .6]]
    table = _region_candidates(face, left+right)
    cfg['process'].update(region_connection_radius_mm=.75, minimum_region_samples=4,
                          region_normal_cosine=.85)
    result = assign_process_regions(table, SimpleNamespace(faces=[face]), cfg)
    assert result.process_eligible.all()
    assert result.region_id.iloc[:4].nunique() == 1
    assert result.region_id.iloc[4:].nunique() == 1
    assert result.region_id.iloc[0] != result.region_id.iloc[4]
