"""Known-geometry checks for scan-supported CAD contact frames and gap handling."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning.scan_mapping import (
    contact_frames, export_contact_poses, fit_local_surface,
    map_reference_paths, validate_contact_poses, validate_mapping_config,
)


@pytest.fixture
def cfg():
    return json.loads((Path(__file__).resolve().parents[1]/'cad_mapping.example.json').read_text())


def grid(radius=1.5, step=.15):
    x, y = np.meshgrid(np.arange(-radius, radius+step/2, step),
                       np.arange(-radius, radius+step/2, step))
    return np.column_stack([x.ravel(), y.ravel()])


def test_exact_quadric_normal_intersection(cfg):
    xy = grid()
    z = .2 + .04*xy[:, 0] - .07*xy[:, 1] + .08*xy[:, 0]**2 + .03*xy[:, 0]*xy[:, 1]
    fit = fit_local_surface(np.zeros(3), [0, 0, 1], np.column_stack([xy, z]), cfg)
    assert fit['valid']
    np.testing.assert_allclose(fit['mapped_point'], [0, 0, .2], atol=1e-10)
    expected = np.array([-.04, .07, 1.])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose([fit[f'scan_normal_{a}'] for a in 'xyz'], expected, atol=1e-10)
    assert fit['fit_rms_mm'] < 1e-10


def test_sparse_burrs_do_not_define_substrate_height(cfg):
    cfg['fit_degree'] = 1
    xy = grid()
    z = np.full(len(xy), .15)
    # Deterministically spread sparse raised points; the target is the substrate.
    z[np.arange(0, len(z), 19)] += .28
    fit = fit_local_surface(np.zeros(3), [0, 0, 1], np.column_stack([xy, z]), cfg)
    assert fit['valid']
    assert abs(fit['offset_mm']-.15) < .01
    assert fit['rejected_neighbor_count'] > 0


def test_parallel_layers_are_ambiguous(cfg):
    xy = grid()
    points = np.concatenate([np.column_stack([xy, np.full(len(xy), z)]) for z in [-.3, .3]])
    fit = fit_local_surface(np.zeros(3), [0, 0, 1], points, cfg)
    assert not fit['valid']
    assert fit['status'] == 'ambiguous_layers'


def test_one_sided_scan_cannot_extrapolate(cfg):
    xy = grid()
    xy = xy[xy[:, 0] > .3]
    fit = fit_local_surface(np.zeros(3), [0, 0, 1], np.column_stack([xy, np.full(len(xy), .1)]), cfg)
    assert not fit['valid']
    assert 'footprint' in fit['status']


def test_unobserved_central_hole_is_not_filled(cfg):
    xy = grid()
    xy = xy[np.linalg.norm(xy, axis=1) > .85]
    fit = fit_local_surface(np.zeros(3), [0, 0, 1], np.column_stack([xy, np.full(len(xy), .1)]), cfg)
    assert not fit['valid'], 'A surrounding convex hull must not legitimize filling an unobserved hole'


def _small_patches(xs, offsets):
    xy = grid(.18, .03)
    xy = xy[np.linalg.norm(xy, axis=1) < .19]
    return np.concatenate([np.column_stack([xy+[x, 0], np.full(len(xy), z)])
                           for x, z in zip(xs, offsets)])


def _reference(xs):
    return [{'path_id': 'test', 'points_C': np.column_stack([xs, np.zeros((len(xs), 2))]),
             'normals_C': np.tile([0., 0., 1.], (len(xs), 1)), 'face_ids': np.zeros(len(xs), int)}]


def _small_cfg(cfg):
    return dict(cfg, tangent_radius_mm=.22, search_radius_mm=.5,
                min_tangent_spread_mm=.04, fit_degree=1)


def test_missing_samples_and_isolates_never_join(cfg):
    cfg = _small_cfg(cfg)
    xs = np.arange(9)*.5
    observed = [0, .5, 1.5, 2.5, 3, 3.5, 4]
    table = map_reference_paths(_reference(xs), _small_patches(observed, [.1]*len(observed)), cfg)
    assert not table.iloc[2].valid
    assert not table.iloc[4].valid
    assert table.iloc[3].status == 'isolated_sample'
    assert table.iloc[0].segment_id != table.iloc[5].segment_id
    assert np.isnan(table.iloc[3].mapped_z_mm)


def test_contact_pose_rigid_transform_and_example_ball_center(cfg):
    n = np.array([.2, -.3, 1.])
    n /= np.linalg.norm(n)
    tangent = np.cross([0., 1., 0.], n)
    tangent /= np.linalg.norm(tangent)
    reference = np.array([10., -4., 3.])+np.arange(5)[:, None]*.5*tangent
    contact = reference+.13*n
    table = pd.DataFrame(dict(path_id='known', sample_index=np.arange(5), valid=True,
                              segment_id=0, status='accepted', support_count=50,
                              offset_mm=.13, fit_rms_mm=0.))
    for j, a in enumerate('xyz'):
        table[f'cad_{a}_mm'] = reference[:, j]
        table[f'mapped_{a}_mm'] = contact[:, j]
        table[f'normal_{a}'] = n[j]
        table[f'scan_normal_{a}'] = n[j]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', [.2, -.4, .8]).as_matrix()
    T[:3, 3] = [100., -50., 23.]
    cad, scan, TC, TS = export_contact_poses(table, T, cfg)
    expected_scan = (contact-T[:3, 3])@T[:3, :3]
    expected_normal = n@T[:3, :3]
    np.testing.assert_allclose(scan[['x_mm', 'y_mm', 'z_mm']], expected_scan, atol=1e-11)
    np.testing.assert_allclose(scan[['normal_x', 'normal_y', 'normal_z']], np.tile(expected_normal, (5, 1)), atol=1e-12)
    np.testing.assert_allclose(cad[[f'example_ball_center_{a}_mm' for a in 'xyz']], contact+2*n, atol=1e-12)
    np.testing.assert_allclose(scan[[f'example_ball_center_{a}_mm' for a in 'xyz']], expected_scan+2*expected_normal, atol=1e-11)
    np.testing.assert_allclose(scan[[f'cad_{a}_mm' for a in 'xyz']], (reference-T[:3, 3])@T[:3, :3], atol=1e-11)
    expected_cad_axes = np.column_stack([tangent, np.cross(-n, tangent), -n])
    expected_scan_axes = T[:3, :3].T@expected_cad_axes
    for j, a in enumerate('xyz'):
        np.testing.assert_allclose(cad[[f'tool_{a}_{c}' for c in 'xyz']], np.tile(expected_cad_axes[:, j], (5, 1)), atol=1e-12)
        np.testing.assert_allclose(scan[[f'tool_{a}_{c}' for c in 'xyz']], np.tile(expected_scan_axes[:, j], (5, 1)), atol=1e-12)
    np.testing.assert_allclose(T[None]@TS, TC, atol=1e-11)
    np.testing.assert_allclose(Rotation.from_quat(cad[['qx','qy','qz','qw']]).apply([0,0,1]), np.tile(-n, (5,1)), atol=1e-12)
    assert not cad.execution_allowed.any()
    assert not scan.execution_allowed.any()
    assert cad.position_reference.eq('surface_contact_point').all()
    assert validate_contact_poses(cad, scan, TC, TS, T, cfg)['passed']
    # CSV fields are deliverables too: a correct NPZ must not conceal corrupted CSV.
    for key, value in [('x_mm', 999.), ('tool_z_x', 7.),
                       ('example_ball_center_y_mm', -999.), ('x_mm', np.nan)]:
        broken = scan.copy()
        broken.loc[0, key] = value
        with pytest.raises(ValueError, match='validation failed'):
            validate_contact_poses(cad, broken, TC, TS, T, cfg)


def test_scan_normal_mode_couples_surface_position_and_axis(cfg):
    cfg = dict(cfg, normal_mode='scan', fit_degree=2)
    xy = grid(3., .12)
    z = .15+.08*xy[:, 0]+.025*xy[:, 0]**2
    xs = np.linspace(-.6, .6, 7)
    paths = _reference(xs)
    paths[0]['uv'] = np.column_stack([xs, np.zeros(len(xs))])
    table = map_reference_paths(paths, np.column_stack([xy, z]), cfg)
    assert table.valid.all()
    np.testing.assert_allclose(table.mapped_z_mm, .15+.08*xs+.025*xs**2, atol=1e-9)
    expected = np.column_stack([-(.08+.05*xs), np.zeros(len(xs)), np.ones(len(xs))])
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    np.testing.assert_allclose(table[['normal_x', 'normal_y', 'normal_z']], expected, atol=1e-9)
    np.testing.assert_allclose(table[['cad_normal_x', 'cad_normal_y', 'cad_normal_z']],
                               np.tile([0, 0, 1.], (len(xs), 1)))
    np.testing.assert_allclose(table.u, xs)
    assert table.normal_source.eq('scan_local_surface_derivative').all()
    cad, scan, TC, TS = export_contact_poses(table, np.eye(4), cfg)
    np.testing.assert_allclose(TC[:, :3, 2], -expected, atol=1e-9)
    assert validate_contact_poses(cad, scan, TC, TS, np.eye(4), cfg)['passed']


def test_external_uncertainty_is_not_inferred_from_small_fit_residual(cfg):
    xy = grid()
    neighbors = np.column_stack([xy, np.full(len(xy), .1)])
    missing = fit_local_surface(np.zeros(3), [0, 0, 1], neighbors,
                                dict(cfg, max_position_uncertainty_mm=.2))
    assert missing['status'] == 'uncertainty_unavailable'
    assert not missing['uncertainty_complete']
    assert np.isnan(missing['position_uncertainty_mm'])
    measured = dict(cfg, scan_sigma_mm=.03, registration_sigma_mm=.04,
                    uncertainty_multiplier=2., max_position_uncertainty_mm=.11)
    fit = fit_local_surface(np.zeros(3), [0, 0, 1], neighbors, measured)
    assert fit['valid'] and fit['uncertainty_complete']
    assert fit['position_uncertainty_mm'] == pytest.approx(.1)
    limited = fit_local_surface(np.zeros(3), [0, 0, 1], neighbors,
                                dict(measured, max_position_uncertainty_mm=.09))
    assert limited['status'] == 'position_uncertainty_limit'


def test_face_filtered_support_excludes_parallel_other_face(cfg):
    xy = grid(3., .15)
    cloud = np.concatenate([np.column_stack([xy, np.full(len(xy), z)]) for z in [.1, -.5]])
    labels = np.r_[np.zeros(len(xy), dtype=int), np.ones(len(xy), dtype=int)]
    path = _reference(np.linspace(-.5, .5, 5))
    unfiltered = map_reference_paths(path, cloud, cfg)
    assert not unfiltered.valid.any()
    table = map_reference_paths(path, cloud, cfg, support_face_ids=labels)
    assert table.valid.all() and table.face_support_filtered.all()
    np.testing.assert_allclose(table.mapped_z_mm, .1, atol=1e-10)
    unknown = map_reference_paths(path, cloud, cfg, support_face_ids=np.full(len(cloud), -1))
    assert not unknown.valid.any()
    with pytest.raises(ValueError, match='support_face_ids'):
        map_reference_paths(path, cloud, cfg, support_face_ids=labels[:-1])


def test_pose_splits_and_rejected_samples_are_synchronized(cfg):
    points = np.array([[0., 0., .1], [.5, 0., .1], [1., 0., .1],
                       [1., .5, .1], [1., 1., .1], [1., 1.5, .1]])
    table = pd.DataFrame(dict(path_id='corner', sample_index=np.arange(len(points)),
                              valid=True, segment_id=0, status='accepted'))
    for j, axis in enumerate('xyz'):
        table[f'mapped_{axis}_mm'] = points[:, j]
        table[f'cad_{axis}_mm'] = points[:, j]
        table[f'normal_{axis}'] = float(j == 2)
    cad, scan, TC, TS = export_contact_poses(table, np.eye(4), cfg)
    assert len(cad) > 0
    diagnostics = table[table.valid].reset_index(drop=True)
    assert diagnostics[['path_id', 'sample_index', 'segment_id']].equals(
        cad[['path_id', 'sample_index', 'segment_id']])
    assert cad.segment_id.nunique() >= 2
    assert validate_contact_poses(cad, scan, TC, TS, np.eye(4), cfg)['passed']
    # Re-export is idempotent even after endpoint tangent recomputation.
    cad2, _, TC2, _ = export_contact_poses(table, np.eye(4), cfg)
    np.testing.assert_allclose(TC, TC2)
    assert cad.sample_index.equals(cad2.sample_index)


def test_principal_curvature_of_paraboloid_and_rigid_rotation(cfg):
    xy = grid()
    kx, ky = .18, -.11
    z = .12+.5*kx*xy[:, 0]**2+.5*ky*xy[:, 1]**2
    neighbors = np.column_stack([xy, z])
    fit = fit_local_surface(np.zeros(3), [0, 0, 1], neighbors, cfg)
    assert fit['valid']
    assert fit['principal_curvature_min_per_mm'] == pytest.approx(ky, abs=1e-10)
    assert fit['principal_curvature_max_per_mm'] == pytest.approx(kx, abs=1e-10)
    assert fit['max_abs_principal_curvature_per_mm'] == pytest.approx(kx, abs=1e-10)
    rotation = Rotation.from_euler('xyz', [.7, -.4, 1.2]).as_matrix()
    origin = np.array([10., 20., -4.])
    rotated = fit_local_surface(origin, rotation@np.array([0., 0., 1.]),
                                neighbors@rotation.T+origin, cfg)
    assert rotated['valid']
    assert rotated['max_abs_principal_curvature_per_mm'] == pytest.approx(kx, abs=1e-10)
    assert rotated['principal_curvature_min_per_mm'] == pytest.approx(ky, abs=1e-10)
