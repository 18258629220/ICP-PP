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


def test_insufficient_scan_support(cfg):
    fit = fit_local_surface(np.zeros(3), [0, 0, 1], np.array([[0, 0, .1], [.1, 0, .1]]), cfg)
    assert not fit['valid']
    assert fit['status'] == 'insufficient_support'


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


def test_offset_discontinuity_starts_new_segment(cfg):
    cfg = _small_cfg(cfg)
    xs = np.arange(6)*.5
    table = map_reference_paths(_reference(xs), _small_patches(xs, [-.3]*3+[.3]*3), cfg)
    assert table.valid.all()
    assert table.segment_id.nunique() == 2
    assert table.iloc[2].segment_id != table.iloc[3].segment_id
    np.testing.assert_allclose(table.offset_mm, [-.3]*3+[.3]*3, atol=1e-10)


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


def test_curved_cad_normals_produce_changing_right_handed_frames():
    angle = np.linspace(-.5, .5, 31)
    # Cylinder radius 20, traversed circumferentially: analytic outward normals.
    normal = np.column_stack([np.sin(angle), np.zeros(len(angle)), np.cos(angle)])
    points = 20*normal
    frames, quat = contact_frames(points, normal)
    np.testing.assert_allclose(frames[:, :, 2], -normal, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(frames), 1., atol=1e-12)
    np.testing.assert_allclose(np.swapaxes(frames, 1, 2)@frames, np.tile(np.eye(3), (len(angle), 1, 1)), atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(quat).as_matrix(), frames, atol=1e-12)
    assert np.all(np.sum(quat[:-1]*quat[1:], axis=1) >= 0)
    assert (Rotation.from_matrix(frames[0]).inv()*Rotation.from_matrix(frames[-1])).magnitude() > .9


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


@pytest.mark.parametrize('key,value', [('max_normal_offset_mm', 0), ('path_sampling_mm', np.nan),
                                      ('fit_degree', 3), ('min_support_points', 3),
                                      ('secondary_layer_fraction', 1), ('normal_mode', 'scan')])
def test_invalid_mapping_configuration_rejected(cfg, key, value):
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_mapping_config(cfg)


def test_undefined_contact_tangent_rejected():
    with pytest.raises(ValueError, match='undefined tangent'):
        contact_frames(np.zeros((3, 3)), np.tile([0, 0, 1.], (3, 1)))
