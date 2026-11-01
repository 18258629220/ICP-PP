"""Bounded CAD-normal scan projection and contact-frame poses (mm, XYZW).

This module estimates the as-is substrate near a prescribed CAD curve. It does
not estimate removal depth, robot calibration, tool engagement or accessibility.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull, QhullError
from scipy.spatial.transform import Rotation
from planning.geometry import tool_frame, unit
from pipeline_common import assert_rigid


def validate_mapping_config(cfg):
    positive = ('path_sampling_mm', 'search_radius_mm', 'tangent_radius_mm',
                'max_normal_offset_mm', 'layer_gap_mm', 'max_fit_rms_mm',
                'min_tangent_spread_mm', 'max_point_jump_mm', 'max_offset_step_mm',
                'max_normal_change_deg', 'max_pose_step_deg', 'example_ball_radius_mm',
                'max_support_gap_mm')
    for key in positive:
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if cfg['min_support_points'] < 8 or cfg['fit_degree'] not in (1, 2):
        raise ValueError('Require >=8 support points and fit_degree 1 or 2')
    if not 0 < cfg['secondary_layer_fraction'] < 1:
        raise ValueError('secondary_layer_fraction must be between zero and one')
    if cfg.get('normal_mode', 'cad') != 'cad':
        raise ValueError('This implementation exports transferred CAD normals; normal_mode must be cad')
    unit(cfg['approach_outward_C'])


def _design(xy, degree):
    x, y = xy.T
    a = np.column_stack([np.ones(len(x)), x, y])
    if degree == 2:
        a = np.column_stack([a, x*x, x*y, y*y])
    return a


def fit_local_surface(point, normal, neighbors, cfg):
    """Intersect a robust local height surface with the reference normal line.

    Reject ambiguous layers and extrapolation. No nearest-point snapping or hole
    filling. Fitted scan normal is diagnostic; output tool axis uses CAD prior.
    """
    n = unit(normal)
    x = tool_frame(n)[:, 0]
    basis = np.column_stack([x, np.cross(n, x), n])
    local = (np.asarray(neighbors)-point) @ basis
    local = local[(np.linalg.norm(local[:, :2], axis=1) <= cfg['tangent_radius_mm']) &
                  (np.abs(local[:, 2]) <= cfg['max_normal_offset_mm'])]
    result = {'valid': False, 'status': 'insufficient_support', 'support_count': len(local)}
    if len(local) < cfg['min_support_points']:
        return result
    # Parallel nearby walls cannot be disambiguated by unconstrained NN lookup.
    heights = np.sort(local[:, 2])
    groups = np.split(heights, np.flatnonzero(np.diff(heights) > cfg['layer_gap_mm'])+1)
    counts = sorted([len(g) for g in groups], reverse=True)
    if len(counts) > 1 and counts[1] >= max(3, counts[0]*cfg['secondary_layer_fraction']):
        result['status'] = 'ambiguous_layers'
        return result
    main = max(groups, key=len)
    local = local[(local[:, 2] >= main[0]-1e-9) & (local[:, 2] <= main[-1]+1e-9)]
    result['support_count'] = len(local)
    if len(local) < cfg['min_support_points']:
        return result
    xy = local[:, :2]
    spread = np.sqrt(np.linalg.eigvalsh(np.cov(xy.T)))
    if spread[0] < cfg['min_tangent_spread_mm']:
        result['status'] = 'degenerate_support'
        return result
    try:
        hull = ConvexHull(xy)
    except QhullError:
        result['status'] = 'degenerate_support'
        return result
    # Origin must lie within the observed footprint; avoid extrapolating over edges.
    if np.max(hull.equations[:, -1]) > 1e-8:
        result['status'] = 'outside_support_footprint'
        return result
    a = _design(xy, cfg['fit_degree'])
    spatial_weights = np.exp(-np.sum(xy*xy, axis=1)/cfg['tangent_radius_mm']**2)
    weights = spatial_weights.copy()
    for _ in range(6):
        aw = a*np.sqrt(weights[:, None])
        if np.linalg.matrix_rank(aw) < a.shape[1] or np.linalg.cond(aw) > 1e5:
            result['status'] = 'ill_conditioned_fit'
            return result
        coef = np.linalg.lstsq(aw, local[:, 2]*np.sqrt(weights), rcond=None)[0]
        residual = local[:, 2]-a@coef
        scale = max(.02, 1.4826*np.median(np.abs(residual-np.median(residual))))
        weights = spatial_weights*np.minimum(1, 1.5*scale/np.maximum(np.abs(residual), 1e-12))
    inlier = np.abs(residual) <= max(.06, 3*scale)
    rms = float(np.sqrt(np.mean(residual[inlier]**2)))
    result.update(fit_rms_mm=rms, inlier_count=int(inlier.sum()),
                  offset_mm=float(coef[0]), rejected_neighbor_count=int((~inlier).sum()))
    if inlier.sum() < cfg['min_support_points']:
        result['status'] = 'insufficient_inliers'
        return result
    result['nearest_inlier_tangent_mm'] = float(np.min(np.linalg.norm(xy[inlier], axis=1)))
    if result['nearest_inlier_tangent_mm'] > cfg['max_support_gap_mm']:
        result['status'] = 'unobserved_contact_gap'
        return result
    # Recheck footprint with inliers, so a burr cannot supply the missing support.
    try:
        if np.max(ConvexHull(xy[inlier]).equations[:, -1]) > 1e-8:
            result['status'] = 'outside_inlier_footprint'
            return result
    except QhullError:
        result['status'] = 'degenerate_inliers'
        return result
    fit_normal = unit(basis @ np.array([-coef[1], -coef[2], 1.]))
    angle = float(np.degrees(np.arccos(np.clip(fit_normal@n, -1, 1))))
    result.update(scan_normal_x=fit_normal[0], scan_normal_y=fit_normal[1],
                  scan_normal_z=fit_normal[2], scan_normal_difference_deg=angle)
    if abs(coef[0]) > cfg['max_normal_offset_mm']:
        result['status'] = 'offset_limit'
    elif rms > cfg['max_fit_rms_mm']:
        result['status'] = 'fit_noise_limit'
    elif angle > cfg['max_normal_change_deg']:
        result['status'] = 'normal_disagreement'
    else:
        result.update(valid=True, status='accepted', mapped_point=point+coef[0]*n)
    return result


def map_reference_paths(paths, scan_points_C, cfg):
    validate_mapping_config(cfg)
    points = np.asarray(scan_points_C, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('scan_points_C must be finite Nx3')
    tree = cKDTree(points)
    rows = []
    segment = -1
    for path in paths:
        previous = None
        for i, (point, normal, face_id) in enumerate(zip(path['points_C'], path['normals_C'], path['face_ids'])):
            ids = tree.query_ball_point(point, cfg['search_radius_mm'])
            fit = fit_local_surface(point, normal, points[ids], cfg)
            row = {'path_id': path['path_id'], 'sample_index': i, 'face_id': int(face_id),
                   **dict(zip(['cad_x_mm', 'cad_y_mm', 'cad_z_mm'], point)),
                   **dict(zip(['normal_x', 'normal_y', 'normal_z'], unit(normal))),
                   **dict(zip(['tool_z_x', 'tool_z_y', 'tool_z_z'], -unit(normal))),
                   **{k: v for k, v in fit.items() if k != 'mapped_point'},
                   'segment_id': -1, 'mapped_x_mm': np.nan, 'mapped_y_mm': np.nan, 'mapped_z_mm': np.nan}
            if fit['valid']:
                mapped = fit['mapped_point']
                boundary = previous is None
                if previous is not None:
                    angle = np.degrees(np.arccos(np.clip(normal@previous['normal'], -1, 1)))
                    boundary = (np.linalg.norm(mapped-previous['point']) > cfg['max_point_jump_mm'] or
                                abs(fit['offset_mm']-previous['offset']) > cfg['max_offset_step_mm'] or
                                angle > cfg['max_pose_step_deg'])
                if boundary:
                    segment += 1
                row.update(dict(zip(['mapped_x_mm', 'mapped_y_mm', 'mapped_z_mm'], mapped)))
                row['segment_id'] = segment
                previous = {'normal': normal, 'point': mapped, 'offset': fit['offset_mm']}
            else:
                previous = None
            rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError('No reference path samples')
    # Isolated accepted points cannot define an oriented machining stroke.
    counts = table.loc[table.valid].groupby('segment_id').size()
    singleton = table.segment_id.isin(counts[counts < 2].index)
    table.loc[singleton, ['valid', 'status', 'segment_id']] = [False, 'isolated_sample', -1]
    table.loc[singleton, ['mapped_x_mm', 'mapped_y_mm', 'mapped_z_mm']] = np.nan
    return table


def contact_frames(points, normals):
    """Right-handed frames: +Z into substrate, +X forward projected on tangent.

    Full poses are segmented at discontinuities by the caller. Quaternion signs
    are continuous within a segment; sign is not an orientation change.
    """
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    if len(points) < 2:
        raise ValueError('A frame sequence requires at least two points')
    tangent = np.gradient(points, axis=0)
    if np.any(np.linalg.norm(tangent, axis=1) < 1e-9):
        raise ValueError('Repeated or reversing samples give an undefined tangent')
    frames = np.array([tool_frame(n, t) for n, t in zip(normals, tangent)])
    quat = Rotation.from_matrix(frames).as_quat()
    for i in range(1, len(quat)):
        if quat[i]@quat[i-1] < 0:
            quat[i] *= -1
    return frames, quat


def export_contact_poses(table, T_scan_to_CAD, cfg):
    """Return accepted contact-frame poses in CAD C and saved-scan S coordinates."""
    assert_rigid(T_scan_to_CAD)
    cad = table.loc[table.valid].copy().reset_index(drop=True)
    if cad.empty:
        raise ValueError('No supported path segments; inspect diagnostics')
    R = np.empty((len(cad), 3, 3))
    Q = np.empty((len(cad), 4))
    # Split further if full tool orientation has a large step despite normal gating.
    for _, ids in cad.groupby('segment_id', sort=False).groups.items():
        ids = np.asarray(ids)
        frames, quat = contact_frames(cad.loc[ids, ['mapped_x_mm','mapped_y_mm','mapped_z_mm']].to_numpy(),
                                      cad.loc[ids, ['normal_x','normal_y','normal_z']].to_numpy())
        R[ids], Q[ids] = frames, quat
    next_segment = -1
    for _, ids in cad.groupby('segment_id', sort=False).groups.items():
        previous = None
        for i in ids:
            if previous is None or np.degrees((Rotation.from_matrix(R[previous]).inv()*Rotation.from_matrix(R[i])).magnitude()) > cfg['max_pose_step_deg']:
                next_segment += 1
            cad.loc[i, 'segment_id'] = next_segment
            previous = i
    # Pose discontinuities can isolate a sample: do not silently connect it.
    counts = cad.groupby('segment_id').size()
    keep = ~cad.segment_id.isin(counts[counts < 2].index).to_numpy()
    cad, R, Q = cad.loc[keep].reset_index(drop=True), R[keep], Q[keep]
    if cad.empty:
        raise ValueError('No continuous full-pose segments')
    cad['frame'] = 'CAD'
    cad['position_reference'] = 'surface_contact_point'
    cad['normal_source'] = 'STEP_face_transferred_rigidly'
    cad['normal_side_confirmed'] = False
    cad['execution_allowed'] = False
    cad['example_ball_radius_mm'] = cfg['example_ball_radius_mm']
    for j, axis in enumerate('xyz'):
        cad[f'{axis}_mm'] = cad[f'mapped_{axis}_mm']
        for k, component in enumerate('xyz'):
            cad[f'tool_{axis}_{component}'] = R[:, k, j]
    for j, key in enumerate(['qx','qy','qz','qw']):
        cad[key] = Q[:, j]
    p = cad[['x_mm','y_mm','z_mm']].to_numpy()
    normals = cad[['normal_x','normal_y','normal_z']].to_numpy()
    center = p+cfg['example_ball_radius_mm']*normals
    for j, axis in enumerate('xyz'):
        cad[f'example_ball_center_{axis}_mm'] = center[:, j]
    T_C = np.tile(np.eye(4), (len(cad),1,1))
    T_C[:, :3, :3], T_C[:, :3, 3] = R, p
    inv = np.linalg.inv(T_scan_to_CAD)
    T_S = inv[None] @ T_C
    scan = cad.copy()
    scan['frame'] = 'SCAN_INPUT'
    # Every vector/position column is transformed consistently, including diagnostics.
    for prefix, suffix, is_point in [('','_mm',True), ('cad_','_mm',True), ('mapped_','_mm',True),
                                     ('example_ball_center_','_mm',True), ('normal_','',False),
                                     ('scan_normal_','',False), ('tool_x_','',False),
                                     ('tool_y_','',False), ('tool_z_','',False)]:
        keys = [f'{prefix}{a}{suffix}' for a in 'xyz']
        if not all(k in scan for k in keys):
            continue
        v = cad[keys].to_numpy() @ inv[:3,:3].T
        if is_point:
            v += inv[:3,3]
        scan[keys] = v
    for _, ids in scan.groupby('segment_id', sort=False).groups.items():
        q = Rotation.from_matrix(T_S[ids, :3, :3]).as_quat()
        for j in range(1,len(q)):
            if q[j]@q[j-1] < 0:
                q[j] *= -1
        scan.loc[ids, ['qx','qy','qz','qw']] = q
    return cad, scan, T_C, T_S


def validate_contact_poses(cad, scan, T_C, T_S, transform, cfg):
    if not len(cad) or len(cad) != len(scan) or len(cad) != len(T_C) or T_C.shape != T_S.shape or not np.isfinite(T_C).all() or not np.isfinite(T_S).all():
        raise ValueError('Empty or nonfinite poses')
    if not cad[['path_id','sample_index','segment_id']].equals(scan[['path_id','sample_index','segment_id']]):
        raise ValueError('CAD and scan pose identifiers differ')
    orth = np.max(np.abs(np.swapaxes(T_C[:,:3,:3],1,2)@T_C[:,:3,:3]-np.eye(3)))
    roundtrip = np.max(np.abs(transform[None]@T_S-T_C))
    qerror = np.max(np.abs(Rotation.from_quat(cad[['qx','qy','qz','qw']]).as_matrix()-T_C[:,:3,:3]))
    sqerror = np.max(np.abs(Rotation.from_quat(scan[['qx','qy','qz','qw']]).as_matrix()-T_S[:,:3,:3]))
    normals = cad[['normal_x','normal_y','normal_z']].to_numpy()
    axis_error = np.max(np.abs(T_C[:,:3,2]+normals))
    det_error = np.max(np.abs(np.linalg.det(T_C[:,:3,:3])-1))
    center = cad[[f'example_ball_center_{a}_mm' for a in 'xyz']].to_numpy()
    center_error = np.max(np.abs(center-T_C[:,:3,3]-cfg['example_ball_radius_mm']*normals))
    maxstep, maxangle = 0., 0.
    for _, ids in cad.groupby('segment_id', sort=False).groups.items():
        p = T_C[ids,:3,3]
        maxstep = max(maxstep,float(np.linalg.norm(np.diff(p,axis=0),axis=1).max()))
        r = Rotation.from_matrix(T_C[ids,:3,:3])
        maxangle = max(maxangle,float(np.degrees((r[:-1].inv()*r[1:]).magnitude()).max()))
    errors = dict(rotation_orthogonality_error=float(orth), transform_roundtrip_max_mm=float(roundtrip),
                  cad_quaternion_matrix_error=float(qerror), scan_quaternion_matrix_error=float(sqerror),
                  axis_normal_error=float(axis_error), rotation_determinant_error=float(det_error),
                  example_center_error_mm=float(center_error))
    for name, frame, matrices in [('cad',cad,T_C),('scan',scan,T_S)]:
        errors[f'{name}_csv_position_error_mm'] = float(np.max(np.abs(frame[['x_mm','y_mm','z_mm']].to_numpy()-matrices[:,:3,3])))
        for j,axis in enumerate('xyz'):
            values = frame[[f'tool_{axis}_{a}' for a in 'xyz']].to_numpy()
            errors[f'{name}_csv_axis_{axis}_error'] = float(np.max(np.abs(values-matrices[:,:3,j])))
        values = frame[[f'example_ball_center_{a}_mm' for a in 'xyz']].to_numpy()
        n = frame[[f'normal_{a}' for a in 'xyz']].to_numpy()
        errors[f'{name}_csv_center_error_mm'] = float(np.max(np.abs(values-matrices[:,:3,3]-cfg['example_ball_radius_mm']*n)))
        errors[f'{name}_normal_direction_error'] = float(np.max(np.abs(n+matrices[:,:3,2])))
    if not np.isfinite(list(errors.values())).all() or max(errors.values()) > 1e-8 or maxstep > cfg['max_point_jump_mm']+1e-8 or maxangle > cfg['max_pose_step_deg']+1e-8:
        raise ValueError(f'Pose validation failed: {errors}, step={maxstep}, angle={maxangle}')
    if cad.execution_allowed.any() or scan.execution_allowed.any():
        raise ValueError('Unexpected execution authorization')
    return dict(passed=True, **errors, max_segment_step_mm=maxstep,
                max_segment_orientation_step_deg=maxangle, physical_accuracy_verified=False)
