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
    if cfg.get('normal_mode', 'cad') not in ('cad', 'scan'):
        raise ValueError('normal_mode must be cad or scan')
    for key in ('scan_sigma_mm', 'registration_sigma_mm'):
        value = cfg.get(key)
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f'{key} must be None or finite and nonnegative')
    for key in ('uncertainty_multiplier', 'max_position_uncertainty_mm'):
        value = cfg.get(key, 2.) if key == 'uncertainty_multiplier' else cfg.get(key)
        if key == 'uncertainty_multiplier' and value is None:
            raise ValueError('uncertainty_multiplier must be finite and positive')
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f'{key} must be None or finite and positive')
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
    filling. The fitted height and its first derivatives define the same local
    surface; normal_mode selects CAD prior or this scan-surface normal downstream.
    Conditional uncertainty uses user-supplied independent scalar standard
    uncertainties along the CAD-normal line, not ICP residuals or a physical
    accuracy certification. Orientation/lever-arm covariance and model bias are
    outside this scalar conditional budget.
    """
    n = unit(normal)
    x = tool_frame(n)[:, 0]
    basis = np.column_stack([x, np.cross(n, x), n])
    local = (np.asarray(neighbors)-point) @ basis
    local = local[(np.linalg.norm(local[:, :2], axis=1) <= cfg['tangent_radius_mm']) &
                  (np.abs(local[:, 2]) <= cfg['max_normal_offset_mm'])]
    result = {'valid': False, 'status': 'insufficient_support', 'support_count': len(local),
              'uncertainty_complete': False, 'fit_prediction_sigma_mm': np.nan,
              'conditional_position_sigma_mm': np.nan, 'position_uncertainty_mm': np.nan}
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
    # Re-solve with the final robust weights so coefficient/derivative diagnostics
    # and their residuals describe one and the same fitted surface.
    aw = a*np.sqrt(weights[:, None])
    coef = np.linalg.lstsq(aw, local[:, 2]*np.sqrt(weights), rcond=None)[0]
    residual = local[:, 2]-a@coef
    inlier = np.abs(residual) <= max(.06, 3*scale)
    rms = float(np.sqrt(np.mean(residual[inlier]**2))) if inlier.any() else np.inf
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
    # Local regression prediction uncertainty is conditional on this height model.
    # Correlated scan/systematic errors are not reduced by the support-point count;
    # their externally determined standard uncertainties enter the RSS separately.
    effective_count = weights[inlier].sum()**2/np.sum(weights[inlier]**2)
    dof = max(effective_count-a.shape[1], 1.)
    variance = float(np.sum(weights[inlier]*residual[inlier]**2)/dof)
    information = a[inlier].T@(weights[inlier, None]*a[inlier])
    prediction_sigma = float(np.sqrt(max(0., variance*np.linalg.pinv(information)[0, 0])))
    result['fit_prediction_sigma_mm'] = prediction_sigma
    result['fit_effective_support_count'] = float(effective_count)
    scan_sigma, registration_sigma = cfg.get('scan_sigma_mm'), cfg.get('registration_sigma_mm')
    complete = scan_sigma is not None and registration_sigma is not None
    result['uncertainty_complete'] = complete
    result['uncertainty_model'] = 'normal_line_conditional_RSS_independent_scan_registration_local_fit'
    result['scan_sigma_mm'] = np.nan if scan_sigma is None else float(scan_sigma)
    result['registration_sigma_mm'] = np.nan if registration_sigma is None else float(registration_sigma)
    if complete:
        sigma = float(np.sqrt(scan_sigma**2+registration_sigma**2+prediction_sigma**2))
        result['conditional_position_sigma_mm'] = sigma
        result['position_uncertainty_mm'] = float(cfg.get('uncertainty_multiplier', 2.)*sigma)
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
    elif cfg.get('max_position_uncertainty_mm') is not None and not complete:
        result['status'] = 'uncertainty_unavailable'
    elif (cfg.get('max_position_uncertainty_mm') is not None and
          result['position_uncertainty_mm'] > cfg['max_position_uncertainty_mm']):
        result['status'] = 'position_uncertainty_limit'
    else:
        result.update(valid=True, status='accepted', mapped_point=point+coef[0]*n)
    return result


def map_reference_paths(paths, scan_points_C, cfg, support_face_ids=None):
    """Map CAD samples using bounded support, optionally from the same CAD face.

    ``support_face_ids`` is aligned row-for-row with scan_points_C. Unknown IDs
    (-1) and other faces never supply fitted support when labels are provided.
    Returned diagnostics already use final full-pose segmentation.
    """
    validate_mapping_config(cfg)
    points = np.asarray(scan_points_C, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('scan_points_C must be finite Nx3')
    face_labels = None if support_face_ids is None else np.asarray(support_face_ids)
    if face_labels is not None and (face_labels.shape != (len(points),) or
                                    not np.issubdtype(face_labels.dtype, np.integer)):
        raise ValueError('support_face_ids must be an integer vector aligned with scan points')
    tree = cKDTree(points)
    rows = []
    segment = -1
    for path in paths:
        p = np.asarray(path['points_C'], dtype=float)
        normals = np.asarray(path['normals_C'], dtype=float)
        faces = np.asarray(path['face_ids'])
        if (p.ndim != 2 or p.shape[1] != 3 or normals.shape != p.shape or
                faces.shape != (len(p),) or not np.issubdtype(faces.dtype, np.integer) or
                not np.isfinite(p).all() or not np.isfinite(normals).all() or (faces < 0).any()):
            raise ValueError('Path points, normals and nonnegative face_ids must be aligned finite arrays')
        uv = None if 'uv' not in path else np.asarray(path['uv'], dtype=float)
        if uv is not None and (uv.shape != (len(p), 2) or not np.isfinite(uv).all()):
            raise ValueError('Path uv must be finite Nx2 aligned with points')
        previous = None
        for i, (point, normal, face_id) in enumerate(zip(p, normals, faces)):
            normal = unit(normal)
            ids = np.asarray(tree.query_ball_point(point, cfg['search_radius_mm']), dtype=int)
            if face_labels is not None:
                ids = ids[face_labels[ids] == face_id]
            fit = fit_local_surface(point, normal, points[ids], cfg)
            used_normal = normal
            if fit['valid'] and cfg.get('normal_mode', 'cad') == 'scan':
                used_normal = np.array([fit[f'scan_normal_{a}'] for a in 'xyz'])
            row = {'path_id': path['path_id'], 'sample_index': i, 'face_id': int(face_id),
                   **dict(zip(['cad_x_mm', 'cad_y_mm', 'cad_z_mm'], point)),
                   **dict(zip(['cad_normal_x', 'cad_normal_y', 'cad_normal_z'], normal)),
                   **dict(zip(['normal_x', 'normal_y', 'normal_z'], used_normal)),
                   **dict(zip(['tool_z_x', 'tool_z_y', 'tool_z_z'], -used_normal)),
                   'normal_source': 'scan_local_surface_derivative' if cfg.get('normal_mode', 'cad') == 'scan' else 'STEP_face_transferred_rigidly',
                   'face_support_filtered': face_labels is not None,
                   **{k: v for k, v in fit.items() if k != 'mapped_point'},
                   'segment_id': -1, 'mapped_x_mm': np.nan, 'mapped_y_mm': np.nan, 'mapped_z_mm': np.nan}
            if uv is not None:
                row.update(u=float(uv[i, 0]), v=float(uv[i, 1]))
            for key in ('region_id', 'pass_id', 'raster_id'):
                if key in path:
                    value = path[key]
                    row[key] = value[i] if isinstance(value, np.ndarray) else value
            if fit['valid']:
                mapped = fit['mapped_point']
                boundary = previous is None
                if previous is not None:
                    angle = np.degrees(np.arccos(np.clip(used_normal@previous['normal'], -1, 1)))
                    boundary = (np.linalg.norm(mapped-previous['point']) > cfg['max_point_jump_mm'] or
                                abs(fit['offset_mm']-previous['offset']) > cfg['max_offset_step_mm'] or
                                angle > cfg['max_pose_step_deg'])
                if boundary:
                    segment += 1
                row.update(dict(zip(['mapped_x_mm', 'mapped_y_mm', 'mapped_z_mm'], mapped)))
                row['segment_id'] = segment
                previous = {'normal': used_normal, 'point': mapped, 'offset': fit['offset_mm']}
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
    return finalize_pose_segments(table, cfg)


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


def finalize_pose_segments(table, cfg):
    """Compute frames and split diagnostics at complete-pose discontinuities.

    Frames are recomputed after each split, since endpoint tangents change with
    the segment. Undefined tangents and singleton segments remain rejected rows.
    This is shared by mapping, CSV export and visualizations via table axes.
    """
    result = table.copy()
    frame_columns = [f'tool_{axis}_{component}' for axis in 'xyz' for component in 'xyz']
    for column in frame_columns+['qx', 'qy', 'qz', 'qw']:
        result[column] = np.nan
    next_segment = 0
    # Keep gaps/path changes even if a caller accidentally reused segment IDs.
    initial, current, previous = [], [], None
    for index, row in result.iterrows():
        good = bool(row.valid)
        connected = (previous is not None and row.path_id == previous.path_id and
                     row.segment_id == previous.segment_id and
                     row.sample_index == previous.sample_index+1)
        if current and (not good or not connected):
            initial.append(current)
            current = []
        if good:
            current.append(index)
        previous = row if good else None
    if current:
        initial.append(current)
    pending = list(reversed(initial))
    while pending:
        ids = pending.pop()
        if len(ids) < 2:
            result.loc[ids, ['valid', 'status', 'segment_id']] = [False, 'isolated_sample', -1]
            continue
        points = result.loc[ids, ['mapped_x_mm', 'mapped_y_mm', 'mapped_z_mm']].to_numpy(float)
        normals = result.loc[ids, ['normal_x', 'normal_y', 'normal_z']].to_numpy(float)
        tangent = np.gradient(points, axis=0)
        tangent -= np.einsum('ij,ij->i', tangent, normals)[:, None]*normals
        defined = np.isfinite(tangent).all(axis=1) & (np.linalg.norm(tangent, axis=1) > 1e-9)
        if not defined.all():
            bad = np.asarray(ids)[~defined]
            result.loc[bad, ['valid', 'status', 'segment_id']] = [False, 'undefined_contact_tangent', -1]
            starts = np.flatnonzero(defined & ~np.r_[False, defined[:-1]])
            ends = np.flatnonzero(defined & ~np.r_[defined[1:], False])+1
            pending.extend(reversed([ids[a:b] for a, b in zip(starts, ends)]))
            continue
        frames, quat = contact_frames(points, normals)
        rotations = Rotation.from_matrix(frames)
        change = np.degrees((rotations[:-1].inv()*rotations[1:]).magnitude())
        cuts = np.flatnonzero(change > cfg['max_pose_step_deg']+1e-10)+1
        if len(cuts):
            bounds = np.r_[0, cuts, len(ids)]
            pending.extend(reversed([ids[a:b] for a, b in zip(bounds[:-1], bounds[1:])]))
            continue
        result.loc[ids, 'segment_id'] = next_segment
        next_segment += 1
        for axis_index, axis in enumerate('xyz'):
            result.loc[ids, [f'tool_{axis}_{a}' for a in 'xyz']] = frames[:, :, axis_index]
        result.loc[ids, ['qx', 'qy', 'qz', 'qw']] = quat
    result.loc[~result.valid.astype(bool), ['mapped_x_mm', 'mapped_y_mm', 'mapped_z_mm']] = np.nan
    return result


def export_contact_poses(table, T_scan_to_CAD, cfg):
    """Return accepted contact-frame poses in CAD C and saved-scan S coordinates.

    Updates input ``table`` validity, segments and frame columns in-place, so the
    all-sample diagnostic CSV agrees with exported poses even for legacy callers.
    Empty accepted results yield schema-preserving CSV tables and (0,4,4) arrays.
    """
    assert_rigid(T_scan_to_CAD)
    finalized = finalize_pose_segments(table, cfg)
    for column in finalized:
        table[column] = finalized[column]
    cad = table.loc[table.valid].copy().reset_index(drop=True)
    R = np.empty((len(cad), 3, 3))
    for j, axis in enumerate('xyz'):
        R[:, :, j] = cad[[f'tool_{axis}_{a}' for a in 'xyz']].to_numpy(float)
    cad['frame'] = 'CAD'
    cad['position_reference'] = 'surface_contact_point'
    if 'normal_source' not in cad:
        cad['normal_source'] = 'STEP_face_transferred_rigidly'
    cad['normal_side_confirmed'] = False
    cad['execution_allowed'] = False
    cad['example_ball_radius_mm'] = cfg['example_ball_radius_mm']
    for j, axis in enumerate('xyz'):
        cad[f'{axis}_mm'] = cad[f'mapped_{axis}_mm']
        for k, component in enumerate('xyz'):
            cad[f'tool_{axis}_{component}'] = R[:, k, j]
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
                                     ('cad_normal_','',False), ('scan_normal_','',False), ('tool_x_','',False),
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
    if (not len(cad) and not len(scan) and T_C.shape == (0, 4, 4) and T_S.shape == (0, 4, 4)):
        return dict(passed=True, empty=True, accepted_pose_count=0, physical_accuracy_verified=False)
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
    return dict(passed=True, empty=False, accepted_pose_count=len(cad), **errors, max_segment_step_mm=maxstep,
                max_segment_orientation_step_deg=maxangle, physical_accuracy_verified=False)
