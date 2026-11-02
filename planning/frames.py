"""CAD geometry -> user-confirmed robot base coordinates (mm, XYZW).

The original scan already uses ROBOT_BASE. Its XYZ must never be transformed
again for the base-frame deliverable. ``T_BASE_to_CAD`` is the existing
registration, and its inverse locates CAD in the robot base frame.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from pipeline_common import ROOT, assert_rigid, dump_json, read_ply, sha256, transform_points

BASE_FRAME = 'ROBOT_BASE'
_AXES = 'xyz'
_QUATERNION = ['qx', 'qy', 'qz', 'qw']


def cad_to_base_transform(T_BASE_to_CAD):
    """Return CAD -> BASE without registering or scaling either geometry."""
    matrix = np.asarray(T_BASE_to_CAD, dtype=float)
    assert_rigid(matrix)
    inverse = np.eye(4)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def _axes(table):
    return np.stack([table[[f'tool_{a}_{c}' for c in _AXES]].to_numpy(float)
                     for a in _AXES], axis=2)


def transform_frame_table(table, T_target_from_source, target_frame=BASE_FRAME,
                          diagnostics=True):
    """Transform positions, vectors and full orientations together.

    UV parameters, face IDs and scalar distances keep their CAD meaning. With
    diagnostics=True, transformed ``cad_*`` positions/normals are renamed
    ``reference_*`` so a BASE table never mislabels them as CAD coordinates.
    """
    matrix = np.asarray(T_target_from_source, dtype=float)
    assert_rigid(matrix)
    result = table.copy()
    positions = ['', 'mapped_', 'cad_', 'reference_', 'scan_', 'cad_closest_']
    vectors = ['normal_', 'cad_normal_', 'reference_normal_', 'scan_normal_',
               'tool_x_', 'tool_y_', 'tool_z_']
    for prefix, suffix, point in ([(p, '_mm', True) for p in positions]
                                  + [(v, '', False) for v in vectors]):
        columns = [f'{prefix}{axis}{suffix}' for axis in _AXES]
        if not all(column in result for column in columns):
            continue
        values = table[columns].to_numpy(float) @ matrix[:3, :3].T
        if point:
            values += matrix[:3, 3]
        result[columns] = values
    if len(table) and all(key in table for key in _QUATERNION):
        original = table[_QUATERNION].to_numpy(float)
        finite = np.isfinite(original).all(axis=1)
        values = np.full_like(original, np.nan)
        if finite.any():
            values[finite] = Rotation.from_matrix(
                matrix[:3, :3][None] @ Rotation.from_quat(original[finite]).as_matrix()).as_quat()
        result[_QUATERNION] = values
        # Quaternion sign is a representation choice, not a physical rotation.
        groups = result.groupby('segment_id', sort=False).groups.values() if 'segment_id' in result else [result.index]
        for ids in groups:
            ids = list(ids)
            q = result.loc[ids, _QUATERNION].to_numpy(float)
            if len(q) > 1:
                signs = np.r_[1., np.where(np.einsum('ij,ij->i',q[:-1],q[1:]) < 0.,-1.,1.)]
                q *= np.cumprod(signs)[:,None]
                result.loc[ids, _QUATERNION] = q
    if diagnostics:
        rename = {f'cad_{a}_mm': f'reference_{a}_mm' for a in _AXES}
        rename.update({f'cad_normal_{a}': f'reference_normal_{a}' for a in _AXES})
        rename.update({f'cad_closest_{a}_mm': f'reference_closest_{a}_mm' for a in _AXES})
        collision = [target for source, target in rename.items() if source in result and target in result]
        if collision:
            raise ValueError(f'Ambiguous reference diagnostic columns: {collision}')
        result = result.rename(columns=rename)
    result['frame'] = target_frame
    return result


def contact_poses_in_base(cad, T_BASE_to_CAD):
    """Return a compact BASE contact table and corresponding Nx4x4 matrices.

    Source rows must explicitly be CAD contact frames. This check prevents
    applying the registration inverse a second time to an existing BASE export.
    """
    if 'frame' not in cad or (len(cad) and not cad.frame.eq('CAD').all()):
        raise ValueError('Contact source frame must be CAD; refusing a second BASE conversion')
    matrix = cad_to_base_transform(T_BASE_to_CAD)
    base = transform_frame_table(cad, matrix)
    keys = ['path_id', 'sample_index', 'segment_id', 'face_id', 'region_id', 'u', 'v',
            'x_mm', 'y_mm', 'z_mm',
            *[f'tool_{a}_{c}' for a in _AXES for c in _AXES],
            *_QUATERNION, *[f'normal_{a}' for a in _AXES], 'normal_source',
            'position_reference', 'frame', 'execution_allowed']
    base['position_reference'] = 'surface_contact_point'
    base['execution_allowed'] = False
    base = base[[key for key in keys if key in base]].copy()
    transforms = np.tile(np.eye(4), (len(base), 1, 1))
    if len(base):
        transforms[:, :3, :3] = _axes(base)
        transforms[:, :3, 3] = base[['x_mm', 'y_mm', 'z_mm']].to_numpy(float)
    return base, transforms


def validate_base_contact_poses(cad, base, T_BASE_to_CAD, cfg=None):
    """Check serialized contact poses, round trip, normals and segment steps."""
    transform = np.asarray(T_BASE_to_CAD, float)
    assert_rigid(transform)
    if 'frame' not in cad or (len(cad) and not cad.frame.eq('CAD').all()):
        raise ValueError('Validation source frame must be CAD')
    if len(cad) != len(base):
        raise ValueError('CAD and BASE contact counts differ')
    if 'frame' not in base or (len(base) and not base.frame.eq(BASE_FRAME).all()):
        raise ValueError('BASE contact frame is not ROBOT_BASE')
    if not len(base):
        return {'passed': True, 'empty': True, 'accepted_pose_count': 0,
                'physical_accuracy_verified': False}
    ids = [key for key in ['path_id', 'sample_index', 'segment_id', 'face_id'] if key in cad]
    if not cad[ids].reset_index(drop=True).equals(base[ids].reset_index(drop=True)):
        raise ValueError('CAD and BASE contact identifiers differ')
    c_axes, b_axes = _axes(cad), _axes(base)
    p_c = cad[['x_mm', 'y_mm', 'z_mm']].to_numpy(float)
    p_b = base[['x_mm', 'y_mm', 'z_mm']].to_numpy(float)
    errors = {
        'base_to_cad_position_error_mm': float(np.max(np.abs(transform_points(p_b, transform)-p_c))),
        'base_to_cad_axis_error': float(np.max(np.abs(transform[:3, :3][None] @ b_axes-c_axes))),
        'base_rotation_orthogonality_error': float(np.max(np.abs(np.swapaxes(b_axes, 1, 2) @ b_axes-np.eye(3)))),
        'base_rotation_determinant_error': float(np.max(np.abs(np.linalg.det(b_axes)-1))),
        'base_quaternion_axis_error': float(np.max(np.abs(Rotation.from_quat(base[_QUATERNION]).as_matrix()-b_axes))),
    }
    normals = [f'normal_{a}' for a in _AXES]
    if all(key in base for key in normals):
        errors['base_axis_normal_error'] = float(np.max(np.abs(base[normals].to_numpy(float)+b_axes[:, :, 2])))
    step, angle = 0., 0.
    for _, group in base.groupby('segment_id', sort=False):
        if len(group) < 2:
            raise ValueError('A BASE contact segment has fewer than two points')
        p = group[['x_mm', 'y_mm', 'z_mm']].to_numpy(float)
        r = Rotation.from_quat(group[_QUATERNION])
        step = max(step, float(np.linalg.norm(np.diff(p, axis=0), axis=1).max()))
        angle = max(angle, float(np.degrees((r[:-1].inv()*r[1:]).magnitude()).max()))
    if not np.isfinite(list(errors.values())).all() or max(errors.values()) > 1e-8:
        raise ValueError(f'BASE contact validation failed: {errors}')
    if cfg and (step > cfg['max_point_jump_mm']+1e-8 or angle > cfg['max_pose_step_deg']+1e-8):
        raise ValueError(f'BASE contact continuity failed: {step} mm / {angle} degrees')
    if base.execution_allowed.any():
        raise ValueError('Unexpected execution authorization')
    return {'passed': True, 'empty': False, 'accepted_pose_count': len(base), **errors,
            'max_segment_step_mm': step, 'max_segment_orientation_step_deg': angle,
            'physical_accuracy_verified': False}


def write_registration_in_base(out, cfg, dest=None):
    """Publish CAD placement in BASE using an existing registration, no ICP.

    The scan's frame is a user declaration. It is not evidence of independent
    hand-eye/TCP calibration accuracy. Original scan XYZ and attributes stay put.
    """
    if cfg.get('scan_frame') != BASE_FRAME:
        raise ValueError("Declare scan_frame='ROBOT_BASE' before exporting base registration")
    if cfg.get('units') != 'mm':
        raise ValueError('Registration requires millimetres')
    out, dest = Path(out), Path(dest) if dest is not None else Path(out)
    matrix = np.load(out/'T_scan_to_CAD.npy')
    inverse = cad_to_base_transform(matrix)
    quality = json.loads((out/'registration_quality.json').read_text())
    if not np.allclose(matrix, quality['selected_candidate']['T_final'], atol=1e-9, rtol=0):
        raise ValueError('Registration matrix differs from registration metadata')
    hashes = {}
    for key, config_key, metadata_name in [('step', 'step_path', 'cad_metadata.json'),
                                            ('scan', 'scan_path', 'scan_metadata.json')]:
        hashes[key] = sha256(ROOT/cfg[config_key])
        if hashes[key] != json.loads((out/metadata_name).read_text())['sha256']:
            raise ValueError(f'{key} changed since registration')
    hashes['registration'] = sha256(out/'T_scan_to_CAD.npy')
    source_mesh = out/'cad_mesh_clean.ply'
    hashes['cad_mesh'] = sha256(source_mesh)
    dest.mkdir(parents=True, exist_ok=True)
    for name, value in [('T_BASE_to_CAD', matrix), ('T_CAD_to_BASE', inverse)]:
        np.save(dest/f'{name}.npy', value)
        np.savetxt(dest/f'{name}.txt', value, fmt='%.15g')
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(source_mesh))
    if not len(mesh.vertices):
        raise ValueError('Registration CAD mesh has no vertices')
    ids = np.unique(np.linspace(0, len(mesh.vertices)-1, min(4096, len(mesh.vertices))).astype(int))
    source = np.asarray(mesh.vertices)[ids].copy()
    mesh.transform(inverse)
    if not o3d.io.write_triangle_mesh(str(dest/'CAD_in_BASE.ply'), mesh, write_ascii=False):
        raise RuntimeError('Could not write CAD_in_BASE.ply')
    saved, _ = read_ply(dest/'CAD_in_BASE.ply')
    actual = np.column_stack([saved[a][ids] for a in _AXES])
    expected = transform_points(source, inverse)
    serialization_error = float(np.max(np.abs(actual-expected)))
    roundtrip = float(np.max(np.abs(transform_points(actual, matrix)-source)))
    identity_error = float(np.max(np.abs(matrix @ inverse-np.eye(4))))
    if max(serialization_error, roundtrip, identity_error) > 1e-8:
        raise ValueError('Serialized CAD BASE registration failed round-trip validation')
    result = {
        'primary_frame': BASE_FRAME, 'scan_frame': BASE_FRAME, 'cad_native_frame': 'CAD',
        'scan_frame_declaration': 'user_confirmed_original_scan_is_robot_base',
        'units': 'mm', 'T_BASE_to_CAD': matrix.tolist(), 'T_CAD_to_BASE': inverse.tolist(),
        'matrix_convention': 'p_target = R_target_from_source @ p_source + t_target_from_source',
        'scan_xyz_transformed': False, 'original_scan_preserved': True,
        'registration_recomputed': False, 'cad_model_file': 'CAD_in_BASE.ply',
        'cad_mesh_triangle_ids': 'indices of cad_mesh_clean.ply; not STEP face IDs',
        'input_sha256': hashes, 'validation': {'passed': True, 'checked_mesh_vertices': len(ids),
            'cad_base_serialization_error_mm': serialization_error,
            'cad_base_roundtrip_error_mm': roundtrip, 'transform_inverse_error': identity_error},
        'independent_base_calibration_accuracy_verified': False, 'execution_allowed': False,
    }
    dump_json(dest/'registration_BASE.json', result)
    return result
