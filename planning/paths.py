"""M8: candidate patches -> clipped raster -> OpenCAMLib spherical cutter locations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import points as shapely_points, buffer, union_all
from shapely.geometry import Point, LineString
from .geometry import TriangleFile, ocl_frame, tool_frame, quaternion_xyzw, ball_drop, aligned_raster_lines, split_valid
from .candidates import classify
from .validation import source_manifest
from pipeline_common import dump_json, sha256


def validate_config(cfg):
    positive = ['registration_target_mm', 'candidate_threshold_mm', 'local_radius_mm', 'cluster_eps_mm',
                'support_radius_mm', 'contact_support_tolerance_mm', 'tool_ball_diameter_mm',
                'tool_length_mm', 'raster_spacing_mm', 'path_sampling_mm', 'minimum_stroke_length_mm',
                'max_contact_jump_mm', 'clearance_mm', 'geometry_tolerance_mm']
    for key in positive:
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if cfg['tool_length_mm'] <= cfg['tool_ball_diameter_mm']:
        raise ValueError('Tool length must exceed ball diameter')
    if cfg['machining_authorized']:
        raise ValueError('This pipeline only exports non-executable M8 previews')
    if cfg['raster_spacing_mm'] > cfg['tool_ball_diameter_mm']:
        raise ValueError('Preview raster spacing exceeds nominal tool diameter')
    # Ball diameter is a geometry limit, not a calibrated abrasive contact width.


def project_region(region, mesh, aabb, frame, zmin, cfg):
    surface_points = region[['cad_closest_x_mm', 'cad_closest_y_mm', 'cad_closest_z_mm']].to_numpy()@frame
    polygon = union_all(buffer(shapely_points(surface_points[:, :2]), cfg['support_radius_mm'], quad_segs=4))
    strokes_xy, raster_angle = aligned_raster_lines(polygon, surface_points[:, :2], cfg['raster_spacing_mm'], cfg['path_sampling_mm'])
    if not strokes_xy:
        return [], {'reason': 'no_raster_intersection', 'input_points': len(region)}
    r = cfg['tool_ball_diameter_mm']/2
    margin = r+cfg['geometry_tolerance_mm']+.001
    xmin, ymin, xmax, ymax = polygon.bounds
    triangles = mesh.crop(aabb, (xmin-margin, ymin-margin, xmax+margin, ymax+margin), frame)
    xy = np.concatenate(strokes_xy)
    tip, contact, touched = ball_drop(triangles, xy, 2*r, cfg['tool_length_mm'], zmin-10-r)
    center = tip + [0, 0, r]
    radial = center-contact
    radial_length = np.linalg.norm(radial, axis=1)
    normal = radial/np.maximum(radial_length[:, None], 1e-12)
    support_tree = cKDTree(surface_points)
    support_distance, support_index = support_tree.query(contact, workers=2)
    contact_in_patch = np.array([polygon.covers(Point(float(p[0]), float(p[1]))) for p in contact])
    original_normals = region[['normal_x', 'normal_y', 'normal_z']].to_numpy()@frame
    normal_agreement = np.einsum('ij,ij->i', normal, original_normals[support_index])
    valid = (touched & (np.abs(radial_length-r) <= cfg['geometry_tolerance_mm']) &
             (normal[:, 2] >= cfg['minimum_normal_alignment']) & contact_in_patch &
             (support_distance <= cfg['contact_support_tolerance_mm']) &
             (normal_agreement >= cfg['normal_neighbor_cosine']))
    # Check actual sphere clearance at all samples AND midpoints of proposed contact edges.
    # Scene is built for one cropped patch only, keeping memory bounded.
    import open3d as o3d
    local_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(triangles.reshape(-1, 3).astype(np.float32)),
        o3d.core.Tensor(np.arange(len(triangles)*3).reshape(-1, 3).astype(np.uint32)))
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(local_mesh)
    clearance = scene.compute_distance(o3d.core.Tensor(center.astype(np.float32)), nthreads=2).numpy()-r
    valid &= clearance >= -cfg['geometry_tolerance_mm']
    result, offset = [], 0
    accepted_sample_count = 0
    min_midpoint_clearance = np.inf
    min_clearance = np.inf
    for line in strokes_xy:
        count = len(line)
        c = center[offset:offset+count]
        v = valid[offset:offset+count].copy()
        # A rejected midpoint forces a new stroke, never an untested direct connection.
        if count > 1:
            midpoint = (c[1:]+c[:-1])/2
            mc = scene.compute_distance(o3d.core.Tensor(midpoint.astype(np.float32)), nthreads=2).numpy()-r
            v[1:] &= mc >= -cfg['geometry_tolerance_mm']
            cc = contact[offset:offset+count]
            # Tool-center clipping alone does not guarantee the shifted contact curve
            # stays in supported material. Check each contact edge as well.
            supported_edges = np.array([polygon.buffer(1e-7).covers(LineString([a[:2], b[:2]]))
                                        for a, b in zip(cc[:-1], cc[1:])])
            midpoint_support = support_tree.query((cc[1:]+cc[:-1])/2, workers=2)[0]
            v[1:] &= supported_edges & (midpoint_support <= cfg['contact_support_tolerance_mm'])
        for ids in split_valid(c, v, cfg['max_contact_jump_mm']):
            if np.linalg.norm(np.diff(c[ids], axis=0), axis=1).sum() < cfg['minimum_stroke_length_mm']:
                continue
            g = offset+ids
            result.append({'center': center[g]@frame.T, 'contact': contact[g]@frame.T,
                           'normal': normal[g]@frame.T,
                           'analysis_rows': region.analysis_row.to_numpy()[support_index[g]],
                           'support_distance': support_distance[g], 'clearance': clearance[g]})
            accepted_sample_count += len(ids)
            min_clearance = min(min_clearance, float(clearance[g].min()))
            min_midpoint_clearance = min(min_midpoint_clearance, float(mc[ids[:-1]].min()))
        offset += count
    return result, {'input_points': len(region), 'cad_triangles_in_crop': len(triangles),
                    'projected_samples': len(xy), 'valid_samples_before_min_length': int(valid.sum()),
                    'accepted_waypoints': accepted_sample_count, 'strokes': len(result),
                    'raster_angle_in_tool_plane_deg': raster_angle,
                    'min_sample_sphere_clearance_mm': min_clearance if np.isfinite(min_clearance) else None,
                    'min_midpoint_sphere_clearance_mm': min_midpoint_clearance if np.isfinite(min_midpoint_clearance) else None,
                    'accepted_penetration_tolerance_mm': cfg['geometry_tolerance_mm'],
                    'clearance_note': 'Negative midpoint values include chord approximation, not only float roundoff'}


def order_strokes(strokes):
    """Deterministic nearest endpoint, allowing reversal; not globally optimal."""
    remaining = list(strokes)
    ordered = []
    previous = None
    while remaining:
        if previous is None:
            index, reverse = 0, False
        else:
            choices = [(float(np.linalg.norm(s['center'][end]-previous)), i, end == -1)
                       for i, s in enumerate(remaining) for end in (0, -1)]
            _, index, reverse = min(choices)
        stroke = remaining.pop(index)
        if reverse:
            stroke = {k: (v[::-1].copy() if isinstance(v, np.ndarray) else v) for k, v in stroke.items()}
        ordered.append(stroke)
        previous = stroke['center'][-1]
    return ordered


def export_poses(strokes, frame, clearance_height, cfg):
    rows, arc = [], 0.
    outward = frame[:, 2]
    # Constant tool orientation avoids unvalidated roll changes at patch boundaries.
    R = tool_frame(outward, frame[:, 0])
    quat = quaternion_xyzw(R)
    previous = None
    def add(center, phase, stroke_id, region_id, contact=None, normal=None, analysis_row=None, clearance=None):
        nonlocal previous, arc
        center = np.asarray(center)
        if previous is not None:
            arc += float(np.linalg.norm(center-previous))
        previous = center
        row = {'sequence': len(rows), 'stroke_id': stroke_id, 'region_id': region_id, 'phase': phase,
               'frame': 'CAD', 'tcp_definition': 'sphere_center', 'path_distance_mm': arc,
               'x_mm': center[0], 'y_mm': center[1], 'z_mm': center[2],
               **dict(zip(['qx', 'qy', 'qz', 'qw'], quat)),
               'analysis_row': analysis_row, 'sphere_clearance_mm': clearance,
               'removal_depth_mm': None, 'feed_mm_s': None, 'force_N': None,
               'material_side_confirmed': cfg['material_side_confirmed'],
               'tool_confirmed': cfg['tool_confirmed'], 'execution_allowed': False}
        for prefix, vector in [('contact', contact), ('normal', normal)]:
            for axis, val in zip('xyz', vector if vector is not None else [None]*3):
                row[f'{prefix}_{axis}'+('_mm' if prefix == 'contact' else '')] = val
        rows.append(row)
    def move_to(destination, phase, sid, rid):
        if previous is None:
            add(destination, phase, sid, rid)
            return
        start = previous.copy()
        count = max(1, int(np.ceil(np.linalg.norm(destination-start)/cfg['max_contact_jump_mm'])))
        for fraction in np.linspace(0, 1, count+1)[1:]:
            add(start+(destination-start)*fraction, phase, sid, rid)
    for sid, stroke in enumerate(strokes):
        rid = stroke['region_id']
        first, last = stroke['center'][0], stroke['center'][-1]
        entry = first+outward*(clearance_height-np.dot(first, outward))
        exit_point = last+outward*(clearance_height-np.dot(last, outward))
        move_to(entry, 'clearance_transfer', sid, rid)
        # Vertical approach samples stop before the first nominal-contact sample.
        distance = np.linalg.norm(first-entry)
        count = max(1, int(np.ceil(distance/cfg['max_contact_jump_mm'])))
        for fraction in np.linspace(0, 1, count+1)[1:-1]:
            add(entry+(first-entry)*fraction, 'approach', sid, rid)
        for i, point in enumerate(stroke['center']):
            add(point, 'nominal_contact', sid, rid, stroke['contact'][i], stroke['normal'][i],
                int(stroke['analysis_rows'][i]), float(stroke['clearance'][i]))
        move_to(exit_point, 'retract', sid, rid)
    return pd.DataFrame(rows)


def run_m8(root, cfg, pcfg, run_dir):
    validate_config(pcfg)
    m7 = json.loads((run_dir/'m7/validation.json').read_text())
    recorded = json.loads((run_dir/'planning_config.json').read_text())
    if recorded != pcfg:
        raise ValueError('M7 and M8 must use the same recorded planning configuration')
    if not m7['numerical_passed']:
        raise ValueError('M7 numerical checks failed')
    out = root/cfg['output_dir']
    if sha256(root/cfg['step_path']) != m7['step_sha256']:
        raise ValueError('STEP input changed after M7')
    for name, expected in m7['input_sha256'].items():
        if sha256(out/name) != expected:
            raise ValueError(f'Input changed after M7: {name}')
    dest = run_dir/'m8'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'path_manifest.json').exists():
        raise FileExistsError('Use a new run directory; existing completed M8 is immutable')
    classified = classify(pd.read_csv(out/'deviation_points.csv'), pcfg)
    classified.to_csv(dest/'candidate_regions.csv', index=False)
    counts = classified.classification.value_counts().to_dict()
    print('Candidate classification', counts, flush=True)
    mesh = TriangleFile(out/'cad_mesh_clean.ply')
    frame = ocl_frame(pcfg['approach_outward_C'])
    aabb, zmin, zmax = mesh.bounds_in_frame(frame, dest/'triangle_projected_bounds.npy')
    print('Mesh projected bounds ready', len(mesh.faces), flush=True)
    all_strokes, regions = [], []
    for rid in sorted(classified.loc[classified.region_id >= 0, 'region_id'].unique()):
        region = classified[classified.region_id == rid]
        strokes, stats = project_region(region, mesh, aabb, frame, zmin, pcfg)
        for stroke in strokes:
            stroke['region_id'] = int(rid)
        all_strokes.extend(strokes)
        regions.append({'region_id': int(rid), **stats})
        print('Region', int(rid), stats, flush=True)
    dump_json(dest/'region_planning.json', regions)
    ordered = order_strokes(all_strokes)
    if not ordered:
        raise RuntimeError('No supported, reachable-from-assumed-side contact strokes survived; do not fabricate a path')
    clearance_height = zmax + pcfg['tool_ball_diameter_mm']/2 + pcfg['clearance_mm']
    poses = export_poses(ordered, frame, clearance_height, pcfg)
    checks = validate_exported_poses(poses, pcfg)
    dump_json(dest/'artifact_checks.json', checks)
    poses.to_csv(dest/'target_pose.csv', index=False)
    contacts = poses[poses.phase == 'nominal_contact']
    contact_length = sum(float(np.linalg.norm(np.diff(s['center'], axis=0), axis=1).sum()) for s in ordered)
    candidate_points = classified.loc[classified.region_id >= 0, ['cad_closest_x_mm', 'cad_closest_y_mm', 'cad_closest_z_mm']].to_numpy()
    distance = cKDTree(contacts[['contact_x_mm', 'contact_y_mm', 'contact_z_mm']].to_numpy()).query(candidate_points, workers=2)[0]
    coverage = classified.loc[classified.region_id >= 0, ['analysis_row', 'region_id']].copy()
    coverage['nearest_contact_sample_mm'] = distance
    coverage['within_1mm_of_contact_sample'] = distance <= 1
    coverage.to_csv(dest/'candidate_path_coverage.csv', index=False)
    manifest = {'status': 'offline_geometry_preview', 'coordinate_frame': 'CAD', 'units': 'mm',
                'tool_model': 'provisional fixed-axis spherical abrasive head; OpenCAMLib BallCutter',
                'tcp_definition': 'sphere_center', 'tool_z_direction': 'toward surface, opposite approach_outward_C',
                'contact_target': 'nominal CAD envelope; removal depth intentionally unspecified',
                'execution_allowed': False, 'physical_validation_passed': m7['physical_gate']['passed'],
                'robot_ik_checked': False, 'holder_fixture_collision_checked': False,
                'continuous_collision_certified': False,
                'classification_counts': counts, 'candidate_regions': len(regions),
                'regions_with_paths': sum(r.get('strokes', 0)>0 for r in regions),
                'strokes': len(ordered), 'total_waypoints': len(poses), 'contact_waypoints': len(contacts),
                'contact_path_length_mm': contact_length, 'total_path_length_mm': float(poses.path_distance_mm.iloc[-1]),
                'geometric_clearance_plane_height_along_outward_mm': clearance_height,
                'candidate_fraction_near_contact_path_1mm': float(np.mean(distance <= 1)),
                'coverage_meaning': 'Point proximity only; not calibrated abrasive footprint or area coverage',
                'source': source_manifest(root), 'input_sha256': m7['input_sha256'],
                'config': pcfg, 'regions': regions}
    dump_json(dest/'path_manifest.json', manifest)
    return classified, poses, ordered, manifest


def validate_exported_poses(poses, cfg):
    xyz = poses[['x_mm', 'y_mm', 'z_mm']].to_numpy()
    quat = poses[['qx', 'qy', 'qz', 'qw']].to_numpy()
    if not len(poses) or not np.isfinite(xyz).all() or not np.isfinite(quat).all():
        raise ValueError('Non-finite or empty pose sequence')
    if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-9, rtol=0):
        raise ValueError('Invalid unit quaternion')
    max_step = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).max())
    if max_step > cfg['max_contact_jump_mm']+1e-7:
        raise ValueError('Exported waypoint jump exceeds configured limit')
    contact = poses[poses.phase == 'nominal_contact']
    r = cfg['tool_ball_diameter_mm']/2
    offset = contact[['x_mm','y_mm','z_mm']].to_numpy()-contact[['contact_x_mm','contact_y_mm','contact_z_mm']].to_numpy()
    max_radius_error = float(np.max(np.abs(np.linalg.norm(offset, axis=1)-r)))
    if max_radius_error > cfg['geometry_tolerance_mm']:
        raise ValueError('Tool center is inconsistent with the spherical contact')
    if poses.execution_allowed.any() or not poses.removal_depth_mm.isna().all():
        raise ValueError('Preview must not contain execution authorization or a removal-depth command')
    return {'poses_finite': True, 'unit_quaternion': True, 'execution_disabled': True,
            'max_step_mm': max_step, 'max_ball_radius_error_mm': max_radius_error}
