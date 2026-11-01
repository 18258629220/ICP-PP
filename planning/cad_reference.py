"""CAD-native paths on exact, trimmed STEP faces (all lengths in mm).

The optional selections contain zero-based STEP face IDs and either UV or XYZ
control points. Controls are joined in that face's UV domain; they are not a
robot program. Auto selection provides representative supported face curves,
not an assertion that those are the correct production deburring locations.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def load_faces(step_path):
    import cadquery as cq
    roots = cq.importers.importStep(str(step_path)).vals()
    return [face for root in roots for face in root.Faces()]


def _inside(face, uv):
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.gp import gp_Pnt2d
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    return BRepClass_FaceClassifier(face.wrapped, gp_Pnt2d(*map(float, uv)), 1e-8).State() in (TopAbs_IN, TopAbs_ON)


def _clip_uv_segment(face, a, b):
    """Split at exact p-curve/trim-boundary intersections, including tiny holes."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    from OCP.Geom2d import Geom2d_Line, Geom2d_TrimmedCurve
    from OCP.Geom2dAPI import Geom2dAPI_InterCurveCurve
    from OCP.gp import gp_Pnt2d, gp_Dir2d
    a, b = np.asarray(a, float), np.asarray(b, float)
    distance = np.linalg.norm(b-a)
    if distance < 1e-12:
        return []
    direction = (b-a)/distance
    line = Geom2d_TrimmedCurve(Geom2d_Line(gp_Pnt2d(*a), gp_Dir2d(*direction)), 0., float(distance))
    cuts = [0., float(distance)]
    for edge in face.Edges():
        adaptor = BRepAdaptor_Curve2d(edge.wrapped, face.wrapped)
        curve = Geom2d_TrimmedCurve(adaptor.Curve(), adaptor.FirstParameter(), adaptor.LastParameter())
        intersection = Geom2dAPI_InterCurveCurve(line, curve, 1e-9)
        for j in range(1, intersection.NbPoints()+1):
            point = intersection.Point(j)
            t = np.dot(np.array([point.X(), point.Y()])-a, direction)
            if -1e-8 <= t <= distance+1e-8:
                cuts.append(float(np.clip(t, 0., distance)))
    cuts = sorted(set(round(x, 11) for x in cuts))
    return [(a+lo*direction, a+hi*direction) for lo, hi in zip(cuts[:-1], cuts[1:])
            if hi-lo > 1e-9 and _inside(face, a+(lo+hi)*.5*direction)]


def _sample_segment(face, a, b, spacing):
    probe_uv = np.linspace(a, b, 17)
    probe = np.array([face.positionAt(*map(float, uv)).toTuple() for uv in probe_uv])
    length = np.linalg.norm(np.diff(probe, axis=0), axis=1).sum()
    count = max(2, int(np.ceil(length/min(spacing/3., .2)))+1)
    if count > 100000:
        raise ValueError('Reference segment is too long; verify CAD units and controls')
    uv = np.linspace(a, b, count)
    points = np.array([face.positionAt(*map(float, x)).toTuple() for x in uv])
    if not np.all(np.isfinite(points)):
        raise ValueError('Non-finite CAD surface position')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if arc[-1] < 1e-8:
        return None
    sample = np.linspace(0., arc[-1], max(2, int(np.ceil(arc[-1]/spacing))+1))
    uv = np.column_stack([np.interp(sample, arc, uv[:, j]) for j in range(2)])
    return uv


def paths_on_face(face, uv_controls, spacing=.5, outward=(0., 0., -1.)):
    """Return separated exact-face paths. A trim hole always creates a split."""
    controls = np.asarray(uv_controls, float)
    outward = np.asarray(outward, float)
    if controls.ndim != 2 or controls.shape[1] != 2 or len(controls) < 2 or not np.isfinite(controls).all():
        raise ValueError('uv_controls must contain at least two finite UV pairs')
    if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(outward).all() or np.linalg.norm(outward) < 1e-9:
        raise ValueError('Invalid spacing or approach direction')
    outward = outward/np.linalg.norm(outward)
    spans, current = [], None
    for a, b in zip(controls[:-1], controls[1:]):
        segments = _clip_uv_segment(face, a, b)
        if not segments and current is not None:
            spans.append(current); current = None
        for lo, hi in segments:
            uv = _sample_segment(face, lo, hi, spacing)
            if uv is None:
                continue
            if current is not None and np.linalg.norm(current[-1]-uv[0]) < 1e-8:
                current = np.concatenate([current, uv[1:]])
            else:
                if current is not None:
                    spans.append(current)
                current = uv
        # The next control segment must not reconnect across a trimmed endpoint.
        if current is not None and np.linalg.norm(current[-1]-b) > 1e-8:
            spans.append(current); current = None
    if current is not None:
        spans.append(current)
    result = []
    for uv in spans:
        pairs = [face.normalAt(*map(float, x)) for x in uv]
        normals = np.array([x[0].toTuple() for x in pairs])
        points = np.array([x[1].toTuple() for x in pairs])
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
        uv, points, normals = uv[keep], points[keep], normals[keep]
        if len(points) < 2:
            continue
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        # One sign per continuous curve: never flip individual samples on a bend.
        dots = normals @ outward
        sign = np.sign(np.mean(dots)) or 1.
        normals *= sign
        if not np.isfinite(normals).all():
            raise ValueError('Undefined CAD normal at selected path')
        result.append({'points_C': points, 'normals_C': normals, 'uv': uv,
                       'normal_sign_assumed': True,
                       'normal_orientation': 'single curve sign aligned to configured approach_outward_C'})
    return result


def _face_record(face, index):
    return {'face_id': index, 'geometry_type': face.geomType(), 'area_mm2': float(face.Area()),
            'uv_bounds': list(map(float, face.uvBounds()))}


def _path_length(path):
    return float(np.linalg.norm(np.diff(path['points_C'], axis=0), axis=1).sum())


def build_reference_paths(step_path, support_points_C, cfg):
    """Return (paths, catalog) with stable zero-based STEP face IDs.

    Auto mode samples a bounded pool of large faces and prioritizes continuous
    scan-supported curves. Manual mode does not silently replace a selection.
    """
    faces = load_faces(Path(step_path))
    print(f'CAD loaded: {len(faces)} exact faces', flush=True)
    spacing = float(cfg.get('path_sampling_mm', .5))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('path_sampling_mm must be positive and finite')
    outward = cfg.get('approach_outward_C', [0., 0., -1.])
    support = np.asarray(support_points_C, float)
    if support.ndim != 2 or support.shape[1] != 3 or not len(support) or not np.isfinite(support).all():
        raise ValueError('support_points_C must be finite, nonempty Nx3')
    tree = cKDTree(support)
    support_distance = float(cfg.get('reference_support_distance_mm', 2.5))
    min_length = float(cfg.get('minimum_reference_length_mm', 5.))
    selections = cfg.get('selections', [])
    records = [_face_record(face, i) for i, face in enumerate(faces)]
    print('CAD face catalog ready', flush=True)
    catalog = {'face_index_convention': 'zero-based STEP imported root/face traversal; NOT mesh triangle IDs',
               'face_count': len(faces), 'faces': records,
               'selection_mode': 'explicit' if selections else 'automatic_representative',
               'material_side_confirmed': False}
    paths = []
    if selections:
        for sid, selection in enumerate(selections):
            index = int(selection['face_id'])
            if index < 0 or index >= len(faces):
                raise ValueError(f'Unknown STEP face_id {index}')
            if ('uv_controls' in selection) == ('xyz_controls' in selection):
                raise ValueError('Provide exactly one of uv_controls or xyz_controls')
            controls = selection.get('uv_controls')
            if controls is None:
                xyz = np.asarray(selection['xyz_controls'], float)
                if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
                    raise ValueError('xyz_controls must contain at least two finite XYZ points')
                controls = [faces[index].paramAt(tuple(p)) for p in xyz]
            found = paths_on_face(faces[index], controls, spacing, outward)
            if not found:
                raise ValueError(f'Selection {sid} has no usable interval on the trimmed face')
            for part, path in enumerate(found):
                path.update(path_id=f'selection_{sid:02d}_part_{part:02d}', face_ids=np.full(len(path['points_C']), index, dtype=int),
                            selection={**selection, 'mode': 'explicit', 'curve_definition': 'piecewise linear UV controls on exact face'})
                paths.append(path)
        return paths, catalog

    # Avoid exhaustive dense sampling of thousands of tiny CAD feature faces.
    areas = np.array([r['area_mm2'] for r in records])
    pool_limit = int(cfg.get('reference_face_candidate_limit', 120))
    ids = np.argsort(-areas, kind='stable')[:pool_limit]
    ranked = []
    for index in ids:
        face, record = faces[index], records[index]
        u0, u1, v0, v1 = record['uv_bounds']
        if not np.isfinite([u0, u1, v0, v1]).all():
            continue
        uv = [(u, v) for u in np.linspace(u0, u1, 9)[1:-1]
              for v in np.linspace(v0, v1, 9)[1:-1] if _inside(face, (u, v))]
        if not uv:
            continue
        points = np.array([face.positionAt(*map(float, x)).toTuple() for x in uv])
        distances = tree.query(points)[0]
        mask = distances <= support_distance
        fraction = float(np.mean(mask))
        record.update(probe_count=len(uv), probe_supported_count=int(mask.sum()), probe_support_fraction=fraction)
        if mask.sum() < 3:
            continue
        ranked.append((fraction*np.sqrt(areas[index]), int(index), np.asarray(uv)[mask]))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    print(f'CAD scan-supported face candidates: {len(ranked)}', flush=True)
    candidates = []
    for _, index, supported_uv in ranked[:int(cfg.get('reference_path_face_limit', 18))]:
        print(f'Generating exact-face curves: face {index}', flush=True)
        face, record = faces[index], records[index]
        u0, u1, v0, v1 = record['uv_bounds']
        for axis in [0, 1]:
            for fraction in [.25, .5, .75]:
                fixed = float(np.quantile(supported_uv[:, 1-axis], fraction))
                controls = [[u0, fixed], [u1, fixed]] if axis == 0 else [[fixed, v0], [fixed, v1]]
                for part in paths_on_face(face, controls, spacing, outward):
                    distance = tree.query(part['points_C'])[0]
                    good = distance <= support_distance
                    # Continuous scan support is required; never bridge its missing spans.
                    starts = np.flatnonzero(good & ~np.r_[False, good[:-1]])
                    ends = np.flatnonzero(good & ~np.r_[good[1:], False])+1
                    for start, end in zip(starts, ends):
                        if end-start < 3:
                            continue
                        path = {k: (v[start:end] if isinstance(v, np.ndarray) else v) for k, v in part.items()}
                        length = _path_length(path)
                        if length < min_length:
                            continue
                        normals = path['normals_C']
                        normal_change = float(np.degrees(np.arccos(np.clip(normals @ normals[len(normals)//2], -1., 1.))).max())
                        path.update(face_ids=np.full(end-start, index, dtype=int),
                                    selection={'mode': 'automatic_representative', 'face_id': index,
                                               'geometry_type': record['geometry_type'], 'uv_controls': controls,
                                               'length_mm': length, 'normal_variation_deg': normal_change,
                                               'support_distance_limit_mm': support_distance,
                                               'production_region_selected_by_user': False})
                        candidates.append(path)
    maximum = int(cfg.get('max_reference_paths', 3))
    if maximum < 1:
        raise ValueError('max_reference_paths must be positive')
    candidates.sort(key=lambda p: (-_path_length(p), p['selection']['face_id']))
    # Reserve one supported genuinely curved path if present, then prefer new faces.
    curved = [p for p in candidates if p['selection']['normal_variation_deg'] >= 3.]
    chosen = ([max(curved, key=_path_length)] if curved and maximum >= 2 else [])
    used = {p['selection']['face_id'] for p in chosen}
    for path in candidates:
        if len(chosen) >= maximum:
            break
        if path['selection']['face_id'] not in used:
            chosen.append(path); used.add(path['selection']['face_id'])
    for path in candidates:
        if len(chosen) >= maximum:
            break
        if not any(path is existing for existing in chosen):
            chosen.append(path)
    for i, path in enumerate(chosen):
        path['path_id'] = f'cad_path_{i:02d}'
    catalog['auto_candidate_curve_count'] = len(candidates)
    catalog['selected_path_count'] = len(chosen)
    if not chosen:
        raise ValueError('No supported trimmed-face reference curve; provide explicit face/controls or inspect registration')
    return chosen, catalog
