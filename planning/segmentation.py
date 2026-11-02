"""Conservative CAD-guided assignment to exact trimmed faces.

A face-AABB hierarchy rejects distant faces before exact surface projection.
Competing faces, trim holes, and physical boundaries remain explicitly unknown.
Confidence is a diagnostic score, not calibrated probability or metrology proof.
"""
import numpy as np
import pandas as pd


DEFAULT_SEGMENTATION_CONFIG = {
    'max_distance_mm': 1., 'ambiguity_margin_mm': .05,
    'boundary_margin_mm': .1, 'max_normal_angle_deg': 35.,
    'normal_orientation_known': False, 'max_candidates': 64,
}


def segmentation_config(cfg=None):
    result = {**DEFAULT_SEGMENTATION_CONFIG, **(cfg or {})}
    for key in ('max_distance_mm', 'ambiguity_margin_mm', 'boundary_margin_mm'):
        if not np.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if result['max_distance_mm'] <= 0:
        raise ValueError('max_distance_mm must be positive')
    if not np.isfinite(result['max_normal_angle_deg']) or not 0 < result['max_normal_angle_deg'] <= 90:
        raise ValueError('max_normal_angle_deg must be in (0,90]')
    if not isinstance(result['max_candidates'], int) or result['max_candidates'] < 1:
        raise ValueError('max_candidates must be a positive integer')
    return result


class FaceBoundsIndex:
    """Conservative AABB BVH; does not rely on triangle mesh approximation."""
    def __init__(self, bounds):
        self.bounds = np.asarray(bounds, float)
        def build(ids):
            lo, hi = self.bounds[ids, :3].min(axis=0), self.bounds[ids, 3:].max(axis=0)
            if len(ids) <= 8:
                return lo, hi, ids, None
            centers = (self.bounds[ids, :3]+self.bounds[ids, 3:])*.5
            axis = np.argmax(np.ptp(centers, axis=0))
            ids = ids[np.argsort(centers[:, axis], kind='stable')]
            middle = len(ids)//2
            return lo, hi, None, (build(ids[:middle]), build(ids[middle:]))
        self.root = build(np.arange(len(self.bounds)))

    def query(self, point, distance):
        result, stack = [], [self.root]
        while stack:
            lo, hi, ids, children = stack.pop()
            if np.linalg.norm(np.maximum(np.maximum(lo-point, point-hi), 0)) > distance:
                continue
            if ids is None:
                stack.extend(children)
            else:
                delta = np.maximum(np.maximum(self.bounds[ids, :3]-point, point-self.bounds[ids, 3:]), 0)
                result.extend(ids[np.linalg.norm(delta, axis=1) <= distance].tolist())
        return sorted(result)


def project_to_face(point, topology, face_id):
    """Nearest orthogonal support-surface projection inside exact face trims.

    Boundary points are retained for diagnostics, never projected across a hole
    to its closest rim. All extrema are checked so periodic charts can be used.
    """
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.gp import gp_Pnt, gp_Pnt2d
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    face = topology.faces[face_id]
    projector = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, point)), topology.surfaces[face_id])
    if not projector.IsDone():
        return None
    best = None
    for i in range(1, projector.NbPoints()+1):
        u, v = projector.Parameters(i)
        state = BRepClass_FaceClassifier(face.wrapped, gp_Pnt2d(u, v), 1e-8).State()
        if state not in (TopAbs_IN, TopAbs_ON):
            continue
        normal, projected = face.normalAt(float(u), float(v))
        q, n = np.asarray(projected.toTuple()), np.asarray(normal.toTuple())
        if not np.isfinite(np.r_[q, n, u, v]).all() or np.linalg.norm(n) < 1e-12:
            continue
        distance = float(np.linalg.norm(point-q))
        if best is None or distance < best['distance_mm']:
            best = {'u': float(u), 'v': float(v), 'projected': q,
                    'normal': n/np.linalg.norm(n), 'distance_mm': distance,
                    'on_trim_boundary': state == TopAbs_ON}
    return best


def boundary_distance(point_on_face, topology, face_id):
    """3D distance to non-seam trim edges, including inner hole wires."""
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    import cadquery as cq
    vertex = cq.Vertex.makeVertex(*map(float, point_on_face))
    best = np.inf
    for edge in topology.boundary_edges[face_id]:
        b = edge.BoundingBox()
        lower = np.linalg.norm(np.maximum(np.maximum(np.array([b.xmin,b.ymin,b.zmin])-point_on_face,
                                                      point_on_face-np.array([b.xmax,b.ymax,b.zmax])), 0))
        if lower > best:
            continue
        calc = BRepExtrema_DistShapeShape(vertex.wrapped, edge.wrapped)
        if not calc.IsDone():
            raise RuntimeError('OCCT boundary distance failed')
        best = min(best, calc.Value())
    return float(best)


def segment_points(points_C, topology, cfg=None, normals_C=None, source_indices=None):
    """Return one diagnostic row per input point; reject assignments use face_id=-1.

    Optional oriented scan normals can distinguish two thin-wall material sides.
    Without known orientation only the acute normal angle is used. No labels
    propagate across topology, holes, or rejected points. ``candidate_face_id``
    preserves the best hypothesis even when a point is not safe to assign.
    """
    cfg = segmentation_config(cfg)
    points = np.asarray(points_C, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_C must be Nx3')
    normals = None if normals_C is None else np.asarray(normals_C, float)
    if normals is not None and normals.shape != points.shape:
        raise ValueError('normals_C must have the same Nx3 shape as points_C')
    indices = np.arange(len(points), dtype=np.int64) if source_indices is None else np.asarray(source_indices)
    if indices.shape != (len(points),) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('source_indices must contain one integer per input point')
    index = FaceBoundsIndex(topology.face_bounds)
    rows = []
    for row_id, point in enumerate(points):
        row = {'source_index': int(indices[row_id]), 'face_id': -1, 'candidate_face_id': -1,
               'valid': False, 'status': 'no_candidate', 'confidence': 0.,
               'u': np.nan, 'v': np.nan, 'distance_mm': np.nan, 'signed_deviation_mm': np.nan,
               'second_distance_mm': np.nan, 'competition_margin_mm': np.nan,
               'boundary_distance_mm': np.nan, 'scan_normal_angle_deg': np.nan,
               'candidate_count': 0, 'exact_projection_count': 0,
               'normal_side_confirmed': False,
               **{f'projected_{a}_mm': np.nan for a in 'xyz'},
               **{f'normal_{a}': np.nan for a in 'xyz'}}
        if not np.isfinite(point).all():
            row['status'] = 'nonfinite_point'; rows.append(row); continue
        scan_normal = None
        if normals is not None:
            scan_normal = normals[row_id]
            if not np.isfinite(scan_normal).all() or np.linalg.norm(scan_normal) < 1e-12:
                row['status'] = 'invalid_scan_normal'; rows.append(row); continue
            scan_normal = scan_normal/np.linalg.norm(scan_normal)
        candidates = index.query(point, cfg['max_distance_mm'])
        row['candidate_count'] = len(candidates)
        if len(candidates) > cfg['max_candidates']:
            row['status'] = 'candidate_overflow'; rows.append(row); continue
        matches, failures = [], 0
        for face_id in candidates:
            row['exact_projection_count'] += 1
            try:
                projection = project_to_face(point, topology, face_id)
            except (ValueError, RuntimeError):
                failures += 1; continue
            if projection is None or projection['distance_mm'] > cfg['max_distance_mm']:
                continue
            angle = np.nan
            if scan_normal is not None:
                dot = float(scan_normal@projection['normal'])
                if not cfg['normal_orientation_known']:
                    dot = abs(dot)
                angle = float(np.degrees(np.arccos(np.clip(dot, -1., 1.))))
                if angle > cfg['max_normal_angle_deg']:
                    continue
            projection.update(face_id=face_id, scan_normal_angle_deg=angle)
            matches.append(projection)
        if not matches:
            row['status'] = ('projection_failure' if failures else
                             'outside_trim_or_normal_gate' if candidates else 'no_candidate')
            rows.append(row); continue
        matches.sort(key=lambda r: (r['distance_mm'], r['face_id']))
        best = matches[0]
        q, n = best['projected'], best['normal']
        row.update(candidate_face_id=best['face_id'], u=best['u'], v=best['v'],
                   distance_mm=best['distance_mm'], signed_deviation_mm=float((point-q)@n),
                   scan_normal_angle_deg=best['scan_normal_angle_deg'],
                   **dict(zip([f'projected_{a}_mm' for a in 'xyz'], q)),
                   **dict(zip([f'normal_{a}' for a in 'xyz'], n)))
        if len(matches) > 1:
            margin = matches[1]['distance_mm']-best['distance_mm']
            row.update(second_distance_mm=matches[1]['distance_mm'], competition_margin_mm=margin)
            if margin <= cfg['ambiguity_margin_mm']:
                row['status'] = 'ambiguous_faces'; rows.append(row); continue
        try:
            boundary = boundary_distance(q, topology, best['face_id'])
        except RuntimeError:
            row['status'] = 'boundary_distance_failure'; rows.append(row); continue
        row['boundary_distance_mm'] = boundary
        if (best['on_trim_boundary'] and boundary <= 1e-7) or boundary < cfg['boundary_margin_mm']:
            row['status'] = 'boundary_uncertain'; rows.append(row); continue
        score = max(0., 1.-best['distance_mm']/cfg['max_distance_mm'])
        if len(matches) > 1:
            score *= min(1., row['competition_margin_mm']/max(2*cfg['ambiguity_margin_mm'], 1e-12))
        if scan_normal is not None:
            score *= max(0., 1.-best['scan_normal_angle_deg']/cfg['max_normal_angle_deg'])
        row.update(face_id=best['face_id'], valid=True, status='assigned', confidence=float(score))
        rows.append(row)
    # Preserve a stable schema even for an empty input table.
    if not rows:
        return segment_points(np.full((1,3), np.nan), topology, cfg).iloc[:0]
    return pd.DataFrame(rows)
