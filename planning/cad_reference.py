"""Exact trimmed STEP-face curves for surface coverage (all lengths in mm).

CAD parameters stay internal to the geometric solver. Public contact poses are
converted to the confirmed robot base coordinate system by the frame layer.
"""
import numpy as np


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
