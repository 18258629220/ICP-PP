"""Focused CAD/scan path comparisons from saved, unmodified BASE results."""
from pathlib import Path
import json
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError, cKDTree

from pipeline_common import dump_json, sha256, transform_points
from planning.real_presentation import _validate_base_poses


POSITION = ['x_mm', 'y_mm', 'z_mm']
REFERENCE = [f'reference_{c}_mm' for c in 'xyz']
CAD_COLOR, SCAN_COLOR = '#dc820c', '#008b7a'
AXIS_COLORS = ['#cc343b', '#21894e', '#9439ba']


def split_lines(points, valid=None, segments=None, sample_indices=None):
    """Insert NaNs at rejected rows, segment changes and missing source samples."""
    points = np.asarray(points, float)
    valid = np.isfinite(points).all(axis=1) if valid is None else np.asarray(valid, bool)
    parts = []
    for i, point in enumerate(points):
        if not valid[i]:
            parts.append(np.full(3, np.nan))
            continue
        if i and (not valid[i-1]
                  or (segments is not None and segments[i] != segments[i-1])
                  or (sample_indices is not None and sample_indices[i] != sample_indices[i-1]+1)):
            parts.append(np.full(3, np.nan))
        parts.append(point)
    return np.asarray(parts, float).reshape(-1, 3)


def _sweep(normals):
    n = normals[np.unique(np.linspace(0, len(normals)-1, min(40, len(normals))).astype(int))]
    return float(np.degrees(np.arccos(np.clip(n @ n.T, -1., 1.))).max()) if len(n) else 0.


def select_paths(field, count=3, path_ids=None):
    """Prefer long, observed, curved CAD paths; measured noise does not rank curvature."""
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 3:
        raise ValueError('Select one to three paths')
    if path_ids:
        if len(path_ids) > 3 or len(set(path_ids)) != len(path_ids):
            raise ValueError('Select one to three different path IDs')
        missing = set(path_ids)-set(field.path_id)
        if missing:
            raise ValueError(f'Unknown path IDs: {sorted(missing)}')
        if any(not field.loc[field.path_id == key, 'valid'].any() for key in path_ids):
            raise ValueError('Selected paths must contain accepted contact samples')
        return list(path_ids)
    records = []
    for key, group in field.groupby('path_id', sort=False):
        good = group[group.valid]
        if len(good) < 3:
            continue
        length = sum(np.linalg.norm(np.diff(g[[f'mapped_{c}_mm' for c in 'xyz']], axis=0), axis=1).sum()
                     for _, g in good.groupby('segment_id', sort=False))
        sweep = _sweep(good[[f'reference_normal_{c}' for c in 'xyz']].to_numpy(float))
        # A broad curved stroke is easier to assess than a tiny high-curvature remnant.
        substantial = length >= 25 and len(good)/len(group) >= .7
        records.append((substantial, np.sqrt(length)*max(sweep, 1.), length,
                        str(key), int(group.face_id.iloc[0])))
    records.sort(reverse=True)
    selected, faces = [], set()
    for distinct in [True, False]:
        for _, _, _, key, face in records:
            if key in selected or (distinct and face in faces):
                continue
            selected.append(key)
            faces.add(face)
            if len(selected) == count:
                return selected
    return selected


def _compact_mesh(vertices, triangles, uv=None):
    ids, inverse = np.unique(np.asarray(triangles).reshape(-1), return_inverse=True)
    result = {'vertices': vertices[ids], 'triangles': inverse.reshape(-1, 3)}
    if uv is not None:
        result['uv'] = uv[ids]
    return result


def _scan_surface(points, uv, cad, max_edge_mm=1.5):
    """Display-only triangulation of original same-face points, never CAD-displaced data.

    Short 3D edges limit filling scan gaps. UV centroid/midpoint tests against
    the actual trimmed CAD mesh further reject triangles outside its domain.
    If its UV mesh is unsuitable (e.g. an overlapping periodic seam), show points.
    """
    empty = {'vertices': points, 'triangles': np.empty((0, 3), int)}
    if len(points) < 3 or not len(cad['triangles']):
        return empty, 'insufficient_points'
    scale = np.ptp(cad['uv'], axis=0)
    if np.any(scale < 1e-12):
        return empty, 'degenerate_uv'
    origin = cad['uv'].min(axis=0)
    normalized = (uv-origin)/scale
    _, unique = np.unique(np.round(normalized, 12), axis=0, return_index=True)
    points, normalized = points[unique], normalized[unique]
    try:
        # Parameter units may differ by orders of magnitude. A physical UV
        # metric prevents skinny triangulation along a narrow curved strip.
        jacobian = np.linalg.lstsq(normalized-normalized.mean(axis=0),
                                   points-points.mean(axis=0), rcond=None)[0]
        metric = np.linalg.cholesky(jacobian @ jacobian.T)
        triangles = Delaunay((normalized-normalized.mean(axis=0)) @ metric).simplices
        edges = np.stack([points[triangles[:, 1]]-points[triangles[:, 0]],
                          points[triangles[:, 2]]-points[triangles[:, 1]],
                          points[triangles[:, 0]]-points[triangles[:, 2]]], axis=1)
        triangles = triangles[(np.linalg.norm(edges, axis=2) <= max_edge_mm).all(axis=1)]
        import matplotlib.tri as mtri
        cad_uv = (cad['uv']-origin)/scale
        cad_uv, inverse = np.unique(np.round(cad_uv, 12), axis=0, return_inverse=True)
        ct = inverse[cad['triangles']]
        a, b = cad_uv[ct[:, 1]]-cad_uv[ct[:, 0]], cad_uv[ct[:, 2]]-cad_uv[ct[:, 0]]
        area = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
        ct = ct[np.abs(area) > 1e-14].copy()
        # TriFinder requires consistently counterclockwise UV cells.
        flip = area[np.abs(area) > 1e-14] < 0
        ct[flip] = ct[flip][:, [0, 2, 1]]
        finder = mtri.Triangulation(*cad_uv.T, triangles=ct).get_trifinder()
        tri_uv = normalized[triangles]
        checks = np.stack([tri_uv.mean(axis=1), (tri_uv[:, 0]+tri_uv[:, 1])/2,
                           (tri_uv[:, 1]+tri_uv[:, 2])/2, (tri_uv[:, 2]+tri_uv[:, 0])/2], axis=1)
        inside = finder(checks[:, :, 0], checks[:, :, 1]) >= 0
        triangles = triangles[inside.all(axis=1)]
    except (QhullError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        warnings.warn(f'Scan surface shown as points: {error}', RuntimeWarning)
        return empty, 'uv_triangulation_unavailable'
    return {'vertices': points, 'triangles': triangles}, 'observed_points_short_edge_uv_clipped'


def _cad_axes(reference, normals):
    tangent = np.gradient(reference, axis=0) if len(reference) > 1 else np.zeros_like(reference)
    z = -normals
    x = tangent-(tangent*z).sum(axis=1)[:, None]*z
    norm = np.linalg.norm(x, axis=1)
    x = np.divide(x, norm[:, None], out=np.full_like(x, np.nan), where=norm[:, None] > 1e-10)
    return np.stack([x, np.cross(z, x), z], axis=2)


def _camera(path):
    # View from the observed side with a tangential component to reveal Z arrows.
    axes = path['axes'][path['valid']]
    outward = -axes[:, :, 2].mean(axis=0)
    outward /= max(np.linalg.norm(outward), 1e-10)
    tangent = axes[:, :, 0].mean(axis=0)
    tangent -= np.dot(tangent, outward)*outward
    tangent /= max(np.linalg.norm(tangent), 1e-10)
    across = np.cross(outward, tangent)
    if across[2] < 0:
        across = -across
    eye = 1.1*outward+.8*across+.25*tangent
    return dict(zip('xyz', eye)), outward


def build_path_bundle(run_dir, path_ids=None, count=3, axis_length_mm=None):
    run = Path(run_dir).resolve()
    if axis_length_mm is not None and (not np.isfinite(axis_length_mm) or axis_length_mm <= 0):
        raise ValueError('Axis length must be positive and finite')
    report = json.loads((run/'summary.json').read_text())
    if report.get('output_frame') != 'ROBOT_BASE' or report.get('units') != 'mm':
        raise ValueError('Saved run must use ROBOT_BASE and mm')
    poses = pd.read_csv(run/'contact_poses_BASE.csv')
    _validate_base_poses(poses, run)
    field = pd.read_csv(run/'surface_field_BASE.csv')
    refs = pd.read_csv(run/'cad_reference_BASE.csv')
    for name, table in [('field', field), ('reference', refs)]:
        if not table.frame.eq('ROBOT_BASE').all() or table.duplicated(['path_id', 'sample_index']).any():
            raise ValueError(f'Mixed frame or duplicate path/sample in {name}')
    if poses.duplicated(['path_id', 'sample_index']).any():
        raise ValueError('Duplicate contact pose correspondence')
    reference_check = field.merge(refs, on=['path_id', 'sample_index'], validate='one_to_one')
    if len(reference_check) != len(field):
        raise ValueError('Missing CAD reference samples')
    np.testing.assert_allclose(reference_check[REFERENCE], reference_check[POSITION], atol=1e-8, rtol=0)
    np.testing.assert_array_equal(reference_check.face_id_x, reference_check.face_id_y)
    with np.load(run/'cad_face_mesh_BASE.npz', allow_pickle=False) as m:
        if str(m['frame']) != 'ROBOT_BASE':
            raise ValueError('CAD mesh must already be in ROBOT_BASE')
        vertices, triangles, triangle_faces, uv = [m[k] for k in
            ['vertices', 'triangles', 'triangle_face_ids', 'vertex_uv']]
    support_dir = Path(report['segmentation_directory'])
    if not support_dir.is_absolute():
        support_dir = run/support_dir
    record = json.loads((support_dir/'segmentation_summary.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    for key in ['step', 'scan', 'transform']:
        if record['input_sha256'][key] != manifest['input_sha256'][key]:
            raise ValueError('Observed support provenance disagrees with the saved run')
    if record.get('support_frame', 'CAD_LOCAL') != 'CAD_LOCAL':
        raise ValueError('Support cache requires the saved CAD-to-BASE transform')
    with np.load(support_dir/'segmentation_support.npz', allow_pickle=False) as s:
        support = transform_points(s['support_points'], np.load(run/'T_CAD_to_BASE.npy'))
        support_faces, support_uv = s['support_faces'], s['support_uv']
    # Global locator uses observed support, with no stray unassigned background.
    overview = support[np.unique(np.linspace(0, len(support)-1, min(25000, len(support))).astype(int))]
    selected = select_paths(field, count=count, path_ids=path_ids)
    paths = []
    for key in selected:
        g = field[field.path_id == key].sort_values('sample_index').reset_index(drop=True)
        fid = int(g.face_id.iloc[0])
        p = poses[poses.path_id == key].set_index('sample_index').reindex(g.sample_index)
        valid = g.valid.to_numpy(bool)
        if p.frame.notna().sum() != valid.sum() or not np.array_equal(p.frame.notna(), valid):
            raise ValueError(f'Accepted sample mismatch: {key}')
        contact = p[POSITION].to_numpy(float)
        np.testing.assert_allclose(contact[valid], g.loc[valid, [f'mapped_{c}_mm' for c in 'xyz']], atol=1e-8, rtol=0)
        np.testing.assert_array_equal(p.loc[valid, 'segment_id'].to_numpy(int), g.loc[valid, 'segment_id'])
        np.testing.assert_array_equal(p.loc[valid, 'face_id'].to_numpy(int), g.loc[valid, 'face_id'])
        axes = np.stack([p[[f'tool_{a}_{c}' for c in 'xyz']].to_numpy(float) for a in 'xyz'], axis=2)
        ref = g[REFERENCE].to_numpy(float)
        # Crop meshes to a band around the path; full trimmed face triangles are retained within it.
        tree = cKDTree(ref)
        local_ids = np.flatnonzero(support_faces == fid)
        local_ids = local_ids[tree.query(support[local_ids])[0] <= 5.]
        q, quv = support[local_ids], support_uv[local_ids]
        face_tri = triangles[triangle_faces == fid]
        close = tree.query(vertices[np.unique(face_tri)])[0] <= 6.
        near_ids = np.unique(face_tri)[close]
        keep = np.isin(face_tri, near_ids).any(axis=1)
        cad = _compact_mesh(vertices, face_tri[keep], uv)
        scan, surface_status = _scan_surface(q, quv, cad)
        # Sample approximately uniformly in arc length, including each segment's interior.
        arrow_ids = []
        for _, segment in g[valid].groupby('segment_id', sort=False):
            ix = segment.index.to_numpy()
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(contact[ix], axis=0), axis=1))]
            n = max(1, min(9, int(np.ceil(distance[-1]/6.))))
            arrow_ids.extend(ix[np.searchsorted(distance, np.linspace(0, distance[-1], n+2)[1:-1])])
        arrow_ids = np.unique(arrow_ids).astype(int)
        if len(arrow_ids) > 12:
            arrow_ids = arrow_ids[np.linspace(0, len(arrow_ids)-1, 12).astype(int)]
        length = sum(np.linalg.norm(np.diff(contact[s.index], axis=0), axis=1).sum()
                     for _, s in g[valid].groupby('segment_id', sort=False))
        path = dict(path_id=key, label=f'P{len(paths)+1}', face_id=fid, reference=ref,
                    contact=contact, valid=valid, segment_ids=g.segment_id.to_numpy(int),
                    axes=axes, cad_axes=_cad_axes(ref, g[[f'reference_normal_{c}' for c in 'xyz']].to_numpy(float)),
                    sample_indices=g.sample_index.to_numpy(int), offset_mm=g.offset_mm.to_numpy(float),
                    fit_rms_mm=g.fit_rms_mm.to_numpy(float), arrow_indices=arrow_ids,
                    axis_length_mm=float(axis_length_mm or np.clip(length/22., 2., 4.)),
                    cad_surface=cad, scan_surface=scan, scan_points=q,
                    stats=dict(length_mm=float(length), normal_sweep_deg=_sweep(axes[valid, :, 2]),
                               accepted_samples=int(valid.sum()), total_samples=len(g),
                               contact_segments=int(g.loc[valid, 'segment_id'].nunique()),
                               cad_normal_sweep_deg=_sweep(g.loc[valid, [f'reference_normal_{c}' for c in 'xyz']].to_numpy(float)),
                               surface_status=surface_status, scan_triangles=len(scan['triangles'])))
        path['camera'], path['outward'] = _camera(path)
        paths.append(path)
    if not paths:
        raise ValueError('No supported contact paths to display')
    return dict(frame='ROBOT_BASE', units='mm', overview_points=overview, paths=paths,
                source_run=run, support_dir=support_dir)


def _plot_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # TTC font discovery may expose only its first locale; register the bundled
    # host CJK font explicitly instead of relying on a stale font cache.
    noto = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if noto.exists():
        font_manager.fontManager.addfont(str(noto))
        family = font_manager.FontProperties(fname=str(noto)).get_name()
        plt.rcParams['font.sans-serif'] = [family, 'DejaVu Sans']
    else:
        for font in ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'AR PL UMing CN']:
            if any(f.name == font for f in font_manager.fontManager.ttflist):
                plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10})
    return plt


def _finish_axis(ax, path, points, margin=0.):
    from matplotlib.ticker import FixedLocator, MaxNLocator, ScalarFormatter
    ax.set_proj_type('ortho')
    eye = np.array([path['camera'][c] for c in 'xyz'])
    ax.view_init(elev=float(np.degrees(np.arcsin(eye[2]/np.linalg.norm(eye)))),
                 azim=float(np.degrees(np.arctan2(eye[1], eye[0]))), roll=-25)
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi-lo, 1.)+2*margin
    center = (lo+hi)/2
    for setter, mid, width in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center, span):
        setter(mid-.56*width, mid+.56*width)
    ax.set_box_aspect(span.copy(), zoom=.94)
    ax.set(xlabel='BASE X / mm', ylabel='BASE Y / mm', zlabel='BASE Z / mm')
    ax.tick_params(labelsize=8, pad=0)
    for i, axis in enumerate([ax.xaxis, ax.yaxis, ax.zaxis]):
        # A physically thin surface band has little room for tick labels.
        axis.set_major_locator(FixedLocator([center[i]]) if span[i] < .12*span.max() else MaxNLocator(4))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
        axis.pane.fill = False
        axis._axinfo['grid']['color'] = (.82, .85, .89, .4)


def _surface(ax, mesh, color, alpha):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if len(mesh['triangles']):
        poly = Poly3DCollection(mesh['vertices'][mesh['triangles']], facecolor=color,
                                alpha=alpha, edgecolor='none', rasterized=True, zorder=1)
        ax.add_collection3d(poly)


def _triads(ax, path, *, cad=False, indices=None, length=None):
    indices = path['arrow_indices'] if indices is None else indices
    p = path['reference' if cad else 'contact'][indices]
    axes = path['cad_axes' if cad else 'axes'][indices]
    length = path['axis_length_mm'] if length is None else length
    for i, color in enumerate(AXIS_COLORS):
        vectors = axes[:, :, i]
        # +Z shaft lies outside the surface and points to the exact contact origin.
        start = p-length*vectors if i == 2 else p
        ax.quiver(*start.T, *vectors.T, length=length, color=color,
                  arrow_length_ratio=.24, linewidth=1.5, normalize=False, zorder=8)


def write_static_views(dest, bundle):
    plt = _plot_setup()
    from matplotlib.lines import Line2D
    dest = Path(dest)
    legends = [Line2D([0], [0], color=CAD_COLOR, ls='--', lw=2.5, label='CAD 参考路径'),
               Line2D([0], [0], color=SCAN_COLOR, lw=2.8, label='实测接触路径'),
               *[Line2D([0], [0], color=c, lw=2, label=name) for c, name in
                 zip(AXIS_COLORS, ['刀轴 X（切向）', '刀轴 Y', '刀轴 +Z（指向表面）'])]]
    paths = bundle['paths']
    # One whole-workpiece locator, only the selected 2–3 paths highlighted.
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    q = bundle['overview_points']
    ax.scatter(*q.T, s=.6, c='#c2cbd3', alpha=.32, depthshade=False, rasterized=True)
    for path in paths:
        cad = split_lines(path['reference'], sample_indices=path['sample_indices'])
        scan = split_lines(path['contact'], path['valid'], path['segment_ids'], path['sample_indices'])
        ax.plot(*cad.T, color=CAD_COLOR, ls='--', lw=2.5, zorder=4)
        ax.plot(*scan.T, color=SCAN_COLOR, lw=2.7, zorder=5)
        good = path['contact'][path['valid']]
        mid = good[len(good)//2]
        ax.text(*mid, f"  {path['label']} · 面 {path['face_id']}", fontsize=11, weight='bold', zorder=9)
    lo, hi = np.quantile(q, [.001, .999], axis=0)
    shown = q[np.all((q >= lo) & (q <= hi), axis=1)]
    _finish_axis(ax, paths[0], shown)
    ax.set_title(f'{len(paths)} 条代表路径在真实工件上的位置\nCAD 参考路径与实测接触路径均使用 ROBOT_BASE', pad=14, fontsize=15)
    ax.legend(handles=legends[:2], loc='upper left', framealpha=.92)
    fig.subplots_adjust(left=0, right=.95, bottom=.02, top=.9)
    fig.savefig(dest/'selected_paths_overview.png', dpi=180, facecolor='white')
    plt.close(fig)
    # Each row uses identical limits/camera: CAD surface left, measured surface right.
    fig = plt.figure(figsize=(16, 5*len(paths)+1.2), facecolor='white')
    for row, path in enumerate(paths):
        p = path['contact'][path['valid']]
        tips = np.vstack([p, p-path['axis_length_mm']*path['axes'][path['valid'], :, 2]])
        bound = np.vstack([path['reference'], path['scan_points'], tips])
        for col in [0, 1]:
            ax = fig.add_subplot(len(paths), 2, 2*row+col+1, projection='3d', computed_zorder=False)
            if col == 0:
                _surface(ax, path['cad_surface'], '#b6c4d5', .68)
            else:
                _surface(ax, path['scan_surface'], '#83c7bc', .55)
                q = path['scan_points']
                ax.scatter(*q.T, s=.5, c='#247a71', alpha=.14, depthshade=False, rasterized=True, zorder=2)
            cad = split_lines(path['reference'], sample_indices=path['sample_indices'])
            scan = split_lines(path['contact'], path['valid'], path['segment_ids'], path['sample_indices'])
            ax.plot(*cad.T, color=CAD_COLOR, ls='--', lw=2.4, zorder=5)
            if col:
                ax.plot(*scan.T, color=SCAN_COLOR, lw=2.5, zorder=6)
                bad = path['reference'][~path['valid']]
                ax.scatter(*bad.T, marker='x', s=22, c='#bd3053', zorder=7)
            _triads(ax, path, cad=not col)
            _finish_axis(ax, path, bound, margin=.7)
            stat = path['stats']
            ax.set_title(f"{path['label']} · CAD 面 {path['face_id']} · "
                         + ('设计曲面 / CAD 位姿' if col == 0 else '实测点云曲面 / 接触位姿')
                         + f"\n有效 {stat['accepted_samples']}/{stat['total_samples']} · 路径 {stat['length_mm']:.1f} mm · 箭头 {path['axis_length_mm']:.1f} mm",
                         fontsize=12, pad=8)
    fig.suptitle('CAD → 真实扫描曲面：路径与完整刀轴位姿对照', fontsize=18, y=.98)
    fig.legend(handles=legends, loc='upper center', bbox_to_anchor=(.5, .955), ncol=5, frameon=False)
    fig.text(.5, .012, '同一基坐标、相同相机与比例；路径偏差未放大。实测曲面由同面原始支持点局部三角化，仅作显示；缺口不连接。',
             ha='center', fontsize=10, color='#4b5967')
    fig.subplots_adjust(top=.91, bottom=.05, left=.01, right=.96, hspace=.17, wspace=.01)
    fig.savefig(dest/'cad_scan_surface_comparison.png', dpi=170, facecolor='white')
    plt.close(fig)
    # A short curved section per path avoids visually compressing the axis triads.
    for path in paths:
        valid = path['valid']
        groups = [np.flatnonzero(valid & (path['segment_ids'] == seg))
                  for seg in np.unique(path['segment_ids'][valid])]
        ix = max(groups, key=len)
        window = min(len(ix), 17)
        best = max((ix[start:start+window] for start in range(max(1, len(ix)-window+1))),
                   key=lambda part: _sweep(path['axes'][part, :, 2]))
        dense = np.unique(best[np.linspace(1 if len(best) > 2 else 0, max(0, len(best)-2), min(6, len(best))).astype(int)])
        length = min(path['axis_length_mm'], 2.5)
        fig = plt.figure(figsize=(13, 8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        r, p = path['reference'][best], path['contact'][best]
        for name, color, alpha in [('cad_surface', '#b8c5d5', .22), ('scan_surface', '#8bcabd', .3)]:
            mesh = path[name]
            if len(mesh['triangles']):
                near = cKDTree(r).query(mesh['vertices'])[0] <= 3.5
                _surface(ax, {'vertices': mesh['vertices'], 'triangles': mesh['triangles'][near[mesh['triangles']].all(axis=1)]}, color, alpha)
        ax.plot(*r.T, color=CAD_COLOR, ls='--', lw=2.8, zorder=5)
        ax.plot(*p.T, color=SCAN_COLOR, lw=3., zorder=6)
        _triads(ax, path, indices=dense, length=length)
        # Subtle connectors show the actual CAD-to-scan displacement without scaling.
        for j in dense:
            ax.plot(*np.vstack([path['reference'][j], path['contact'][j]]).T,
                    color='#55616e', lw=.65, alpha=.8, zorder=4)
        ends = np.vstack([p, r, p-length*path['axes'][best, :, 2],
                          p+length*path['axes'][best, :, 0], p+length*path['axes'][best, :, 1]])
        _finish_axis(ax, path, ends, margin=1.)
        s = path['sample_indices'][best]
        ax.set_title(f"{path['label']} 刀轴放大 · CAD 面 {path['face_id']} · 样本 {s[0]}–{s[-1]}\n"
                     f'三轴箭头长 {length:.1f} mm；+Z 从曲面外侧指向接触点', fontsize=15, pad=16)
        ax.legend(handles=legends, loc='upper left', fontsize=10)
        fig.text(.5, .035, '橙色虚线 = CAD 参考；青色实线 = 实测接触路径；浅灰连线 = 实际位置差；长度单位 mm，偏差未放大。',
                 ha='center', fontsize=10, color='#4b5967')
        fig.subplots_adjust(left=.02, right=.94, bottom=.09, top=.88)
        fig.savefig(dest/f"{path['label']}_contact_axes_detail.png", dpi=180, facecolor='white')
        plt.close(fig)


def write_path_details(run_dir, dest, path_ids=None, count=3, axis_length_mm=None):
    """Re-render without registration, segmentation or changing exported robot geometry."""
    run, dest = Path(run_dir).resolve(), Path(dest).resolve()
    if dest == run or (dest.exists() and any(dest.iterdir())):
        raise FileExistsError('Use a new empty visualization directory; keep original run intact')
    bundle = build_path_bundle(run, path_ids, count, axis_length_mm)
    dest.mkdir(parents=True, exist_ok=True)
    write_static_views(dest, bundle)
    from planning.path_view_html import write_path_view_html
    write_path_view_html(dest, bundle)
    source_files = ['cad_reference_BASE.csv', 'surface_field_BASE.csv', 'contact_poses_BASE.csv',
                    'cad_face_mesh_BASE.npz', 'T_CAD_to_BASE.npy', 'manifest.json']
    root = Path(__file__).resolve().parents[1]
    dump_json(dest/'visualization_manifest.json', dict(frame='ROBOT_BASE', units='mm',
        source_run=str(run), geometric_results_changed=False, deviations_exaggerated=False,
        scan_surface_purpose='display_only_same_face_observed_points; short edges <=1.5mm; UV mesh sample clipping',
        sources={name: sha256(run/name) for name in source_files},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip()),
        visualization_source_sha256={str(f.relative_to(root)): sha256(f) for f in
            [Path(__file__), root/'planning/path_view_html.py']},
        paths=[dict(path_id=p['path_id'], label=p['label'], face_id=p['face_id'],
                    arrow_length_mm=p['axis_length_mm'], **p['stats']) for p in bundle['paths']]))
    (dest/'README_zh.md').write_text('\n'.join([
        '# 路径与刀轴可视化', '',
        '先打开 PATH_DETAILS.html，可选择 P1/P2/P3、旋转和放大，并切换 CAD/实测曲面及三轴图层。',
        'selected_paths_overview.png 定位选中的三条路径；cad_scan_surface_comparison.png 逐行对照 CAD 与实测曲面。',
        'P1/P2/P3_contact_axes_detail.png 放大连续路径中的一段，显示接触点和完整 XYZ 刀轴。',
        '橙色虚线是 CAD 参考，青色实线是实测路径；红 X、绿 Y、紫 +Z。+Z 箭头尖端落在接触点，箭杆位于曲面外侧。',
        '所有位置和方向为 ROBOT_BASE，单位 mm；箭头长度只控制显示，CAD/实测偏差没有人为放大。',
        '实测曲面由已归属同一 CAD 面的原始扫描支持点局部三角化，仅用于显示，不回写规划和位姿。',
        '长边与 CAD UV 网格采样检查限制补面；它不是重新扫描得到的连续真值，也不是加工模型。',
        '无效样本、不同 segment_id 与缺失序号不连接。悬停读取位姿数值；CSV 是完整导出依据。',
        '原始配准、分割、CSV/NPZ 与工件数据保持原样。', '']), encoding='utf-8')
    return dest
