"""Offline views of CAD reference paths and constrained scan mappings.

Rendering meshes are simplified only for display. Lines are split at invalid
samples and segment boundaries; a rejected mapping is never shown as contact.
All rendered coordinates are CAD coordinates in millimetres.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go

CAD_COLOR = '#a3adb8'
REFERENCE_COLOR = '#d97900'
MAPPED_COLOR = '#008e69'
REJECTED_COLOR = '#c52b4a'
NORMAL_COLOR = '#2377bd'
TOOL_COLOR = '#823ab5'
SCAN_COLOR = '#547990'


def _thin(points, limit):
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > limit:
        points = points[np.linspace(0, len(points)-1, limit).astype(int)]
    return points


def _display_mesh(mesh_path, output_dir, target):
    """Cache deterministic display-only quadric decimation, never face sampling."""
    import open3d as o3d
    mesh_path = Path(mesh_path)
    stat = mesh_path.stat()
    signature = {'source': str(mesh_path.resolve()), 'size': stat.st_size,
                 'mtime_ns': stat.st_mtime_ns, 'target_triangles': target}
    cache = Path(output_dir)/'display_mesh.npz'
    metadata = Path(output_dir)/'display_mesh.json'
    if cache.exists() and metadata.exists() and json.loads(metadata.read_text()) == signature:
        with np.load(cache) as stored:
            return stored['vertices'].copy(), stored['triangles'].copy()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not len(mesh.triangles):
        raise ValueError('Display input must be a nonempty triangle mesh')
    if len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles).copy()
    np.savez_compressed(cache, vertices=vertices, triangles=triangles)
    metadata.write_text(json.dumps(signature, indent=2)+'\n')
    return vertices, triangles


def _runs(table):
    """Yield valid contiguous samples; retain gaps even with reused segment IDs."""
    if not len(table):
        return
    for _, path in table.groupby('path_id', sort=False):
        path = path.sort_values('sample_index')
        current, previous, segment = [], None, None
        for idx, row in path.iterrows():
            good = bool(row.valid) and np.isfinite([row.mapped_x_mm, row.mapped_y_mm, row.mapped_z_mm]).all()
            if not good or (current and (row.segment_id != segment or row.sample_index != previous+1)):
                if current:
                    yield path.loc[current]
                current = []
            if good:
                current.append(idx)
                previous, segment = row.sample_index, row.segment_id
        if current:
            yield path.loc[current]


def _coords(frame, prefix):
    return frame[[f'{prefix}_{a}_mm' for a in 'xyz']].to_numpy(float)


def _normals(frame, prefix):
    return frame[[f'{prefix}_{a}' for a in 'xyz']].to_numpy(float)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    size = np.linalg.norm(vector)
    return vector/size if size > 1e-12 else np.array([0., 0., 1.])


def _equal(ax, points, minimum_span=1.):
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        points = np.zeros((1, 3))
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi-lo, minimum_span)
    mid = (lo+hi)/2
    padding = .06*max(span)
    span += 2*padding
    for setter, c, d in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid, span):
        setter(c-d/2, c+d/2)
    ax.set_box_aspect(span)
    ax.set(xlabel='CAD X (mm)', ylabel='CAD Y (mm)', zlabel='CAD Z (mm)')
    ax.grid(alpha=.12)
    ax.tick_params(labelsize=8, pad=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
        axis.pane.set_alpha(.04)
    ax.set_proj_type('ortho')


def _view(ax, outward):
    n = _unit(outward)
    elev = float(np.clip(np.degrees(np.arcsin(n[2])), -65, 65))
    azim = -70. if abs(n[2]) > .9 else float(np.degrees(np.arctan2(n[1], n[0]))-30)
    if abs(elev) < 20.:
        elev = 25.
    ax.view_init(elev=elev, azim=azim)


def _mesh_collection(ax, vertices, triangles, alpha=.16):
    artist = Poly3DCollection(vertices[triangles], facecolor=CAD_COLOR,
                              edgecolor='none', alpha=alpha, rasterized=True)
    ax.add_collection3d(artist)


def _line_trace(points, name, color, width=5, dash=None, **kwargs):
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                        mode='lines', name=name,
                        line=dict(color=color, width=width, dash=dash), **kwargs)


def _arrow_trace(origins, directions, name, color, length):
    ends = origins+length*directions
    points = np.full((len(origins)*3, 3), np.nan)
    points[::3], points[1::3] = origins, ends
    return _line_trace(points, name, color, width=4)


def _figure_legend():
    return [Line2D([0], [0], color=CAD_COLOR, lw=7, alpha=.5, label='CAD surface'),
            Line2D([0], [0], color=REFERENCE_COLOR, lw=2.5, ls='--', label='CAD reference path'),
            Line2D([0], [0], color=MAPPED_COLOR, lw=3, label='Accepted scan-surface path'),
            Line2D([0], [0], marker='x', color=REJECTED_COLOR, lw=0, label='Rejected / unsupported sample'),
            Line2D([0], [0], color=NORMAL_COLOR, lw=2, label='Oriented surface normal'),
            Line2D([0], [0], color=TOOL_COLOR, lw=2, label='Tool +Z toward surface')]


def write_mapping_visuals(output_dir, mesh_path, scan_points_C, reference_paths, table, cfg, report):
    """Write three PNGs and embedded offline HTML, returning their filenames.

    Required table fields are documented by the CLI's exported mapping table.
    Optional display settings: ``display_mesh_triangles`` and
    ``axis_arrow_length_mm``. Normal signs are assumed, not physically verified.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, triangles = _display_mesh(mesh_path, output_dir,
                                        int(cfg.get('display_mesh_triangles', 35000)))
    scan = _thin(scan_points_C, 24000)
    valid = table[table.valid.astype(bool)].copy()
    references = np.vstack([p['points_C'] for p in reference_paths])
    original_normals = np.vstack([p['normals_C'] for p in reference_paths])
    outward = np.mean(original_normals, axis=0)
    arrow_length = float(cfg.get('axis_arrow_length_mm', 3.))
    runs = list(_runs(table))
    rejected = table[~table.valid.astype(bool)]
    reference_rejected = _coords(rejected, 'cad')

    figure = go.Figure()
    figure.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], name='CAD surface',
        color=CAD_COLOR, opacity=.28, flatshading=False, showlegend=True,
        hoverinfo='skip', lighting=dict(ambient=.7, diffuse=.5)))
    figure.add_trace(go.Scatter3d(x=scan[:, 0], y=scan[:, 1], z=scan[:, 2], mode='markers',
        name='Aligned scan (display sample)', visible='legendonly',
        marker=dict(size=1.5, color=SCAN_COLOR, opacity=.35), hoverinfo='skip'))
    for number, path in enumerate(reference_paths):
        points = np.asarray(path['points_C'])
        figure.add_trace(_line_trace(points, f"CAD path {path['path_id']}", REFERENCE_COLOR,
            width=5, dash='dash', legendgroup='reference',
            hovertemplate=f"Path {path['path_id']}<br>CAD (%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}}) mm<extra></extra>"))
    for number, run in enumerate(runs):
        points = _coords(run, 'mapped')
        custom = np.column_stack([run.sample_index, run.offset_mm, run.fit_rms_mm, run.support_count])
        figure.add_trace(_line_trace(points, 'Accepted scan-surface path', MAPPED_COLOR,
            width=7, showlegend=number == 0, legendgroup='mapped', customdata=custom,
            hovertemplate='Sample %{customdata[0]:.0f}<br>Mapped CAD (%{x:.3f}, %{y:.3f}, %{z:.3f}) mm'
                          '<br>Normal offset=%{customdata[1]:.4f} mm'
                          '<br>Local fit RMS=%{customdata[2]:.4f} mm'
                          '<br>Support=%{customdata[3]:.0f}<extra></extra>'))
    if len(reference_rejected):
        figure.add_trace(go.Scatter3d(x=reference_rejected[:, 0], y=reference_rejected[:, 1],
            z=reference_rejected[:, 2], mode='markers', name='Rejected / unsupported CAD sample',
            text=rejected.status.astype(str), marker=dict(size=4, symbol='x', color=REJECTED_COLOR),
            hovertemplate='%{text}<extra></extra>'))
    arrow_rows = valid.iloc[np.linspace(0, len(valid)-1, min(24, len(valid))).astype(int)]
    if len(arrow_rows):
        origins = _coords(arrow_rows, 'mapped')
        figure.add_trace(_arrow_trace(origins, _normals(arrow_rows, 'normal'),
                                     'Oriented surface normal (assumed side)', NORMAL_COLOR, arrow_length*.45))
        # Tool arrows start above contact so +Z visibly approaches the surface.
        tool_z = _normals(arrow_rows, 'tool_z')
        figure.add_trace(_arrow_trace(origins-arrow_length*tool_z, tool_z,
                                     'Tool +Z toward surface', TOOL_COLOR, arrow_length*.45))
    figure.update_layout(title='CAD reference and scan-adapted paths — CAD coordinates (mm)'
                         '<br><sup>Orange: CAD reference; green: accepted local mapping; red: rejected. '
                         'Normal side assumed; offline geometry only.</sup>',
        template='plotly_white', scene=dict(xaxis_title='CAD X (mm)', yaxis_title='CAD Y (mm)',
        zaxis_title='CAD Z (mm)', aspectmode='data',
        camera=dict(eye=dict(zip('xyz', (_unit(outward)*2+[.25, -.3, 0]).tolist())))),
        margin=dict(l=0, r=0, t=85, b=0), height=850,
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,.8)'))
    figure.write_html(output_dir/'CAD_scan_paths.html', include_plotlyjs=True,
                      config={'displaylogo': False, 'scrollZoom': True})

    fig = plt.figure(figsize=(14, 9), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    _mesh_collection(ax, vertices, triangles)
    for path in reference_paths:
        points = np.asarray(path['points_C'])
        ax.plot(*points.T, color=REFERENCE_COLOR, ls='--', lw=2.4)
        ax.text(*points[len(points)//2], f" {path['path_id']}", fontsize=10, color='#774700')
    for run in runs:
        ax.plot(*_coords(run, 'mapped').T, color=MAPPED_COLOR, lw=2.8)
    if len(reference_rejected):
        ax.scatter(*reference_rejected.T, color=REJECTED_COLOR, s=8, marker='x', depthshade=False)
    if len(arrow_rows):
        origins, normals = _coords(arrow_rows, 'mapped'), _normals(arrow_rows, 'normal')
        ax.quiver(*origins.T, *normals.T, color=NORMAL_COLOR,
                  length=arrow_length, normalize=True, arrow_length_ratio=.25, linewidth=1.2)
    axis_bounds = [vertices]
    if len(arrow_rows):
        axis_bounds.append(origins+arrow_length*normals)
    _equal(ax, np.vstack(axis_bounds))
    _view(ax, outward)
    ax.set_title(f'CAD-selected reference paths and scan mapping\n'
                 f'{len(reference_paths)} paths; {len(valid)}/{len(table)} accepted samples', pad=18)
    ax.legend(handles=_figure_legend()[:5], loc='upper left', fontsize=9)
    fig.text(.5, .012, 'CAD frame, mm. Red samples are not connected. Geometric preview; tool/material side unverified.',
             ha='center', fontsize=10, color='#555555')
    fig.savefig(output_dir/'CAD_path_overview.png', dpi=190, facecolor='white')
    plt.close(fig)

    # Prefer a supported curved path so the figure demonstrates changing axes.
    # Flat paths tie by accepted length. Every rejected sample remains visible.
    if len(valid):
        def pose_variation(group):
            normals = _normals(group, 'normal')
            mean = _unit(normals.mean(axis=0))
            variation = float(np.max(np.degrees(np.arccos(np.clip(normals@mean, -1, 1)))))
            return (variation >= 2., variation if variation >= 2. else 0., len(group))
        representative_id = max(valid.groupby('path_id', sort=False), key=lambda item: pose_variation(item[1]))[0]
    else:
        representative_id = reference_paths[0]['path_id']
    part = table[table.path_id == representative_id].sort_values('sample_index')
    part_runs = list(_runs(part))
    reference = _coords(part, 'cad')
    reference_definition = next(p for p in reference_paths if p['path_id'] == representative_id)
    center = reference.mean(axis=0)
    normal = _unit(np.mean(reference_definition['normals_C'], axis=0))
    _, _, vt = np.linalg.svd(reference-center, full_matrices=False)
    along = _unit(vt[0]-np.dot(vt[0], normal)*normal)
    if np.dot(along, reference[-1]-reference[0]) < 0:
        along = -along
    across = _unit(np.cross(normal, along))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=1))]
    arc_by_sample = dict(zip(part.sample_index, arc))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')
    ax = axes[0, 0]
    local = reference-center
    ax.plot(local@along, local@across, ls='--', color=REFERENCE_COLOR, lw=2, label='CAD reference')
    for number, run in enumerate(part_runs):
        local = _coords(run, 'mapped')-center
        ax.plot(local@along, local@across, color=MAPPED_COLOR, lw=2,
                label='Accepted scan mapping' if number == 0 else None)
    bad = ~part.valid.astype(bool).to_numpy()
    if bad.any():
        bad_points = reference[bad]-center
        ax.scatter(bad_points@along, bad_points@across, c=REJECTED_COLOR, marker='x', s=28,
                   label='Rejected CAD sample', zorder=4)
    ax.set(xlabel='Along-path tangent coordinate (mm)', ylabel='Cross-path coordinate (mm)',
           title='Path footprint in local CAD tangent plane')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(arc, np.zeros(len(arc)), color=REFERENCE_COLOR, ls='--', label='CAD reference: zero')
    for number, run in enumerate(part_runs):
        x = [arc_by_sample[v] for v in run.sample_index]
        ax.plot(x, run.offset_mm, color=MAPPED_COLOR, label='Normal offset to fitted scan' if number == 0 else None)
    ax.set(xlabel='Distance along CAD reference (mm)', ylabel='Oriented offset (mm)',
           title='Local correction; sign uses assumed CAD side')
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for number, run in enumerate(part_runs):
        x = [arc_by_sample[v] for v in run.sample_index]
        ax.plot(x, run.fit_rms_mm, color=NORMAL_COLOR, label='Local support fit RMS' if number == 0 else None)
    ax.set(xlabel='Distance along CAD reference (mm)', ylabel='Fit RMS (mm)',
           title='Local surface consistency (not physical positioning accuracy)')
    if part_runs:
        ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(arc, part.support_count, color='#4b5660', label='Retained local support')
    if bad.any():
        ax.scatter(arc[bad], part.support_count.to_numpy()[bad], color=REJECTED_COLOR,
                   marker='x', s=25, label='Rejected sample')
    ax.set(xlabel='Distance along CAD reference (mm)', ylabel='Support points',
           title='Support availability and rejected intervals')
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f'CAD-to-scan mapping detail — {representative_id}\n'
                 'Gaps remain gaps; no interpolation across rejected samples', fontsize=14)
    fig.savefig(output_dir/'path_mapping_detail.png', dpi=180, facecolor='white')
    plt.close(fig)

    # Reserve a header: projected 3D labels extend beyond the axes rectangle.
    fig = plt.figure(figsize=(12, 9))
    fig.subplots_adjust(left=.03, right=.88, bottom=.10, top=.76)
    ax = fig.add_subplot(111, projection='3d')
    if len(valid):
        from planning.scan_mapping import contact_frames
        accepted_part = part[part.valid.astype(bool)]
        midpoint = accepted_part.iloc[len(accepted_part)//2]
        origin = np.array([midpoint[f'mapped_{a}_mm'] for a in 'xyz'])
        # Derive frames on each entire accepted segment before thinning. This is
        # the identical tangent/roll convention used by export_contact_poses.
        frame_by_index = {}
        for run in part_runs:
            if len(run) < 2:
                continue
            frames, _ = contact_frames(_coords(run, 'mapped'), _normals(run, 'normal'))
            for index, frame in zip(run.index, frames):
                frame_by_index[index] = frame
        selectable = accepted_part.loc[accepted_part.index.isin(frame_by_index)]
        chosen = selectable.iloc[np.linspace(0, len(selectable)-1, min(8, len(selectable))).astype(int)]
        sample_points = _coords(chosen, 'mapped')
        normals = _normals(chosen, 'normal')
        frames = np.array([frame_by_index[index] for index in chosen.index])
        scale = min(arrow_length, max(.25, arc[-1]/(2*max(len(chosen), 1))))
        near_scan = scan[np.linalg.norm(scan-origin, axis=1) < max(12., scale*4)]
        if len(near_scan):
            ax.scatter(*near_scan.T, s=3, color=SCAN_COLOR, alpha=.2, label='Aligned scan')
        for run in part_runs:
            ax.plot(*_coords(run, 'mapped').T, color=MAPPED_COLOR, lw=2)
        ax.plot(*reference.T, color=REFERENCE_COLOR, ls='--', lw=1.6)
        axis_ends = [reference]
        if len(chosen):
            ax.quiver(*sample_points.T, *normals.T, color=NORMAL_COLOR, length=scale*.7,
                      normalize=True, arrow_length_ratio=.25, linewidth=1.2)
            axis_ends.append(sample_points+scale*.7*normals)
            for column, color in [(0, '#c63a36'), (1, '#16834d'), (2, TOOL_COLOR)]:
                direction = frames[:, :, column]
                ax.quiver(*sample_points.T, *direction.T, color=color, length=scale,
                          normalize=True, arrow_length_ratio=.24, linewidth=1.8)
                axis_ends.append(sample_points+scale*direction)
            ax.scatter(*sample_points.T, color='#222222', s=12)
        _equal(ax, np.vstack(axis_ends))
    else:
        _equal(ax, reference)
        ax.text(*center, 'No accepted scan mapping; no mapped tool axes', color=REJECTED_COLOR)
    _view(ax, normal)
    fig.suptitle(f'Full contact-frame poses along {representative_id}\n'
                 'X: forward tangent; Y: right-handed cross axis; +Z: toward surface', y=.975)
    triad_legend = [Line2D([0], [0], color='#c63a36', lw=2, label='Tool X: forward tangent'),
                    Line2D([0], [0], color='#16834d', lw=2, label='Tool Y'),
                    Line2D([0], [0], color=TOOL_COLOR, lw=2, label='Tool +Z: toward surface'),
                    Line2D([0], [0], color=NORMAL_COLOR, lw=2, label='Oriented CAD normal (assumed side)')]
    ax.legend(handles=_figure_legend()[1:3]+triad_legend, loc='upper left', fontsize=9)
    fig.text(.5, .012, 'Origins are surface contact points. Triads match the exported pose convention. CAD frame, mm.',
             ha='center', fontsize=10, color='#555555')
    fig.savefig(output_dir/'pose_axes_detail.png', dpi=190, facecolor='white')
    plt.close(fig)
    return ['CAD_scan_paths.html', 'CAD_path_overview.png', 'path_mapping_detail.png', 'pose_axes_detail.png']
