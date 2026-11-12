"""CAD-distance foreground extraction without moving the robot-base scan.

Open3D's unsigned point-to-triangle distance works on an open mesh. It does
not identify fixture semantics: geometry close to the CAD remains a candidate.
Every retained/rejected row can be traced back to the original scan.
"""
from pathlib import Path
import time

import numpy as np

from pipeline_common import assert_rigid, transform_points, xyz_of


def filter_scan_by_cad(raw, transform, segmenter, dest, threshold_mm,
                       chunk_points=200000):
    """Save full-source distances/mask and ordered retained/rejected indices.

    Distances are evaluated in the internal CAD frame with the existing rigid
    registration. Saved PLY coordinates are never transformed. A missing CAD
    face's nearby bounding box is conservatively protected from deletion; this
    exception is recorded separately and does not establish a valid face label.
    """
    import open3d as o3d

    if not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError('threshold_mm must be finite and positive')
    if isinstance(chunk_points, bool) or not isinstance(chunk_points, int) or chunk_points < 1:
        raise ValueError('chunk_points must be a positive integer')
    if not len(raw):
        raise ValueError('Input scan is empty')
    assert_rigid(transform)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    names = ['cad_distance_mm.npy', 'filter_keep_mask.npy',
             'filter_protected_unmeshed.npy', 'source_indices.npy',
             'removed_source_indices.npy']
    if any((dest/name).exists() for name in names):
        raise FileExistsError('Filter output already exists; use a new run directory')
    distances = np.lib.format.open_memmap(dest/names[0], mode='w+', dtype=np.float32,
                                         shape=(len(raw),))
    keep = np.lib.format.open_memmap(dest/names[1], mode='w+', dtype=bool, shape=(len(raw),))
    protected = np.lib.format.open_memmap(dest/names[2], mode='w+', dtype=bool,
                                         shape=(len(raw),))
    unmeshed = np.asarray(segmenter.unmeshed_face_ids, dtype=np.int64)
    margin = float(threshold_mm)+segmenter.mesh_error_margin_mm
    started = time.perf_counter()
    nonfinite = 0
    query_invalid = 0
    for start in range(0, len(raw), chunk_points):
        stop = min(start+chunk_points, len(raw))
        points = transform_points(xyz_of(raw[start:stop]), transform)
        finite = np.isfinite(points).all(axis=1)
        values = np.full(stop-start, np.inf, dtype=np.float32)
        if finite.any():
            values[finite] = segmenter.scene.compute_distance(
                o3d.core.Tensor(np.ascontiguousarray(points[finite], dtype=np.float32)),
                nthreads=segmenter.nthreads).numpy()
        valid_distance = np.isfinite(values) & (values >= 0)
        accepted = finite & valid_distance & (values <= threshold_mm)
        exception = np.zeros(stop-start, dtype=bool)
        for face_id in unmeshed:
            bounds = segmenter.topology.face_bounds[face_id]
            box_distance = np.linalg.norm(
                np.maximum(np.maximum(bounds[:3]-points, points-bounds[3:]), 0), axis=1)
            exception |= finite & ~accepted & (box_distance <= margin)
        distances[start:stop] = values
        keep[start:stop] = accepted | exception
        protected[start:stop] = exception
        nonfinite += int((~finite).sum())
        query_invalid += int((finite & ~valid_distance).sum())
        print(f'CAD foreground distances: {stop}/{len(raw)}', flush=True)
    for array in (distances, keep, protected):
        array.flush()
    source_indices = np.flatnonzero(keep).astype(np.int64, copy=False)
    removed_indices = np.flatnonzero(~keep).astype(np.int64, copy=False)
    np.save(dest/'source_indices.npy', source_indices)
    np.save(dest/'removed_source_indices.npy', removed_indices)
    finite_distances = np.asarray(distances[np.isfinite(distances)], dtype=float)
    levels = [0, 25, 50, 75, 90, 95, 97, 99, 99.5, 100]
    percentiles = (dict(zip(map(str, levels), map(float, np.percentile(finite_distances, levels))))
                   if len(finite_distances) else {})
    thresholds = sorted(set([.5, 1., 1.5, 2., 3., 5., 10., float(threshold_mm)]))
    return {
        'source_points': len(raw), 'kept_points': len(source_indices),
        'removed_points': len(removed_indices),
        'distance_threshold_mm': float(threshold_mm),
        'protected_unmeshed_points': int(protected.sum()),
        'nonfinite_points': nonfinite, 'invalid_distance_queries': query_invalid,
        'distance_percentiles_mm': percentiles,
        'threshold_sensitivity': [
            {'distance_mm': t, 'within_distance_points': int(np.count_nonzero(distances <= t))}
            for t in thresholds],
        'method': 'Open3D RaycastingScene.compute_distance; unsigned point-to-triangle distance',
        'distance_frame': 'CAD_LOCAL', 'output_frame': 'ROBOT_BASE', 'units': 'mm',
        'raw_xyz_transformed': False, 'registration_refitted': False,
        'fixture_semantics_identified': False,
        'unmeshed_policy': 'retain nearby incomplete-face bounding boxes as uncertain candidates',
        'filter_seconds': time.perf_counter()-started,
    }


def write_point_subset_ply(path, raw, source_indices, face_labels=None, chunk_points=200000):
    """Write original BASE rows in original order, optionally with CAD labels.

    ``face_labels`` aligns with the retained rows, not the unfiltered source.
    All assigned CAD faces are colored; unknown labels are gray. No planned
    surface flag is added because foreground extraction does not plan paths.
    """
    import colorsys

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if isinstance(chunk_points, bool) or not isinstance(chunk_points, int) or chunk_points < 1:
        raise ValueError('chunk_points must be a positive integer')
    indices = np.asarray(source_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('source_indices must be a one-dimensional integer array')
    if len(indices) and (indices[0] < 0 or indices[-1] >= len(raw) or
                         np.any(indices[1:] <= indices[:-1])):
        raise ValueError('Source indices must be unique, increasing and within the input scan')
    if not raw.dtype.names or not set('xyz').issubset(raw.dtype.names):
        raise ValueError('Input must be a structured scan with XYZ scalar attributes')
    fields = [(name, raw.dtype.fields[name][0].newbyteorder('<')) for name in raw.dtype.names]
    labels = None if face_labels is None else np.asarray(face_labels)
    if labels is not None:
        if labels.shape != (len(indices),) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('face_labels must align with the filtered scan rows')
        if (labels < -1).any() or (labels > np.iinfo(np.int32).max).any():
            raise ValueError('Completed face labels must be -1 or a nonnegative int32 CAD face ID')
        if set(raw.dtype.names).intersection({'cad_face_id', 'red', 'green', 'blue'}):
            raise ValueError('Refuse to overwrite existing label/color attributes')
        fields.extend([('cad_face_id', '<i4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    dtype = np.dtype(fields)
    ply_types = {('f', 4): 'float', ('f', 8): 'double', ('i', 1): 'char', ('i', 2): 'short',
                 ('i', 4): 'int', ('u', 1): 'uchar', ('u', 2): 'ushort', ('u', 4): 'uint'}
    header = ['ply', 'format binary_little_endian 1.0',
              'comment frame ROBOT_BASE; units mm; original selected XYZ and attributes unchanged',
              'comment Row provenance: source_indices.npy (kept) or removed_source_indices.npy (removed)',
              f'element vertex {len(indices)}']
    for name in dtype.names:
        field = dtype.fields[name][0]
        if field.subdtype or (field.kind, field.itemsize) not in ply_types:
            raise ValueError(f'Unsupported PLY scalar type: {name} {field}')
        header.append(f'property {ply_types[(field.kind, field.itemsize)]} {name}')
    palette = {}
    if labels is not None:
        for fid in np.unique(labels[labels >= 0]):
            rgb = colorsys.hsv_to_rgb((int(fid)*.61803398875) % 1., .68, .88)
            palette[int(fid)] = np.round(255*np.asarray(rgb)).astype(np.uint8)
    palette_ids = np.asarray(sorted(palette), dtype=np.int64)
    palette_rgb = np.asarray([palette[int(fid)] for fid in palette_ids], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.write(('\n'.join(header+['end_header'])+'\n').encode('ascii'))
        for start in range(0, len(indices), chunk_points):
            stop = min(start+chunk_points, len(indices))
            block = np.empty(stop-start, dtype=dtype)
            selected = raw[indices[start:stop]]
            for name in raw.dtype.names:
                block[name] = selected[name]
            if labels is not None:
                local = labels[start:stop]
                block['cad_face_id'] = local
                rgb = np.full((len(local), 3), [150, 158, 167], dtype=np.uint8)
                assigned = local >= 0
                if assigned.any():
                    rgb[assigned] = palette_rgb[np.searchsorted(palette_ids, local[assigned])]
                for j, name in enumerate(['red', 'green', 'blue']):
                    block[name] = rgb[:, j]
            block.tofile(stream)
