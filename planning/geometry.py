"""Geometry helpers. All positions are mm. No implicit mesh repair or hole filling."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pipeline_common import read_ply


def unit(v):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError('Direction must be finite and nonzero')
    return v / length


def tool_frame(outward, tangent=(1., 0., 0.)):
    """Tool +Z approaches the surface; columns express tool axes in CAD frame."""
    z = -unit(outward)
    tangent = np.asarray(tangent, dtype=float)
    x = tangent - z * np.dot(tangent, z)
    if np.linalg.norm(x) < 1e-8:
        basis = np.eye(3)[np.argmin(np.abs(z))]
        x = basis - z * np.dot(basis, z)
    x = unit(x)
    return np.column_stack((x, np.cross(z, x), z))


def ocl_frame(outward):
    """Proper rotation: OCL +Z (away from surface) expressed in CAD frame."""
    z = unit(outward)
    x = unit(np.eye(3)[np.argmin(np.abs(z))] - z * z[np.argmin(np.abs(z))])
    return np.column_stack((x, np.cross(z, x), z))


def quaternion_xyzw(matrix):
    return Rotation.from_matrix(matrix).as_quat()


class TriangleFile:
    """Memory-map the project's packed binary triangle PLY without loading all faces."""
    def __init__(self, path):
        self.vertices, meta = read_ply(path)
        header = meta['header']
        if meta['elements'][0][0] != 'vertex' or len(meta['elements']) != 2 or meta['elements'][1][0] != 'face':
            raise ValueError('Expected vertex then face elements only')
        face_start = next(i for i, s in enumerate(header) if s.startswith('element face '))
        face_properties = [s for s in header[face_start+1:] if s.startswith('property ')]
        if face_properties not in [['property list uchar uint vertex_indices'],
                                   ['property list uchar int vertex_indices']]:
            raise ValueError('Unsupported packed triangle layout')
        endian = '<' if 'format binary_little_endian 1.0' in header else '>'
        idx_type = 'u4' if ' uchar uint ' in face_properties[0] else 'i4'
        dtype = np.dtype([('count', 'u1'), ('indices', endian+idx_type, (3,))])
        offset = meta['offset'] + meta['count'] * self.vertices.dtype.itemsize
        count = meta['elements'][1][1]
        if Path(path).stat().st_size != offset + count * dtype.itemsize:
            raise ValueError('Non-triangular, truncated, or trailing face payload')
        self.faces = np.memmap(path, mode='r', dtype=dtype, offset=offset, shape=(count,))
        for start in range(0, count, 100000):
            block = self.faces[start:start+100000]
            if not np.all(block['count'] == 3):
                raise ValueError('Non-triangular face')
            if np.any(block['indices'] < 0) or np.any(block['indices'] >= len(self.vertices)):
                raise ValueError('Invalid vertex index')

    def triangles(self, ids):
        v = self.vertices[self.faces['indices'][ids]]
        return np.stack([v[n] for n in ('x', 'y', 'z')], axis=-1)

    def bounds_in_frame(self, frame, cache_path):
        """Rebuilt per run; float64 projected bounds prevent crop roundoff omissions."""
        aabb = np.lib.format.open_memmap(cache_path, mode='w+', dtype='float64', shape=(len(self.faces), 4))
        zmin, zmax = np.inf, -np.inf
        for start in range(0, len(self.faces), 30000):
            tri = self.triangles(slice(start, start+30000)) @ frame
            aabb[start:start+len(tri), :2] = tri[:, :, :2].min(axis=1)
            aabb[start:start+len(tri), 2:] = tri[:, :, :2].max(axis=1)
            zmin = min(zmin, float(tri[:, :, 2].min()))
            zmax = max(zmax, float(tri[:, :, 2].max()))
        aabb.flush()
        return aabb, zmin, zmax

    def crop(self, aabb, bounds, frame):
        xmin, ymin, xmax, ymax = bounds
        ids = np.flatnonzero((aabb[:, 0] <= xmax) & (aabb[:, 2] >= xmin) &
                             (aabb[:, 1] <= ymax) & (aabb[:, 3] >= ymin))
        return self.triangles(ids) @ frame


def ball_drop(triangles, xy, diameter, length, floor_z):
    """Call OpenCAMLib; return axial bottom points and actual cutter contacts."""
    from opencamlib import ocl
    if diameter <= 0 or length <= diameter or not len(triangles):
        raise ValueError('Invalid tool or empty CAD subset')
    surface = ocl.STLSurf()
    for triangle in triangles:
        surface.addTriangle(ocl.Triangle(*(ocl.Point(*map(float, p)) for p in triangle)))
    batch = ocl.BatchDropCutter()
    batch.setSTL(surface)
    # Boost.Python keeps a raw pointer here: keep the cutter alive until run returns.
    cutter = ocl.BallCutter(diameter, length)
    batch.setCutter(cutter)
    batch.setThreads(2)
    for x, y in xy:
        batch.appendPoint(ocl.CLPoint(float(x), float(y), float(floor_z)))
    batch.run()
    cl = batch.getCLPoints()
    tip = np.array([[p.x, p.y, p.z] for p in cl])
    contact = np.array([[p.getCC().x, p.getCC().y, p.getCC().z] for p in cl])
    touched = tip[:, 2] > floor_z + 1e-6
    return tip, contact, touched


def raster_lines(polygon, spacing, sample_step):
    """Clip each pass independently, preserving holes and disconnected intervals."""
    from shapely.geometry import LineString
    if spacing <= 0 or sample_step <= 0:
        raise ValueError('Path spacing must be positive')
    xmin, ymin, xmax, ymax = polygon.bounds
    lines = []
    for row, y in enumerate(np.arange(ymin + spacing/2, ymax, spacing)):
        cut = polygon.intersection(LineString([(xmin-1, y), (xmax+1, y)]))
        parts = list(cut.geoms) if hasattr(cut, 'geoms') else [cut]
        parts = sorted([p for p in parts if p.geom_type == 'LineString' and p.length > 0],
                       key=lambda p: p.bounds[0], reverse=bool(row % 2))
        for part in parts:
            t = np.linspace(0, part.length, max(2, int(np.ceil(part.length/sample_step))+1))
            points = np.array([[part.interpolate(s).x, y] for s in t])
            lines.append(points[::-1] if row % 2 else points)
    return lines


def split_valid(points, valid, max_jump):
    """Never connect across an invalid sample or a discontinuous height jump."""
    start = None
    result = []
    for i in range(len(points)):
        if not valid[i]:
            if start is not None:
                result.append(np.arange(start, i))
            start = None
            continue
        if start is None:
            start = i
        elif np.linalg.norm(points[i]-points[i-1]) > max_jump:
            result.append(np.arange(start, i))
            start = i
    if start is not None:
        result.append(np.arange(start, len(points)))
    return [ids for ids in result if len(ids) >= 2]


def aligned_raster_lines(polygon, sample_xy, spacing, sample_step):
    """Orient raster passes along the patch's principal direction to avoid short strokes."""
    from shapely.affinity import affine_transform
    centered = np.asarray(sample_xy)-np.mean(sample_xy, axis=0)
    values, vectors = np.linalg.eigh(centered.T@centered)
    major = vectors[:, -1] if values[-1] > 1e-12 else np.array([1., 0.])
    if major[np.argmax(np.abs(major))] < 0:
        major = -major
    rotation = np.column_stack([major, [-major[1], major[0]]])
    local = affine_transform(polygon, [rotation[0, 0], rotation[1, 0], rotation[0, 1], rotation[1, 1], 0, 0])
    lines = [points@rotation.T for points in raster_lines(local, spacing, sample_step)]
    return lines, float(np.degrees(np.arctan2(major[1], major[0])))
