"""Exact STEP face topology and face-preserving absolute OCCT tessellation.

Face indices follow ``cad_reference.load_faces``. Vertices deliberately remain
separate across faces, so their UV coordinates are parameters of exactly one
STEP surface. These are not triangle barycentric coordinates. All lengths: mm.
"""
from itertools import combinations
import warnings
import numpy as np
from planning.cad_reference import load_faces


class CadTopology:
    """Keep exact faces, true shared-edge adjacency, and triangle provenance."""

    @classmethod
    def from_step(cls, path, **kwargs):
        return cls(load_faces(path), **kwargs)

    def __init__(self, faces, linear_deflection_mm=.05, angular_deflection_rad=.15):
        from OCP.BRep import BRep_Tool
        if (not np.isfinite(linear_deflection_mm) or linear_deflection_mm <= 0 or
                not np.isfinite(angular_deflection_rad) or angular_deflection_rad <= 0):
            raise ValueError('Mesh deflections must be finite and positive')
        self.faces = list(faces)
        if not self.faces:
            raise ValueError('CAD requires at least one face')
        self.boundary_edges = []
        self.face_bounds = []
        self.surfaces = []
        records, edge_records, edge_objects, buckets = [], [], [], {}
        vertices, uv_nodes, vertex_faces, triangles, triangle_faces = [], [], [], [], []
        offset = 0
        mesh_warnings = []
        for face_id, face in enumerate(self.faces):
            bounds = face.BoundingBox()
            bbox = [bounds.xmin, bounds.ymin, bounds.zmin, bounds.xmax, bounds.ymax, bounds.zmax]
            if not np.isfinite(bbox).all():
                raise ValueError(f'Nonfinite CAD bounds on face {face_id}')
            self.face_bounds.append(bbox)
            self.surfaces.append(BRep_Tool.Surface_s(face.wrapped))
            face_edge_ids, non_seam = [], []
            for edge in face.Edges():
                key = edge.hashCode()
                edge_id = next((i for i in buckets.get(key, []) if edge.isSame(edge_objects[i])), None)
                if edge_id is None:
                    edge_id = len(edge_objects)
                    edge_objects.append(edge)
                    buckets.setdefault(key, []).append(edge_id)
                    edge_records.append({'edge_id': edge_id, 'face_ids': [],
                                         'length_mm': float(edge.Length()), 'geometry_type': edge.geomType()})
                if face_id not in edge_records[edge_id]['face_ids']:
                    edge_records[edge_id]['face_ids'].append(face_id)
                face_edge_ids.append(edge_id)
                # Periodic parameter seams are not physical trim boundaries.
                if not BRep_Tool.IsClosed_s(edge.wrapped, face.wrapped):
                    non_seam.append(edge)
            self.boundary_edges.append(non_seam)
            record = {'face_id': face_id, 'geometry_type': face.geomType(),
                      'area_mm2': float(face.Area()), 'uv_bounds': list(map(float, face.uvBounds())),
                      'bounds_mm': bbox, 'edge_ids': face_edge_ids,
                      'inner_wire_count': len(face.innerWires()),
                      'orientation': str(face.wrapped.Orientation()), 'adjacent_face_ids': []}
            # Exact faces remain available even if one face cannot be meshed.
            # Per-face arrays are appended atomically, so failed tessellation
            # never shifts subsequent face IDs or leaves orphan mesh vertices.
            mesh = _triangulate_face(face, linear_deflection_mm, angular_deflection_rad)
            record['mesh_status'] = mesh['status']
            record['triangle_count'] = 0
            if mesh['status'] == 'meshed':
                local_vertices, local_uv, local_triangles = mesh['vertices'], mesh['uv'], mesh['triangles']
                vertices.extend(local_vertices)
                uv_nodes.extend(local_uv)
                vertex_faces.extend([face_id]*len(local_vertices))
                triangles.extend(local_triangles+offset)
                triangle_faces.extend([face_id]*len(local_triangles))
                record['triangle_count'] = len(local_triangles)
                offset += len(local_vertices)
            else:
                record['cad_valid'] = bool(face.isValid())
                mesh_warnings.append({'face_id': face_id, 'mesh_status': mesh['status'],
                    'geometry_type': record['geometry_type'], 'area_mm2': record['area_mm2'],
                    'cad_valid': record['cad_valid'],
                    **({'error_type': mesh['error_type']} if 'error_type' in mesh else {})})
            records.append(record)
        adjacency = []
        for record in edge_records:
            for a, b in combinations(record['face_ids'], 2):
                records[a]['adjacent_face_ids'].append(b)
                records[b]['adjacent_face_ids'].append(a)
                adjacency.append({'face_a': a, 'face_b': b, 'edge_id': record['edge_id']})
        for record in records:
            record['adjacent_face_ids'] = sorted(set(record['adjacent_face_ids']))
        self.face_bounds = np.asarray(self.face_bounds, dtype=float)
        self.vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
        self.vertex_uv = np.asarray(uv_nodes, dtype=float).reshape(-1, 2)
        self.vertex_face_ids = np.asarray(vertex_faces, dtype=np.int64)
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.triangle_face_ids = np.asarray(triangle_faces, dtype=np.int64)
        mesh_status = 'complete' if not mesh_warnings else ('partial' if len(triangles) else 'empty')
        self.catalog = {'mesh_status': mesh_status, 'mesh_warnings': mesh_warnings,
                        'meshed_face_count': len(self.faces)-len(mesh_warnings),
                        'unmeshed_face_count': len(mesh_warnings),
                        'exact_faces_preserved': True, 'mesh_triangle_count': len(triangles),
                        'face_count': len(self.faces), 'faces': records, 'edges': edge_records,
                        'adjacency': adjacency, 'units': 'mm',
                        'face_index_convention': 'zero-based imported STEP root/face traversal; NOT triangle IDs',
                        'uv_convention': 'OCCT face surface parameters; NOT barycentric coordinates',
                        'mesh_linear_deflection_mm': float(linear_deflection_mm),
                        'mesh_angular_deflection_rad': float(angular_deflection_rad),
                        'relative_deflection': False, 'material_side_confirmed': False,
                        'adjacency_convention': 'shared topological edges; geometric coincidence alone is not adjacency'}

        if mesh_warnings:
            warnings.warn(f'CAD tessellation is {mesh_status}: {len(mesh_warnings)} of {len(self.faces)} '
                          'faces have no usable mesh; exact faces are retained. See mesh_warnings.',
                          RuntimeWarning, stacklevel=2)

    def save_mesh(self, path):
        """Save a local NPZ with complete face/UV provenance, never mesh cleanup."""
        np.savez_compressed(path, vertices=self.vertices, triangles=self.triangles,
                            triangle_face_ids=self.triangle_face_ids, vertex_uv=self.vertex_uv,
                            vertex_face_ids=self.vertex_face_ids, mesh_status=np.array(self.catalog['mesh_status']),
                            unmeshed_face_ids=np.asarray([w['face_id'] for w in self.catalog['mesh_warnings']], dtype=np.int64),
                            exact_face_count=np.array(len(self.faces), dtype=np.int64))


def _triangulate_face(face, linear_deflection_mm, angular_deflection_rad):
    """Return one complete face mesh or an explicit, recoverable failure status."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopAbs import TopAbs_REVERSED
    try:
        # relative=False matters: CadQuery's Shape.mesh defaults to relative.
        mesher = BRepMesh_IncrementalMesh(face.wrapped, float(linear_deflection_mm), False,
                                        float(angular_deflection_rad), False)
        if not mesher.IsDone():
            return {'status': 'mesher_not_done'}
        location = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face.wrapped, location)
        if poly is None:
            return {'status': 'missing_triangulation'}
        if not poly.HasUVNodes():
            return {'status': 'missing_uv_nodes'}
        if not poly.NbTriangles() or not poly.NbNodes():
            return {'status': 'empty_triangulation'}
        transform = location.Transformation()
        vertices, uv_nodes, triangles = [], [], []
        for i in range(1, poly.NbNodes()+1):
            p = poly.Node(i).Transformed(transform)
            uv = poly.UVNode(i)
            vertices.append([p.X(), p.Y(), p.Z()])
            uv_nodes.append([uv.X(), uv.Y()])
        reverse = face.wrapped.Orientation() == TopAbs_REVERSED
        for i in range(1, poly.NbTriangles()+1):
            tri = list(poly.Triangle(i).Get())
            if reverse:
                tri[1], tri[2] = tri[2], tri[1]
            triangles.append([v-1 for v in tri])
        vertices, uv_nodes = np.asarray(vertices, float), np.asarray(uv_nodes, float)
        triangles = np.asarray(triangles, np.int64)
        if (not np.isfinite(vertices).all() or not np.isfinite(uv_nodes).all()
                or triangles.min() < 0 or triangles.max() >= len(vertices)):
            return {'status': 'invalid_triangulation'}
        return {'status': 'meshed', 'vertices': vertices, 'uv': uv_nodes, 'triangles': triangles}
    except (RuntimeError, ValueError) as error:
        return {'status': 'meshing_exception', 'error_type': type(error).__name__}
