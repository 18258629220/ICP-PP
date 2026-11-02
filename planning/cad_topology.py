"""Exact STEP face topology and face-preserving absolute OCCT tessellation.

Face indices follow ``cad_reference.load_faces``. Vertices deliberately remain
separate across faces, so their UV coordinates are parameters of exactly one
STEP surface. These are not triangle barycentric coordinates. All lengths: mm.
"""
from itertools import combinations
import numpy as np
from planning.cad_reference import load_faces


class CadTopology:
    """Keep exact faces, true shared-edge adjacency, and triangle provenance."""

    @classmethod
    def from_step(cls, path, **kwargs):
        return cls(load_faces(path), **kwargs)

    def __init__(self, faces, linear_deflection_mm=.05, angular_deflection_rad=.15):
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopAbs import TopAbs_REVERSED
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
            # relative=False matters: CadQuery's Shape.mesh defaults to relative.
            mesher = BRepMesh_IncrementalMesh(face.wrapped, float(linear_deflection_mm), False,
                                            float(angular_deflection_rad), False)
            if not mesher.IsDone():
                raise ValueError(f'OCCT triangulation failed for face {face_id}')
            location = TopLoc_Location()
            poly = BRep_Tool.Triangulation_s(face.wrapped, location)
            if poly is None or not poly.HasUVNodes() or not poly.NbTriangles():
                raise ValueError(f'No face/UV triangulation for face {face_id}')
            transform = location.Transformation()
            for i in range(1, poly.NbNodes()+1):
                p = poly.Node(i).Transformed(transform)
                uv = poly.UVNode(i)
                vertices.append([p.X(), p.Y(), p.Z()])
                uv_nodes.append([uv.X(), uv.Y()])
                vertex_faces.append(face_id)
            reverse = face.wrapped.Orientation() == TopAbs_REVERSED
            for i in range(1, poly.NbTriangles()+1):
                tri = list(poly.Triangle(i).Get())
                if reverse:
                    tri[1], tri[2] = tri[2], tri[1]
                triangles.append([v-1+offset for v in tri])
                triangle_faces.append(face_id)
            record['triangle_count'] = poly.NbTriangles()
            records.append(record)
            offset += poly.NbNodes()
        adjacency = []
        for record in edge_records:
            for a, b in combinations(record['face_ids'], 2):
                records[a]['adjacent_face_ids'].append(b)
                records[b]['adjacent_face_ids'].append(a)
                adjacency.append({'face_a': a, 'face_b': b, 'edge_id': record['edge_id']})
        for record in records:
            record['adjacent_face_ids'] = sorted(set(record['adjacent_face_ids']))
        self.face_bounds = np.asarray(self.face_bounds, dtype=float)
        self.vertices = np.asarray(vertices, dtype=float)
        self.vertex_uv = np.asarray(uv_nodes, dtype=float)
        self.vertex_face_ids = np.asarray(vertex_faces, dtype=np.int64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.triangle_face_ids = np.asarray(triangle_faces, dtype=np.int64)
        self.catalog = {'face_count': len(self.faces), 'faces': records, 'edges': edge_records,
                        'adjacency': adjacency, 'units': 'mm',
                        'face_index_convention': 'zero-based imported STEP root/face traversal; NOT triangle IDs',
                        'uv_convention': 'OCCT face surface parameters; NOT barycentric coordinates',
                        'mesh_linear_deflection_mm': float(linear_deflection_mm),
                        'mesh_angular_deflection_rad': float(angular_deflection_rad),
                        'relative_deflection': False, 'material_side_confirmed': False,
                        'adjacency_convention': 'shared topological edges; geometric coincidence alone is not adjacency'}

    def save_mesh(self, path):
        """Save a local NPZ with complete face/UV provenance, never mesh cleanup."""
        np.savez_compressed(path, vertices=self.vertices, triangles=self.triangles,
                            triangle_face_ids=self.triangle_face_ids, vertex_uv=self.vertex_uv,
                            vertex_face_ids=self.vertex_face_ids)
