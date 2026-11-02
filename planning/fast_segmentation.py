"""Batched, conservative face-preserving mesh labels for full-resolution scans.

This module explicitly reports mesh approximation. Exact STEP audits and exact
surface evaluation for paths remain separate; mesh labels are not a certificate
of metrological accuracy. All positions are mm and original row order is kept.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from planning.segmentation import segmentation_config, segment_points


class MeshFaceSegmenter:
    """One Open3D BVH reused across many point blocks.

    Physical trim edges are sampled lazily per observed face. The reported lower
    estimate subtracts half the quasi-uniform sample spacing and mesh allowance.
    Local perturbation queries detect unstable neighboring-face assignments;
    they do not constitute an exhaustive second-nearest-surface proof.
    """
    def __init__(self, topology, boundary_sampling_mm=.4, mesh_error_margin_mm=None,
                 boundary_cache_dir=None, nthreads=4):
        import open3d as o3d
        if not np.isfinite(boundary_sampling_mm) or boundary_sampling_mm <= 0:
            raise ValueError('boundary_sampling_mm must be finite and positive')
        self.topology = topology
        self.boundary_sampling_mm = float(boundary_sampling_mm)
        self.mesh_error_margin_mm = float(topology.catalog['mesh_linear_deflection_mm']
            if mesh_error_margin_mm is None else mesh_error_margin_mm)
        if not np.isfinite(self.mesh_error_margin_mm) or self.mesh_error_margin_mm < 0:
            raise ValueError('mesh_error_margin_mm must be finite and nonnegative')
        self.nthreads = nthreads
        self.boundary_cache_dir = None if boundary_cache_dir is None else Path(boundary_cache_dir)
        if self.boundary_cache_dir is not None:
            self.boundary_cache_dir.mkdir(parents=True, exist_ok=True)
        self.boundary_trees = {}
        self.boundary_spacing = {}
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=nthreads)
        if not len(topology.triangles):
            raise ValueError('Fast segmentation requires a nonempty face-preserving mesh')
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(topology.vertices, np.float32)),
                                 o3d.core.Tensor(np.asarray(topology.triangles, np.uint32)))
        self.unmeshed_face_ids = np.array([r['face_id'] for r in topology.catalog.get('mesh_warnings', [])], int)

    def _query(self, points):
        import open3d as o3d
        answer = self.scene.compute_closest_points(o3d.core.Tensor(np.asarray(points,np.float32)),
                                                   nthreads=self.nthreads)
        return {key: value.numpy() for key,value in answer.items()}

    def _boundary(self, face_id):
        if face_id in self.boundary_trees:
            return self.boundary_trees[face_id], self.boundary_spacing[face_id]
        cache = None if self.boundary_cache_dir is None else self.boundary_cache_dir/f'face_{face_id:05d}.npz'
        points, spacing = None, 0.
        if cache is not None and cache.exists():
            with np.load(cache,allow_pickle=False) as saved:
                if (float(saved['sampling_mm']) == self.boundary_sampling_mm and
                        str(saved['step_sha256']) == self.topology.catalog.get('step_sha256','')):
                    points, spacing = saved['points'].copy(), float(saved['max_arc_step_mm'])
        if points is None:
            samples = []
            try:
                for edge in self.topology.boundary_edges[face_id]:
                    length = float(edge.Length())
                    count = max(2, int(np.ceil(length/self.boundary_sampling_mm))+1)
                    # Avoid unbounded allocation from a malformed edge/units.
                    if count > 500000:
                        raise ValueError('Trim edge sampling exceeds 500000 points')
                    vectors, _ = edge.sample(count)
                    values = np.asarray([p.toTuple() for p in vectors],float)
                    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
                        raise ValueError('Invalid trim edge sampling')
                    samples.append(values)
                    spacing = max(spacing,length/max(count-1,1))
                points = np.vstack(samples) if samples else np.empty((0,3))
            except (RuntimeError, ValueError):
                points, spacing = np.empty((0,3)), np.nan
            if cache is not None:
                np.savez_compressed(cache, points=points, max_arc_step_mm=spacing,
                    sampling_mm=self.boundary_sampling_mm,
                    step_sha256=np.array(self.topology.catalog.get('step_sha256','')))
        tree = cKDTree(points) if len(points) else None
        self.boundary_trees[face_id], self.boundary_spacing[face_id] = tree, spacing
        return tree, spacing

    def segment(self, points_C, cfg=None, normals_C=None, source_indices=None):
        cfg = segmentation_config(cfg)
        points = np.asarray(points_C,float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points_C must be Nx3')
        count = len(points)
        ids = np.arange(count,dtype=np.int64) if source_indices is None else np.asarray(source_indices)
        if ids.shape != (count,) or not np.issubdtype(ids.dtype,np.integer):
            raise ValueError('source_indices must be one integer per point')
        finite = np.isfinite(points).all(axis=1)
        query = points.copy()
        query[~finite] = 0.
        hit = self._query(query)
        tri_ids = hit['primitive_ids'].astype(np.int64)
        safe = (tri_ids >= 0) & (tri_ids < len(self.topology.triangles))
        tri_ids[~safe] = 0
        faces = self.topology.triangle_face_ids[tri_ids]
        projected = np.asarray(hit['points'],float)
        normals = np.asarray(hit['primitive_normals'],float)
        norms = np.linalg.norm(normals,axis=1)
        normal_valid = np.isfinite(normals).all(axis=1) & (norms > 1e-12)
        normals /= np.maximum(norms[:,None],1e-12)
        distance = np.linalg.norm(query-projected,axis=1)
        bary = hit['primitive_uvs']
        tri = self.topology.triangles[tri_ids]
        node_uv = self.topology.vertex_uv[tri]
        uv = ((1-bary.sum(axis=1))[:,None]*node_uv[:,0]+bary[:,0,None]*node_uv[:,1]+bary[:,1,None]*node_uv[:,2])
        status = np.full(count,'mesh_assigned',dtype=object)
        status[~finite] = 'nonfinite_point'
        status[finite & (~safe | ~normal_valid)] = 'invalid_mesh_projection'
        valid = finite & safe & normal_valid & (distance <= cfg['max_distance_mm'])
        status[finite & safe & normal_valid & ~valid] = 'too_far'
        boundary = np.full(count,np.nan)
        for fid in np.unique(faces[valid]):
            selected = np.flatnonzero(valid & (faces == fid))
            tree, step = self._boundary(int(fid))
            if tree is None:
                boundary[selected] = np.nan if not np.isfinite(step) else np.inf
            else:
                boundary[selected] = np.maximum(0.,tree.query(projected[selected],workers=1)[0]
                    -.55*step-self.mesh_error_margin_mm)
        outside = valid & (~np.isfinite(boundary)) & ~np.isinf(boundary)
        status[outside] = 'boundary_unresolved'; valid[outside] = False
        edge = valid & (boundary < cfg['boundary_margin_mm'])
        status[edge] = 'mesh_boundary_uncertain'; valid[edge] = False
        # Exclude incomplete-mesh neighborhoods rather than assigning another face.
        incomplete = np.zeros(count,bool)
        for fid in self.unmeshed_face_ids:
            low,high = self.topology.face_bounds[fid,:3],self.topology.face_bounds[fid,3:]
            box_distance = np.linalg.norm(np.maximum(np.maximum(low-query,query-high),0),axis=1)
            incomplete |= box_distance <= cfg['max_distance_mm']+self.mesh_error_margin_mm
        status[valid & incomplete] = 'near_unmeshed_face'; valid &= ~incomplete
        # Query nearby offsets in the local normal and two tangent directions.
        # Instability rejects labels; no perturbation ever moves the saved point.
        active = np.flatnonzero(valid)
        competition = np.full(count,np.nan)
        unstable = np.zeros(count,bool)
        probe = cfg['ambiguity_margin_mm']+self.mesh_error_margin_mm
        if len(active) and probe > 0:
            n = normals[active]
            seeds = np.eye(3)[np.argmin(np.abs(n),axis=1)]
            tangent = np.cross(n,seeds); tangent/=np.linalg.norm(tangent,axis=1)[:,None]
            across = np.cross(n,tangent)
            for vector in (n,-n,tangent,-tangent,across,-across):
                other = self._query(query[active]+probe*vector)
                tids = other['primitive_ids'].astype(np.int64)
                good = (tids >= 0) & (tids < len(self.topology.triangles))
                tids[~good] = 0
                changed = good & (self.topology.triangle_face_ids[tids] != faces[active])
                unstable[active[changed]] = True
                margin = np.linalg.norm(query[active]-other['points'],axis=1)-distance[active]
                take = changed & (np.isnan(competition[active]) | (margin < competition[active]))
                competition[active[take]] = margin[take]
        status[valid & unstable] = 'mesh_face_competition'; valid &= ~unstable
        scan_angle = np.full(count,np.nan)
        if normals_C is not None:
            scan_normals = np.asarray(normals_C,float)
            if scan_normals.shape != points.shape:
                raise ValueError('normals_C must align with points_C')
            snorm = np.linalg.norm(scan_normals,axis=1)
            dot = np.sum(scan_normals*normals,axis=1)/np.maximum(snorm,1e-12)
            if not cfg['normal_orientation_known']:
                dot = np.abs(dot)
            scan_angle = np.degrees(np.arccos(np.clip(dot,-1,1)))
            wrong = ~np.isfinite(scan_normals).all(axis=1) | (snorm < 1e-12) | (scan_angle > cfg['max_normal_angle_deg'])
            status[valid & wrong] = 'scan_normal_disagreement'; valid &= ~wrong
        signed = np.sum((query-projected)*normals,axis=1)
        result = pd.DataFrame({'source_index':ids,'face_id':np.where(valid,faces,-1),
            'candidate_face_id':faces,'valid':valid,'status':status,
            'confidence':np.where(valid,np.maximum(0.,1-distance/cfg['max_distance_mm']),0.),
            'u':uv[:,0],'v':uv[:,1],'distance_mm':distance,'signed_deviation_mm':signed,
            'boundary_distance_mm':boundary,'competition_margin_mm':competition,
            'scan_normal_angle_deg':scan_angle,'exact_projection_count':0,
            'normal_side_confirmed':False,'geometry_method':'face_preserving_mesh',
            'uv_method':'triangle_interpolation_of_OCCT_face_UV',
            'boundary_method':'sampled_trim_lower_estimate',
            'competition_method':'six_local_perturbation_queries',
            'mesh_error_margin_mm':self.mesh_error_margin_mm})
        for j,axis in enumerate('xyz'):
            result[f'projected_{axis}_mm'] = projected[:,j]
            result[f'normal_{axis}'] = normals[:,j]
        return result

    def audit(self, points_C, cfg=None, source_indices=None):
        """Compare the full-resolution method against independent exact faces.

        Caller chooses spatially representative points. Include rejected points
        in that selection; do not report agreement only for accepted labels.
        """
        mesh = self.segment(points_C,cfg,source_indices=source_indices)
        exact = segment_points(points_C,self.topology,cfg,source_indices=source_indices)
        records = pd.DataFrame({'source_index':mesh.source_index,
            'mesh_face_id':mesh.face_id,'mesh_status':mesh.status,
            'exact_face_id':exact.face_id,'exact_status':exact.status,
            'mesh_assigned':mesh.valid,'exact_assigned':exact.valid,
            'label_equal':mesh.face_id == exact.face_id,
            'distance_difference_mm':mesh.distance_mm-exact.distance_mm})
        summary = {'sample_points':len(records),'mesh_assigned':int(mesh.valid.sum()),
            'exact_assigned':int(exact.valid.sum()),
            'both_assigned':int((mesh.valid & exact.valid).sum()),
            'both_assigned_same_face':int((mesh.valid & exact.valid & records.label_equal).sum()),
            'mesh_assigned_exact_rejected':int((mesh.valid & ~exact.valid).sum()),
            'exact_assigned_mesh_rejected':int((~mesh.valid & exact.valid).sum()),
            'assigned_face_conflicts':int((mesh.valid & exact.valid & ~records.label_equal).sum()),
            'physical_accuracy_verified':False}
        return records, summary
