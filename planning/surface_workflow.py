"""Exact-face coverage and conditional process regions; all lengths are mm.

Coverage uses sampled surface metric bounds, not uniform UV steps. It is a
geometric preview: contact footprint, material side and uncertainty assumptions
need external validation. Rejected samples always remain in diagnostics.
"""
import numpy as np
import pandas as pd
from planning.cad_reference import paths_on_face


def validate_surface_config(cfg):
    coverage, process = cfg['coverage'], cfg['process']
    chunk = cfg.get('segmentation_chunk_points', 20000)
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError('segmentation_chunk_points must be a positive integer')
    if not isinstance(process['material_side_confirmed'], bool):
        raise ValueError('material_side_confirmed must be a boolean')
    for key in ('line_spacing_mm', 'path_sampling_mm', 'max_metric_probe_step'):
        if not np.isfinite(coverage[key]) or coverage[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('max_lines_per_face', 'max_path_samples', 'minimum_face_support'):
        if not isinstance(coverage[key], int) or coverage[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if coverage['raster_axis'] not in (0, 1, 'auto'):
        raise ValueError('raster_axis must be 0, 1 or auto')
    for key in ('auto_min_face_area_mm2','support_padding_mm','minimum_stroke_length_mm'):
        value = coverage.get(key,0.)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if coverage.get('auto_min_support',1) < 1 or coverage.get('metric_probe_count',65) < 17:
        raise ValueError('Invalid coverage support/probe limits')
    if coverage.get('max_support_distance_mm',2.5) <= 0:
        raise ValueError('max_support_distance_mm must be positive')
    for key in ('boundary_margin_mm', 'effective_contact_radius_mm',
                'allowance_threshold_mm', 'deficit_threshold_mm', 'following_error_bound_mm'):
        if not np.isfinite(process[key]) or process[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if process['uncertainty_multiplier'] <= 0 or not np.isfinite(process['uncertainty_multiplier']):
        raise ValueError('uncertainty_multiplier must be positive')
    boundary_bound = process.get('boundary_position_bound_mm')
    if boundary_bound is not None and (not np.isfinite(boundary_bound) or boundary_bound < 0):
        raise ValueError('boundary_position_bound_mm must be null or finite and nonnegative')
    depths = np.asarray(process['layer_depths_mm'], float)
    if (depths.ndim != 1 or not len(depths) or not np.isfinite(depths).all()
            or depths[0] != 0 or np.any(np.diff(depths) <= 0)):
        raise ValueError('layer_depths_mm must start at zero and strictly increase')
    if np.any(depths > 0) and not process['material_side_confirmed']:
        raise ValueError('Positive layer depths require material_side_confirmed')
    for box in process.get('protected_uv_boxes', []):
        bounds = np.asarray(box['bounds'], float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError('Protected UV bounds must be [u_min,u_max,v_min,v_max]')


def _metric_rows(face, axis, spacing, max_probe_step, maximum, uv_bounds=None,
                 metric_probe_count=65):
    """Advance the cross-raster coordinate by sampled physical arc increments.

This measures the largest displacement over a probe grid along the raster.
Curvature between probes is not a certified bound; record the probe resolution.
"""
    bounds = np.asarray(face.uvBounds() if uv_bounds is None else uv_bounds, float).reshape(2, 2)
    if not np.isfinite(bounds).all() or np.any(bounds[:,1] <= bounds[:,0]):
        return np.array([]), 0.
    along, cross = bounds[axis], bounds[1-axis]
    cap = max(17, int(metric_probe_count))
    probes = np.linspace(*along, min(cap, max(17, int(np.ceil(np.diff(along)[0]/max_probe_step))+1)))
    surface = face._geomAdaptor()
    def positions(value):
        uv = np.column_stack([probes, np.full(len(probes), value)])
        if axis == 1:
            uv = uv[:, ::-1]
        return np.array([[p.X(),p.Y(),p.Z()] for p in
                         (surface.Value(*map(float, pair)) for pair in uv)])
    count = min(max(cap,33), max(33, int(np.ceil(np.diff(cross)[0]/max_probe_step))+1))
    fine = np.linspace(*cross, count)
    previous = positions(fine[0])
    arc = [0.]
    for value in fine[1:]:
        current = positions(value)
        arc.append(arc[-1]+float(np.linalg.norm(current-previous, axis=1).max()))
        previous = current
    if arc[-1] < 1e-9:
        return np.array([]), 0.
    n = max(2, int(np.ceil(arc[-1]/spacing))+1)
    if n > maximum:
        raise ValueError('Coverage exceeds max_lines_per_face; increase spacing or select fewer faces')
    return np.interp(np.linspace(0., arc[-1], n), arc, fine), float(arc[-1]/(n-1))


def select_processing_faces(topology, segmentation, cfg):
    """Select observed principal faces and explicitly list every excluded face.

    This is geometry-based ROI selection, not a production approval. Explicit
    face_ids override area/type screening, but missing scan support is retained
    as a diagnostic rather than fabricated surface coverage.
    """
    coverage = cfg['coverage']
    observed = segmentation.loc[segmentation.valid.astype(bool)]
    counts = (observed.groupby('face_id').point_count.sum() if 'point_count' in observed
              else observed.groupby('face_id').size())
    explicit = coverage.get('face_ids')
    if explicit is not None:
        if len(set(explicit)) != len(explicit):
            raise ValueError('Duplicate coverage face_ids')
        if any(not isinstance(i,int) or i < 0 or i >= len(topology.faces) for i in explicit):
            raise ValueError('Invalid coverage face_ids')
    minimum_area = float(coverage.get('auto_min_face_area_mm2', 0.))
    minimum_count = int(coverage.get('auto_min_support', coverage['minimum_face_support']))
    allowed_types = coverage.get('auto_geometry_types')
    records, selected = [], []
    metadata = {int(record['face_id']):record for record in getattr(topology,'catalog',{}).get('faces',[])}
    for fid, face in enumerate(topology.faces):
        cached = metadata.get(fid,{})
        kind = cached.get('geometry_type') or face.geomType()
        area = float(cached['area_mm2']) if 'area_mm2' in cached else float(face.Area())
        count = int(counts.get(fid,0))
        if explicit is not None:
            reason = 'selected_explicit' if fid in explicit else 'not_in_explicit_roi'
        elif count < minimum_count:
            reason = 'insufficient_observation'
        elif area < minimum_area:
            reason = 'small_feature_face'
        elif allowed_types is not None and kind not in allowed_types:
            reason = 'excluded_geometry_type'
        else:
            reason = 'selected_observed_principal_face'
        choose = reason.startswith('selected_')
        if choose:
            selected.append(fid)
        records.append({'face_id':fid, 'geometry_type':kind, 'area_mm2':area,
                        'support_points':count, 'selected':choose,
                        'selection_reason':reason,
                        'production_region_selected_by_user':False})
    return selected, records


def _supported_uv_bounds(face, points, uv, padding_mm):
    """Observed UV rectangle plus physical padding, clipped to the CAD chart.

    The rectangle limits work to observed chart ranges; exact face trim and
    same-face support-distance splitting still handle holes and missing strips.
    Periodic seam-spanning observations conservatively retain the full chart.
    """
    bounds = np.asarray(face.uvBounds(),float).reshape(2,2)
    if uv is None:
        uv = np.asarray([face.paramAt(tuple(p)) for p in points],float)
    else:
        uv = np.asarray(uv,float)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if not len(uv):
        return bounds
    low, high = uv.min(axis=0), uv.max(axis=0)
    from OCP.gp import gp_Pnt, gp_Vec
    point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    surface = face._geomAdaptor()
    try:
        surface.D1(*map(float,(low+high)*.5),point,du,dv)
        speed = np.asarray([du.Magnitude(),dv.Magnitude()])
        pad = padding_mm/np.maximum(speed,1e-9)
    except (RuntimeError,ValueError):
        pad = .02*(bounds[:,1]-bounds[:,0])
    low, high = np.maximum(bounds[:,0],low-pad), np.minimum(bounds[:,1],high+pad)
    for axis in (0,1):
        periodic = surface.IsUPeriodic() if axis == 0 else surface.IsVPeriodic()
        if periodic and high[axis]-low[axis] > .8*(bounds[axis,1]-bounds[axis,0]):
            low[axis],high[axis] = bounds[axis]
    return np.column_stack([low,high])


def _split_observed_path(path, tree, distance, minimum_length):
    """Keep continuous, observed CAD intervals; never connect across a gap."""
    good = tree.query(path['points_C'])[0] <= distance
    starts = np.flatnonzero(good & ~np.r_[False,good[:-1]])
    ends = np.flatnonzero(good & ~np.r_[good[1:],False])+1
    result = []
    for start,end in zip(starts,ends):
        if end-start < 2:
            continue
        length = np.linalg.norm(np.diff(path['points_C'][start:end],axis=0),axis=1).sum()
        if length < minimum_length:
            continue
        result.append({key:(value[start:end] if isinstance(value,np.ndarray) else value)
                       for key,value in path.items()})
    return result, int((~good).sum())


def build_surface_rasters(topology, segmentation, cfg, outward, support_points_C=None,
                          support_face_ids=None, support_uv=None):
    """Cover selected observed STEP faces, splitting every exact trim hole.

All faces stay in the catalog. Auto mode selects every sufficiently observed
face, without asserting a production ROI. Each raster is an independent stroke.
"""
    coverage = cfg['coverage']
    from scipy.spatial import cKDTree
    selected, selection_records = select_processing_faces(topology,segmentation,cfg)
    record_by_face = {record['face_id']:record for record in selection_records}
    support = None if support_points_C is None else np.asarray(support_points_C,float)
    labels = None if support_face_ids is None else np.asarray(support_face_ids)
    uv_support = None if support_uv is None else np.asarray(support_uv,float)
    if support is not None:
        if (support.ndim != 2 or support.shape[1] != 3 or labels is None
                or labels.shape != (len(support),) or not np.isfinite(support).all()):
            raise ValueError('Coverage support points and face IDs must be finite aligned arrays')
        if uv_support is not None and uv_support.shape != (len(support),2):
            raise ValueError('Coverage support UV must align with support points')
    paths, records, total = [], [], 0
    for index in selected:
        if not isinstance(index, int) or index < 0 or index >= len(topology.faces):
            raise ValueError(f'Invalid coverage face_id {index}')
        face = topology.faces[index]
        axis = coverage['raster_axis']
        bounds = np.asarray(face.uvBounds()).reshape(2, 2)
        tree = None
        record = record_by_face[index]
        if support is not None:
            ids = np.flatnonzero(labels == index)
            if not len(ids):
                record.update(path_status='no_labeled_scan_support',raster_lines=0,
                              reference_samples=0,generated_strokes=0)
                records.append(record)
                continue
            local_support = support[ids]
            tree = cKDTree(local_support)
            if coverage.get('restrict_to_observed_uv',True):
                # UV projection cost is paid once per support point, not per
                # raster line. Bound the support probe pool deterministically.
                if len(ids) > 12000:
                    ids = ids[np.linspace(0,len(ids)-1,12000).astype(int)]
                bounds = _supported_uv_bounds(face,support[ids],
                    None if uv_support is None else uv_support[ids],
                    float(coverage.get('support_padding_mm',coverage['line_spacing_mm'])))
        if axis == 'auto':
            lengths = []
            surface = face._geomAdaptor()
            for candidate in (0,1):
                values = np.linspace(*bounds[candidate],33)
                uv = np.column_stack([values,np.full(len(values),bounds[1-candidate].mean())])
                if candidate == 1:
                    uv = uv[:,::-1]
                pts = np.array([[p.X(),p.Y(),p.Z()] for p in
                    (surface.Value(*map(float,pair)) for pair in uv)])
                lengths.append(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
            axis = int(np.argmax(lengths))
        rows, measured = _metric_rows(face, axis, coverage['line_spacing_mm'],
                                      coverage['max_metric_probe_step'], coverage['max_lines_per_face'],
                                      uv_bounds=bounds,
                                      metric_probe_count=coverage.get('metric_probe_count',65))
        record.update(raster_lines=len(rows),sampled_cross_line_spacing_mm=measured,
                      raster_axis=axis,
                      line_spacing_certified=False,planning_uv_bounds=bounds.ravel().tolist(),
                      generated_strokes=0,reference_samples=0,unsupported_reference_samples=0,
                      path_status='generated')
        records.append(record)
        print(f'Coverage face {index}: {len(rows)} raster lines, {record["area_mm2"]:.1f} mm2',flush=True)
        for line, value in enumerate(rows):
            uv = np.column_stack([bounds[axis], np.full(2, value)])
            if axis == 1:
                uv = uv[:, ::-1]
            # Preserve tool roll direction on alternate rows; no implicit zigzag
            # links through holes, gaps, or between different faces.
            observed_parts = []
            for path in paths_on_face(face,uv,coverage['path_sampling_mm'],outward):
                if tree is None:
                    observed_parts.append(path)
                else:
                    portions,rejected = _split_observed_path(path,tree,
                        float(coverage.get('max_support_distance_mm',2.5)),
                        float(coverage.get('minimum_stroke_length_mm',0.)))
                    observed_parts.extend(portions)
                    record['unsupported_reference_samples'] += rejected
            for part, path in enumerate(observed_parts):
                total += len(path['points_C'])
                if total > coverage['max_path_samples']:
                    raise ValueError('Coverage exceeds max_path_samples; refine ROI or spacing')
                path.update(path_id=f'face_{index:04d}_line_{line:04d}_part_{part:02d}',
                            face_ids=np.full(len(path['points_C']), index, int),
                            raster_id=len(paths),pass_id=0,
                            selection={'mode': 'surface_coverage', 'face_id': index,
                                       'geometry_type': face.geomType(), 'line': line,
                                       'production_region_selected_by_user': False,
                                       'scan_support_filtered':tree is not None})
                paths.append(path)
                record['generated_strokes'] += 1
                record['reference_samples'] += len(path['points_C'])
    records.extend(record for record in selection_records if not record['selected'])
    return paths, records


def boundary_distances(topology, table):
    """Exact 3D distance from reference contact to all trim wires of its face."""
    import cadquery as cq
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    result = np.full(len(table), np.nan)
    edges = {int(fid): topology.boundary_edges[int(fid)] for fid in table.face_id.unique()}
    boxes = {}
    for fid,values in edges.items():
        bounds = [edge.BoundingBox() for edge in values]
        boxes[fid] = np.array([[b.xmin,b.ymin,b.zmin,b.xmax,b.ymax,b.zmax]
                              for b in bounds],float).reshape(-1,6)
    for i, row in enumerate(table.itertuples()):
        point = cq.Vertex.makeVertex(row.cad_x_mm, row.cad_y_mm, row.cad_z_mm)
        xyz = np.array([row.cad_x_mm,row.cad_y_mm,row.cad_z_mm])
        bounds = boxes[int(row.face_id)]
        lower = np.linalg.norm(np.maximum(np.maximum(bounds[:,:3]-xyz,xyz-bounds[:,3:]),0.),axis=1)
        best = np.inf
        for edge_id in np.argsort(lower):
            if lower[edge_id] > best+1e-9:
                break
            edge = edges[int(row.face_id)][edge_id]
            query = BRepExtrema_DistShapeShape(point.wrapped, edge.wrapped)
            if not query.IsDone():
                raise ValueError('Trim boundary distance solver failed')
            best = min(best,query.Value())
        result[i] = best
    return result


def classify_surface_samples(table, boundary_mm, cfg):
    """Keep geometry validity distinct from conditional process eligibility."""
    process = cfg['process']
    result = table.copy()
    result['boundary_distance_mm'] = np.asarray(boundary_mm, float)
    # A zero fit residual never substitutes for unknown scanner/registration error.
    sigma = result.get('conditional_position_sigma_mm', pd.Series(np.nan, index=result.index))
    complete = result.get('uncertainty_complete', pd.Series(False, index=result.index)).astype(bool)
    # Mapping exports also expose an explicit complete flag under older names.
    complete &= np.isfinite(sigma)
    bound = process['uncertainty_multiplier']*sigma
    result['offset_lower_mm'] = result.offset_mm-bound
    result['offset_upper_mm'] = result.offset_mm+bound
    radius = process['effective_contact_radius_mm']
    # Normal-line sigma does not bound tangential boundary localization error.
    boundary_bound = process.get('boundary_position_bound_mm')
    result['required_boundary_margin_mm'] = (process['boundary_margin_mm']+radius
        +process['following_error_bound_mm']+(np.nan if boundary_bound is None else boundary_bound))
    safe_boundary = result.boundary_distance_mm >= result.required_boundary_margin_mm
    protected = np.zeros(len(result), bool)
    for box in process.get('protected_uv_boxes', []):
        u0, u1, v0, v1 = box['bounds']
        protected |= ((result.face_id == box['face_id']) & result.u.between(u0,u1) & result.v.between(v0,v1))
    result['process_status'] = 'unobserved_or_rejected'
    valid = result.valid.astype(bool)
    result.loc[valid, 'process_status'] = 'uncertain_metrology'
    known = valid & complete
    result.loc[known, 'process_status'] = 'within_or_indeterminate'
    result.loc[known & (result.offset_upper_mm < -process['deficit_threshold_mm']), 'process_status'] = 'deficit'
    eligible = known & (result.offset_lower_mm > process['allowance_threshold_mm']) & safe_boundary & ~protected
    if process['material_side_confirmed']:
        result.loc[eligible, 'process_status'] = 'allowance_candidate'
    else:
        result.loc[eligible, 'process_status'] = 'material_side_unconfirmed'
    result.loc[known & ~safe_boundary, 'process_status'] = 'boundary_protection'
    if boundary_bound is None:
        result.loc[known, 'process_status'] = 'boundary_uncertainty_unknown'
    result.loc[valid & protected, 'process_status'] = 'protected_uv_region'
    result['process_eligible'] = result.process_status.eq('allowance_candidate')
    # Whole-surface polishing needs a supported contact frame even when removal
    # allowance and scanner accuracy have not been independently established.
    # Keep that geometrical role explicit and separate from removal eligibility.
    mode = process.get('mode','allowance_analysis')
    if mode not in ('allowance_analysis','surface_following'):
        raise ValueError('process.mode must be allowance_analysis or surface_following')
    clearance = float(process.get('geometric_edge_clearance_mm',0.))
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError('geometric_edge_clearance_mm must be finite and nonnegative')
    following = valid & ~protected & (result.boundary_distance_mm >= clearance)
    result['surface_following_eligible'] = following
    result['planning_role'] = mode
    if mode == 'surface_following':
        result['allowance_status'] = result['process_status']
        result.loc[following,'process_status'] = 'surface_following_contact'
        result.loc[valid & ~following & ~protected,'process_status'] = 'geometric_edge_protection'
    result['execution_allowed'] = False
    return result


def assign_process_regions(table, topology, cfg):
    """Connected allowance candidates on one exact face at sampling resolution.

    Neighbor graph edges are gated by 3D distance, normal agreement and exact
    trim visibility in the UV chart. This is not a sub-sampling defect detector.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from planning.cad_reference import _clip_uv_segment
    result = table.copy()
    result['region_id'] = -1
    process = cfg['process']
    radius = process.get('region_connection_radius_mm', 1.6)
    minimum = process.get('minimum_region_samples', 4)
    cosine = process.get('region_normal_cosine', .85)
    if not np.isfinite(radius) or radius <= 0 or not isinstance(minimum, int) or minimum < 1 or not 0 <= cosine <= 1:
        raise ValueError('Invalid region connectivity settings')
    next_region = 0
    for fid, group in result[result.process_eligible].groupby('face_id', sort=True):
        xyz = group[[f'cad_{a}_mm' for a in 'xyz']].to_numpy(float)
        normals = group[[f'normal_{a}' for a in 'xyz']].to_numpy(float)
        uv = group[['u','v']].to_numpy(float)
        edges = []
        for a, b in sorted(cKDTree(xyz).query_pairs(radius)):
            if normals[a]@normals[b] < cosine:
                continue
            length = np.linalg.norm(uv[b]-uv[a])
            spans = _clip_uv_segment(topology.faces[int(fid)], uv[a], uv[b])
            if sum(np.linalg.norm(y-x) for x,y in spans) < length-1e-7:
                continue
            # Avoid joining across the long way around a periodic chart.
            middle = topology.faces[int(fid)].positionAt(*map(float, (uv[a]+uv[b])/2)).toTuple()
            if np.linalg.norm(np.asarray(middle)-(xyz[a]+xyz[b])/2) > radius:
                continue
            edges.append((a,b))
        pairs = np.asarray(edges, int).reshape(-1,2)
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:,0],pairs[:,1])), shape=(len(group),len(group)))
        _, labels = connected_components(graph, directed=False)
        for label in np.unique(labels):
            ids = group.index[labels == label]
            if len(ids) < minimum:
                result.loc[ids, 'process_eligible'] = False
                result.loc[ids, 'process_status'] = 'small_candidate_region'
            else:
                result.loc[ids, 'region_id'] = next_region
                next_region += 1
    return result


def target_layer_table(table, depth, cfg):
    """Constant-depth parallel offsets from the fitted as-is surface.

Positive layers are restricted by lower-bound CAD allowance and material-side
confirmation. Normals are unchanged for a regular constant normal offset;
curvature singularities/tool engagement are not certified by this preview.
"""
    if not np.isfinite(depth) or depth < 0:
        raise ValueError('Invalid layer depth')
    result = table.copy()
    result['layer_depth_mm'] = depth
    if depth == 0:
        return result
    if not cfg['process']['material_side_confirmed']:
        raise ValueError('Material side must be confirmed for positive target layers')
    normal = result[[f'normal_{a}' for a in 'xyz']].to_numpy(float)
    cad_normal = result[[f'cad_normal_{a}' for a in 'xyz']].to_numpy(float)
    projection = np.einsum('ij,ij->i', normal, cad_normal)
    curvature = result.get('max_abs_principal_curvature_per_mm', pd.Series(np.nan, index=result.index))
    allowed = (result.valid.astype(bool) & result.process_eligible & (projection > 0)
               & (result.offset_lower_mm >= depth*projection)
               & np.isfinite(curvature) & (depth*curvature < .5)
               & (result.boundary_distance_mm >= result.required_boundary_margin_mm+depth))
    result.loc[~allowed, 'valid'] = False
    result.loc[~allowed, 'process_eligible'] = False
    result.loc[~allowed, 'status'] = 'target_layer_not_supported'
    for j, axis in enumerate('xyz'):
        result.loc[allowed, f'mapped_{axis}_mm'] -= depth*normal[allowed, j]
        result.loc[~allowed, f'mapped_{axis}_mm'] = np.nan
    # Rebuild segment identities after removing points, never bridge a lost sample.
    segment, previous = -1, None
    for idx, row in result.iterrows():
        if not row.valid:
            result.loc[idx, 'segment_id'] = -1
            previous = None
            continue
        if previous is None or row.path_id != previous.path_id or row.sample_index != previous.sample_index+1 or row.segment_id != previous.segment_id:
            segment += 1
        result.loc[idx, 'segment_id'] = segment
        previous = row
    sizes = result.loc[result.valid].groupby('segment_id').size()
    single = result.segment_id.isin(sizes[sizes < 2].index)
    result.loc[single, ['valid','status','segment_id']] = [False,'isolated_target_sample',-1]
    result.loc[~result.valid.astype(bool), 'process_eligible'] = False
    result.loc[~result.valid.astype(bool), [f'mapped_{a}_mm' for a in 'xyz']] = np.nan
    return result
