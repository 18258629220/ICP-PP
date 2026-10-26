"""Traceable candidate protrusions, not confirmed burr/material labels."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .geometry import unit


def local_prominence(points, normals, signed, seed_mask, radius, normal_cosine):
    tree = cKDTree(points)
    seeds = np.flatnonzero(seed_mask)
    baseline = np.full(len(points), np.nan)
    for start in range(0, len(seeds), 2000):
        ids = seeds[start:start+2000]
        distances, indices = tree.query(points[ids], k=min(48, len(points)), distance_upper_bound=radius, workers=2)
        if indices.ndim == 1:
            indices, distances = indices[:, None], distances[:, None]
        valid = (indices < len(points)) & (distances > 1e-6)
        safe_indices = np.minimum(indices, len(points)-1)
        valid &= np.einsum('ijk,ik->ij', normals[safe_indices], normals[ids]) >= normal_cosine
        for row, index in enumerate(ids):
            neighbors = safe_indices[row, valid[row]]
            if len(neighbors) >= 5:
                baseline[index] = np.median(signed[neighbors])
    return baseline


def connected_regions(points, normals, radius, minimum_size, normal_cosine):
    """3D adjacency preserves separated layers; normal gating prevents sharp-fold bridges."""
    if not len(points):
        return np.empty(0, dtype=int)
    edges = cKDTree(points).query_pairs(radius, output_type='ndarray')
    if len(edges):
        edges = edges[np.einsum('ij,ij->i', normals[edges[:, 0]], normals[edges[:, 1]]) >= normal_cosine]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    retained = sorted(np.flatnonzero(counts >= minimum_size), key=lambda i: (-counts[i], int(i)))
    mapping = {int(old): new for new, old in enumerate(retained)}
    return np.array([mapping.get(int(lab), -1) for lab in labels])


def classify(frame, cfg):
    p = frame[['cad_closest_x_mm', 'cad_closest_y_mm', 'cad_closest_z_mm']].to_numpy()
    n = frame[['normal_x', 'normal_y', 'normal_z']].to_numpy()
    d = frame.signed_deviation_mm.to_numpy()
    good = frame.quality_mask.to_numpy(bool)
    eligible = good & ((n@unit(cfg['approach_outward_C'])) >= cfg['minimum_normal_alignment'])
    seed = eligible & (d > cfg['candidate_threshold_mm'])
    baseline = np.full(len(frame), np.nan)
    # Quality/approach filtering also applies to neighbors, not just seed points.
    ids = np.flatnonzero(eligible)
    baseline[ids] = local_prominence(p[ids], n[ids], d[ids], seed[ids], cfg['local_radius_mm'], cfg['normal_neighbor_cosine'])
    excess = d-baseline
    candidate = seed & (excess >= cfg['local_prominence_mm'])
    labels = np.full(len(frame), -1, int)
    labels[candidate] = connected_regions(p[candidate], n[candidate], cfg['cluster_eps_mm'], cfg['region_min_points'], cfg['normal_neighbor_cosine'])
    reasons = np.full(len(frame), 'below_candidate_threshold', dtype=object)
    reasons[~good] = 'uncertain_correspondence'
    reasons[good & ~eligible] = 'other_surface_or_approach_direction'
    reasons[seed & ~np.isfinite(baseline)] = 'insufficient_local_support'
    reasons[seed & np.isfinite(baseline) & ~candidate] = 'broad_offset_or_low_local_prominence'
    reasons[candidate & (labels < 0)] = 'small_or_isolated_candidate'
    reasons[labels >= 0] = 'candidate_preview_region'
    result = frame.copy()
    result.insert(0, 'analysis_row', np.arange(len(frame)))
    result['local_baseline_mm'] = baseline
    result['local_excess_mm'] = excess
    result['candidate_mask'] = candidate
    result['region_id'] = labels
    result['classification'] = reasons
    result['material_sign_confirmed'] = cfg['material_side_confirmed']
    return result
