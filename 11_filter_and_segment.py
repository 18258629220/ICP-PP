"""Reuse a verified registration, filter scan clutter, then assign CAD face IDs.

This entry point stops at point-cloud segmentation. It never constructs paths,
surface contact frames, robot poses, or machining instructions.
"""
import pyexpat  # Load system expat before OCCT shared libraries.
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd

from pipeline_common import ROOT, assert_rigid, dump_json, read_ply, sha256, transform_points, xyz_of
from planning.cad_topology import CadTopology
from planning.fast_segmentation import MeshFaceSegmenter
from planning.frames import BASE_FRAME, cad_to_base_transform
from planning.segmentation import segmentation_config
from planning.workpiece_filter import filter_scan_by_cad, write_point_subset_ply


def _validated_inputs(config_path):
    cfg = json.loads(Path(config_path).read_text())
    if cfg.get('scan_frame') != BASE_FRAME or cfg.get('units') != 'mm':
        raise ValueError("Require units='mm' and user-confirmed scan_frame='ROBOT_BASE'")
    step, scan = ROOT / cfg['step_path'], ROOT / cfg['scan_path']
    upstream = ROOT / cfg['output_dir']
    matrix_path = upstream / 'T_scan_to_CAD.npy'
    matrix = np.load(matrix_path, allow_pickle=False)
    assert_rigid(matrix)
    quality = json.loads((upstream / 'registration_quality.json').read_text())
    if quality.get('passed') is not True:
        raise ValueError('Existing registration did not pass its screening gate')
    if not np.allclose(matrix, quality['selected_candidate']['T_final'], atol=1e-9, rtol=0):
        raise ValueError('Saved registration matrix differs from registration metadata')
    hashes = {'step': sha256(step), 'scan': sha256(scan), 'transform': sha256(matrix_path)}
    for key, name in [('step', 'cad_metadata.json'), ('scan', 'scan_metadata.json')]:
        if hashes[key] != json.loads((upstream / name).read_text())['sha256']:
            raise ValueError(f'{key} changed since registration; refuse stale alignment')
    return cfg, step, scan, matrix_path, matrix, hashes


def _segment_retained(raw, source_indices, transform, segmenter, cfg, dest, chunk_points):
    """Label retained rows while recording their original, global scan indices."""
    count = len(source_indices)
    if not count:
        raise ValueError('CAD distance filter retained no points')
    labels = np.lib.format.open_memmap(dest / 'face_labels.npy', mode='w+',
                                      dtype=np.int32, shape=(count,))
    labels[:] = -2
    dtype = np.dtype([('source_index', '<i8'), ('face_id', '<i4'),
                      ('candidate_face_id', '<i4'), ('u', '<f8'), ('v', '<f8'),
                      ('distance_mm', '<f4'), ('signed_deviation_mm', '<f4'),
                      ('boundary_distance_mm', '<f4'), ('confidence', '<f4'),
                      ('status_code', '<u2')])
    diagnostics = np.lib.format.open_memmap(dest / 'point_diagnostics.npy', mode='w+',
                                           dtype=dtype, shape=(count,))
    status_codes, status_counts, face_counts = {}, Counter(), Counter()
    started = time.perf_counter()
    for start in range(0, count, chunk_points):
        stop = min(count, start + chunk_points)
        original_ids = np.asarray(source_indices[start:stop])
        points = transform_points(xyz_of(raw[original_ids]), transform)
        block = segmenter.segment(points, cfg, source_indices=original_ids)
        labels[start:stop] = block.face_id.to_numpy(np.int32)
        for status in block.status.unique():
            if status not in status_codes:
                status_codes[status] = len(status_codes)
        for name in dtype.names:
            if name == 'status_code':
                diagnostics[name][start:stop] = block.status.map(status_codes).to_numpy(np.uint16)
            else:
                diagnostics[name][start:stop] = block[name].to_numpy()
        status_counts.update({str(k): int(v) for k, v in block.status.value_counts().items()})
        face_counts.update({int(k): int(v) for k, v in
                            block.loc[block.valid, 'face_id'].value_counts().items()})
        labels.flush()
        diagnostics.flush()
        print(f'Segmentation retained rows {stop}/{count}; '
              f'assigned {sum(face_counts.values())}', flush=True)
    del labels, diagnostics
    schema = {
        'source_order': 'one row per vertex of scan_workpiece_BASE.ply and scan_segmented_BASE.ply',
        'original_scan_index': 'source_index is zero-based in the original unfiltered input PLY',
        'source_indices_file': 'source_indices.npy', 'source_frame': BASE_FRAME,
        'uv_frame': 'CAD_LOCAL', 'scalar_distances_are_frame_invariant': True,
        'status_codes': status_codes, 'geometry_method': 'face_preserving_mesh',
        'uv_method': 'triangle_interpolation_of_OCCT_face_UV',
        'physical_accuracy_verified': False,
    }
    dump_json(dest / 'point_diagnostics_schema.json', schema)
    return {
        'source_points': count, 'accepted_face_points': sum(face_counts.values()),
        'unassigned_points': count - sum(face_counts.values()),
        'assigned_face_count': len(face_counts), 'status_counts': dict(sorted(status_counts.items())),
        'face_counts': {str(k): int(v) for k, v in sorted(face_counts.items())},
        'segmentation_config': cfg, 'segmentation_seconds': time.perf_counter() - started,
        'normal_filter_applied': False,
    }


def _verify_exports(raw, kept, removed, dest, face_count, chunk_points):
    """Check every retained/deleted source attribute and each exported face label."""
    checks = {'passed': False, 'original_attributes_bitwise_preserved': list(raw.dtype.names),
              'original_scan_xyz_transformed': False, 'output_frame': BASE_FRAME,
              'point_order': 'original source order within each subset',
              'physical_accuracy_verified': False}
    mask = np.load(dest / 'filter_keep_mask.npy', mmap_mode='r')
    if mask.shape != (len(raw),) or mask.dtype != np.bool_:
        raise ValueError('Filter mask does not align with original scan')
    if len(kept) + len(removed) != len(raw) or np.count_nonzero(mask) != len(kept):
        raise ValueError('Retained and removed point counts do not partition the original scan')
    for indices, keep_value in [(kept, True), (removed, False)]:
        if len(indices) and (indices[0] < 0 or indices[-1] >= len(raw) or
                             np.any(np.diff(indices) <= 0) or
                             np.any(mask[indices] != keep_value)):
            raise ValueError('Source indices have invalid bounds, ordering, or mask values')
    labels = np.load(dest / 'face_labels.npy', mmap_mode='r')
    diagnostics = np.load(dest / 'point_diagnostics.npy', mmap_mode='r')
    if labels.shape != (len(kept),) or diagnostics.shape != labels.shape:
        raise ValueError('Filtered labels/diagnostics differ from the retained point count')
    if np.any(labels < -1) or np.any(labels >= face_count):
        raise ValueError('Incomplete segmentation or out-of-range CAD face IDs')
    if not np.array_equal(diagnostics['source_index'], kept) or not np.array_equal(diagnostics['face_id'], labels):
        raise ValueError('Diagnostics do not preserve original indices and exported labels')
    counts = {}
    for name, indices in [('scan_workpiece_BASE.ply', kept),
                           ('scan_removed_BASE.ply', removed),
                           ('scan_segmented_BASE.ply', kept)]:
        # A zero-point removed cloud is legal, but the shared PLY reader rejects it.
        if not len(indices):
            header = (dest / name).read_bytes().split(b'end_header', 1)[0]
            if b'element vertex 0\n' not in header:
                raise ValueError('Empty removed cloud has an invalid point count')
            counts[name] = 0
            continue
        actual, _ = read_ply(dest / name)
        if len(actual) != len(indices):
            raise ValueError(f'{name} point count differs from its source index array')
        for start in range(0, len(indices), chunk_points):
            stop = min(len(indices), start + chunk_points)
            expected = raw[indices[start:stop]]
            for field in raw.dtype.names:
                original_dtype = raw.dtype.fields[field][0]
                serialized = actual[field][start:stop].astype(original_dtype, copy=False)
                if serialized.tobytes() != expected[field].tobytes():
                    raise ValueError(f'{name} changed source field {field}')
            if name == 'scan_segmented_BASE.ply' and not np.array_equal(
                    actual['cad_face_id'][start:stop], labels[start:stop]):
                raise ValueError('PLY CAD face IDs differ from the label sidecar')
        counts[name] = len(actual)
    checks.update(passed=True, source_points=len(raw), retained_points=len(kept),
                  removed_points=len(removed), exported_counts=counts,
                  labels_match_segmented_ply=True, source_indices_match_diagnostics=True,
                  path_planning_executed=False)
    return checks


def _git_value(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def run(config_path, surface_path, dest, threshold_mm, cad_cache_dir=None,
        chunk_points=100000, nthreads=4):
    if (isinstance(chunk_points, bool) or not isinstance(chunk_points, int) or chunk_points < 1 or
            isinstance(nthreads, bool) or not isinstance(nthreads, int) or nthreads < 1):
        raise ValueError('chunk_points and nthreads must be positive integers')
    if not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError('distance-mm must be finite and positive')
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Use a new empty output directory; preserve previous runs')
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stage = 'input_validation'
    dump_json(dest / 'run_state.json', {'status': 'running', 'stage': stage,
              'output_frame': BASE_FRAME, 'path_planning_executed': False})
    try:
        cfg, step, scan, matrix_path, transform, hashes = _validated_inputs(config_path)
        scfg = json.loads(Path(surface_path).read_text())
        seg_cfg = segmentation_config(scfg.get('segmentation'))
        if threshold_mm < seg_cfg['max_distance_mm']:
            raise ValueError('CAD clutter threshold must not be smaller than the face-assignment distance')
        cache = cad_cache_dir or scfg.get('cad_cache_dir')
        cache = None if cache is None else ROOT / cache
        raw, meta = read_ply(scan)
        if len(meta['elements']) != 1:
            raise ValueError('Input scan must contain only vertices')
        topology = (CadTopology.from_cache(step, cache,
                    linear_deflection_mm=cfg['mesh_linear_deflection_mm'],
                    angular_deflection_rad=cfg['mesh_angular_deflection_rad']) if cache else
                    CadTopology.from_step(step, linear_deflection_mm=cfg['mesh_linear_deflection_mm'],
                                          angular_deflection_rad=cfg['mesh_angular_deflection_rad']))
        inverse = cad_to_base_transform(transform)
        for name, matrix in [('T_BASE_to_CAD', transform), ('T_CAD_to_BASE', inverse)]:
            np.save(dest / f'{name}.npy', matrix)
            np.savetxt(dest / f'{name}.txt', matrix, fmt='%.15g')
        dump_json(dest / 'face_topology.json', {**topology.catalog, 'geometry_frame': 'CAD_LOCAL',
                  'output_frame': BASE_FRAME, 'T_CAD_to_BASE': inverse.tolist()})
        segmenter = MeshFaceSegmenter(topology,
                    boundary_sampling_mm=scfg.get('boundary_sampling_mm', .4),
                    boundary_cache_dir=None if cache is None else cache / 'boundaries', nthreads=nthreads)
        dump_json(dest / 'input_config.json', cfg)
        dump_json(dest / 'input_surface_config.json', scfg)
        resolved = {'distance_threshold_mm': threshold_mm, 'segmentation': seg_cfg,
                    'chunk_points': chunk_points, 'threads': nthreads,
                    'cad_cache_dir': None if cache is None else str(cache.resolve()),
                    'boundary_sampling_mm': scfg.get('boundary_sampling_mm', .4),
                    'input_frame': BASE_FRAME, 'output_frame': BASE_FRAME, 'units': 'mm',
                    'stop_after': 'CAD_face_segmentation', 'path_planning_executed': False}
        dump_json(dest / 'resolved_config.json', resolved)
        stage = 'CAD_distance_filter'
        dump_json(dest / 'run_state.json', {'status': 'running', 'stage': stage,
                  'source_points': len(raw), 'path_planning_executed': False})
        filtered = filter_scan_by_cad(raw, transform, segmenter, dest,
                                     threshold_mm, chunk_points=chunk_points)
        dump_json(dest / 'filter_summary.json', filtered)
        kept = np.load(dest / 'source_indices.npy', mmap_mode='r')
        removed = np.load(dest / 'removed_source_indices.npy', mmap_mode='r')
        write_point_subset_ply(dest / 'scan_workpiece_BASE.ply', raw, kept, chunk_points=chunk_points)
        write_point_subset_ply(dest / 'scan_removed_BASE.ply', raw, removed, chunk_points=chunk_points)
        stage = 'CAD_face_segmentation'
        dump_json(dest / 'run_state.json', {'status': 'running', 'stage': stage,
                  'source_points': len(raw), 'retained_points': len(kept),
                  'path_planning_executed': False})
        segmented = _segment_retained(raw, kept, transform, segmenter, seg_cfg, dest, chunk_points)
        labels = np.load(dest / 'face_labels.npy', mmap_mode='r')
        write_point_subset_ply(dest / 'scan_segmented_BASE.ply', raw, kept,
                               face_labels=labels, chunk_points=chunk_points)
        rows = []
        for key, count in segmented['face_counts'].items():
            face = topology.catalog['faces'][int(key)]
            rows.append({'face_id': int(key), 'point_count': count,
                         'geometry_type': face['geometry_type'], 'area_mm2': face['area_mm2']})
        pd.DataFrame(rows, columns=['face_id', 'point_count', 'geometry_type', 'area_mm2']).to_csv(
            dest / 'face_counts.csv', index=False)
        dump_json(dest / 'segmentation_summary.json', {**segmented, 'input_sha256': hashes,
                  'original_scan_points': len(raw), 'source_indices_file': 'source_indices.npy',
                  'output_frame': BASE_FRAME, 'path_planning_executed': False})
        stage = 'verification'
        verification = _verify_exports(raw, kept, removed, dest, len(topology.faces), chunk_points)
        dump_json(dest / 'verification.json', verification)
        forbidden = ('contact_poses*', 'contact_transforms*', 'cad_reference*', 'surface_field*', 'coverage*')
        if any(any(dest.glob(pattern)) for pattern in forbidden):
            raise ValueError('Unexpected path-planning output in segmentation-only run')
        summary = {'status': 'completed', 'source_points': len(raw), 'retained_points': len(kept),
                   'removed_points': len(removed), 'distance_threshold_mm': threshold_mm,
                   'accepted_face_points': segmented['accepted_face_points'],
                   'unassigned_retained_points': segmented['unassigned_points'],
                   'assigned_faces': segmented['assigned_face_count'], 'CAD_faces': len(topology.faces),
                   'output_frame': BASE_FRAME, 'units': 'mm', 'registration_recomputed': False,
                   'original_scan_xyz_transformed': False, 'path_planning_executed': False,
                   'physical_accuracy_verified': False, 'verification_passed': verification['passed'],
                   'elapsed_seconds': time.perf_counter() - started}
        dump_json(dest / 'summary.json', summary)
        source_files = [Path(__file__), ROOT / 'planning/workpiece_filter.py',
                        ROOT / 'planning/fast_segmentation.py', ROOT / 'planning/cad_topology.py',
                        ROOT / 'planning/cad_reference.py', ROOT / 'planning/frames.py',
                        ROOT / 'planning/segmentation.py', ROOT / 'pipeline_common.py']
        manifest = {'input_sha256': hashes, 'transform_path': str(matrix_path.resolve()),
                    'config_path': str(Path(config_path).resolve()),
                    'surface_config_path': str(Path(surface_path).resolve()),
                    'cad_cache_dir': None if cache is None else str(cache.resolve()),
                    'cad_cache_sha256': {} if cache is None else
                        {name: sha256(cache / name) for name in
                         ['face_topology.json', 'cad_face_mesh.npz', 'cache_manifest.json']
                         if (cache / name).exists()},
                    'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
                    'git_commit': _git_value('rev-parse', 'HEAD'),
                    'git_branch': _git_value('branch', '--show-current'),
                    'working_tree_dirty': bool(_git_value('status', '--porcelain')),
                    'python': platform.python_version(), 'numpy': np.__version__,
                    'registration_reused': True, 'path_planning_executed': False,
                    'input_scan_frame': BASE_FRAME, 'output_frame': BASE_FRAME,
                    'created_outputs': sorted(p.name for p in dest.iterdir() if p.is_file())}
        dump_json(dest / 'manifest.json', manifest)
        (dest / 'REPORT_zh.md').write_text(
            '# CAD 引导的工件点云筛选与型面分割\n\n'
            f'- 原始扫描：{len(raw):,} 点；沿用原有粗细配准，不重新拟合。\n'
            f'- 无符号点到 CAD 三角面的距离阈值：{threshold_mm:g} mm。\n'
            f'- 保留：{len(kept):,} 点；剔除：{len(removed):,} 点。\n'
            f'- 型面归属成功：{segmented["accepted_face_points"]:,} 点，'
            f'涉及 {segmented["assigned_face_count"]:,} 个 CAD 型面。\n'
            f'- 保留但面归属未知：{segmented["unassigned_points"]:,} 点。\n'
            '- 主要文件 scan_segmented_BASE.ply 仅含筛选后点；原始 XYZ、标量属性及相对顺序保持。\n'
            '- face_labels.npy 与新筛选点云逐行对应；source_indices.npy 保存各行在原始扫描中的零起始索引。\n'
            '- scan_workpiece_BASE.ply 为筛选后原属性点云；scan_removed_BASE.ply 保存被距离规则剔除的点供复核。\n'
            '- cad_face_id>=0 表示型面归属通过；-1 表示边界或归属歧义等未知点，不自动删除。\n'
            '- 全部已归属型面按面号分色，未知点为灰色；面号沿用当前 STEP 导入的零起始编号。\n'
            '- 距离筛选保留 CAD 邻域，不能识别与工件贴合、几何上无法区分的夹具点。\n'
            '- CAD 未离散型面附近采取保守保留，具体数量见 filter_summary.json。\n'
            '- 坐标均为 ROBOT_BASE，单位 mm。流程在 CAD 型面分割后结束，未生成任何路径或刀轴。\n'
            '- 软件坐标/属性一致性检查通过，不代表独立实物测量精度已验证。\n', encoding='utf-8')
        dump_json(dest / 'run_state.json', {'status': 'completed', 'stage': 'CAD_face_segmentation',
                  'path_planning_executed': False, 'verification_passed': True})
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return dest
    except Exception as exc:
        dump_json(dest / 'run_state.json', {'status': 'failed', 'stage': stage,
                  'error_type': type(exc).__name__, 'error': str(exc), 'path_planning_executed': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--surface-config', type=Path, default=ROOT / 'surface_workflow.example.json')
    parser.add_argument('--cad-cache-dir', type=Path)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--distance-mm', type=float, required=True)
    parser.add_argument('--chunk-points', type=int, default=100000)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    run(args.config, args.surface_config, args.run_dir, args.distance_mm,
        args.cad_cache_dir, args.chunk_points, args.threads)


if __name__ == '__main__':
    main()
