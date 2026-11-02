"""CAD-guided full scan segmentation, surface coverage and contact-axis mapping.

Reuses an existing registration. Outputs are CAD/SCAN_INPUT geometry only.
"""
import pyexpat  # Load system expat before OCCT shared libraries.
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
from collections import Counter
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'output/.mplconfig'))
import platform
import subprocess
import time
import numpy as np
import pandas as pd
from pipeline_common import ROOT, assert_rigid, read_ply, xyz_of, transform_points, sha256, dump_json
from planning.cad_topology import CadTopology
from planning.segmentation import segment_points
from planning.scan_mapping import map_reference_paths, export_contact_poses, validate_contact_poses, validate_mapping_config
from planning.surface_workflow import (validate_surface_config, build_surface_rasters,
    boundary_distances, classify_surface_samples, target_layer_table, assign_process_regions)


def stream_segment_scan(raw, transform, topology, segmentation_cfg, voxel_mm, dest,
                        chunk_points=20000, segmenter=None, write_csv=True):
    """Stream row-aligned full-resolution labels; retain actual source support.

    Only one XYZ block and one diagnostic DataFrame are resident at a time.
    Support de-duplication keys include the assigned face, preserving nearby
    thin-wall layers. The first original source point wins across all chunks.
    The NPY face sidecar is memmapped and the original PLY remains untouched.
    """
    if isinstance(chunk_points, bool) or not isinstance(chunk_points, int) or chunk_points < 1:
        raise ValueError('segmentation_chunk_points must be a positive integer')
    if not np.isfinite(voxel_mm) or voxel_mm <= 0:
        raise ValueError('scan_voxel_mm must be positive and finite')
    if not len(raw):
        raise ValueError('Scan contains no points')
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    labels_path = dest/'scan_face_labels.csv'
    if labels_path.exists() or (dest/'face_labels.npy').exists():
        raise FileExistsError('Segmentation sidecars already exist; use a new directory')
    face_labels = np.lib.format.open_memmap(dest/'face_labels.npy', mode='w+',
                                            dtype=np.int32, shape=(len(raw),))
    face_labels[:] = -2  # Unprocessed rows in an interrupted run, distinct from rejected=-1.
    face_labels.flush()
    diagnostic_dtype = np.dtype([('face_id','<i4'), ('candidate_face_id','<i4'),
        ('u','<f8'), ('v','<f8'), ('distance_mm','<f4'), ('signed_deviation_mm','<f4'),
        ('boundary_distance_mm','<f4'), ('confidence','<f4'), ('status_code','<u2')])
    diagnostic = np.lib.format.open_memmap(dest/'point_diagnostics.npy', mode='w+',
                                         dtype=diagnostic_dtype, shape=(len(raw),))
    status_codes = {}
    status_counts, face_counts = Counter(), Counter()
    seen, source_support, support_faces, support_uv = set(), [], [], []
    for start in range(0, len(raw), chunk_points):
        end = min(start+chunk_points, len(raw))
        points = transform_points(xyz_of(raw[start:end]), transform)
        if not np.isfinite(points).all():
            raise ValueError(f'Nonfinite scan positions in source rows {start}:{end}')
        source_ids = np.arange(start, end, dtype=np.int64)
        block = (segment_points(points, topology, segmentation_cfg, source_indices=source_ids)
                 if segmenter is None else segmenter.segment(points, segmentation_cfg, source_indices=source_ids))
        if write_csv:
            block.to_csv(labels_path, index=False, mode='w' if start == 0 else 'a', header=start == 0)
        for name in diagnostic_dtype.names:
            if name == 'status_code':
                for status in block.status.unique():
                    if status not in status_codes:
                        status_codes[status] = len(status_codes)
                diagnostic[name][start:end] = block.status.map(status_codes).to_numpy(np.uint16)
            else:
                diagnostic[name][start:end] = block[name].to_numpy()
        labels = block.face_id.to_numpy(np.int32)
        face_labels[start:end] = labels
        status_counts.update({str(k): int(v) for k,v in block.status.value_counts().items()})
        valid = block.valid.to_numpy(bool)
        block_uv = block[['u','v']].to_numpy(float)
        face_counts.update({int(k): int(v) for k,v in block.loc[valid, 'face_id'].value_counts().items()})
        ids = np.flatnonzero(valid)
        if len(ids):
            scaled = np.floor(points[ids]/voxel_mm)
            if np.any(np.abs(scaled) >= np.iinfo(np.int64).max):
                raise ValueError('Voxel coordinate exceeds int64 range; inspect scan units')
            keys = np.column_stack([scaled.astype(np.int64), labels[ids]])
            _, first = np.unique(keys, axis=0, return_index=True)
            for local in np.sort(first):
                key = tuple(keys[local])
                if key not in seen:
                    seen.add(key)
                    source_support.append(start+int(ids[local]))
                    support_faces.append(int(labels[ids[local]]))
                    support_uv.append(block_uv[ids[local]])
        face_labels.flush()
        diagnostic.flush()
        print(f'Segmentation rows {end}/{len(raw)}; accepted {sum(face_counts.values())}; '
              f'support {len(source_support)}', flush=True)
    del face_labels, diagnostic, seen
    dump_json(dest/'point_diagnostics_schema.json', {'source_order':'one row per original PLY vertex',
        'status_codes':status_codes, 'geometry_method':'exact_STEP' if segmenter is None else 'face_preserving_mesh',
        'uv_method':'exact_STEP' if segmenter is None else 'triangle_interpolation_of_OCCT_face_UV',
        'physical_accuracy_verified':False})
    source_support = np.asarray(source_support, dtype=np.int64)
    support_faces = np.asarray(support_faces, dtype=np.int32)
    support_points = np.empty((len(source_support), 3), dtype=float)
    provenance_path = dest/'support_provenance.csv'
    if not len(source_support):
        pd.DataFrame(columns=['support_index','source_index','face_id']).to_csv(provenance_path, index=False)
    for start in range(0, len(source_support), chunk_points):
        end = min(start+chunk_points, len(source_support))
        support_points[start:end] = transform_points(xyz_of(raw[source_support[start:end]]), transform)
        pd.DataFrame({'support_index': np.arange(start,end), 'source_index': source_support[start:end],
                      'face_id': support_faces[start:end]}).to_csv(provenance_path, index=False,
                        mode='w' if start == 0 else 'a', header=start == 0)
    compact = pd.DataFrame([{'face_id': key, 'valid': True, 'point_count': value}
                            for key,value in sorted(face_counts.items())],
                           columns=['face_id','valid','point_count'])
    return {'source_points': len(raw), 'accepted_face_points': sum(face_counts.values()),
            'status_counts': dict(sorted(status_counts.items())),
            'face_counts': dict(sorted(face_counts.items())),
            'support_points': support_points, 'support_faces': support_faces,
            'support_source_indices': source_support, 'compact_segmentation': compact,
            'support_uv': np.asarray(support_uv,float).reshape(-1,2),
            'chunk_points': chunk_points}


def _run(config_path, surface_path, dest, skip_visuals=False):
    cfg = json.loads(Path(config_path).read_text())
    scfg = json.loads(Path(surface_path).read_text())
    validate_surface_config(scfg)
    mapping = json.loads((ROOT/'cad_mapping.example.json').read_text())
    mapping.update(scfg['mapping'])
    mapping['path_sampling_mm'] = scfg['coverage']['path_sampling_mm']
    validate_mapping_config(mapping)
    if not np.isfinite(mapping['scan_voxel_mm']) or mapping['scan_voxel_mm'] <= 0:
        raise ValueError('scan_voxel_mm must be positive')
    if len(scfg['process']['layer_depths_mm']) > 1 and mapping['normal_mode'] != 'scan':
        raise ValueError('Positive target layers require scan-derived normals')
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Use a new empty run directory; preserve previous results')
    step, scan_path = ROOT/cfg['step_path'], ROOT/cfg['scan_path']
    upstream = ROOT/cfg['output_dir']
    if cfg.get('units') != 'mm':
        raise ValueError('This workflow requires mm')
    transform_path = upstream/'T_scan_to_CAD.npy'
    transform = np.load(transform_path)
    assert_rigid(transform)
    quality = json.loads((upstream/'registration_quality.json').read_text())
    if not np.allclose(transform, quality['selected_candidate']['T_final'], atol=1e-9, rtol=0):
        raise ValueError('Registration matrix and metadata differ')
    hashes = {'step': sha256(step), 'scan': sha256(scan_path), 'transform': sha256(transform_path)}
    for key, name in [('step','cad_metadata.json'), ('scan','scan_metadata.json')]:
        if hashes[key] != json.loads((upstream/name).read_text())['sha256']:
            raise ValueError(f'{key} changed since registration')
    dest.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    dump_json(dest/'run_state.json', {'status': 'running', 'stage': 'topology',
                                     'execution_allowed': False})
    raw, meta = read_ply(scan_path)
    if len(meta['elements']) != 1:
        raise ValueError('Input scan must contain only vertices')
    cache = scfg.get('cad_cache_dir')
    topology = (CadTopology.from_cache(step, ROOT/cache,
        linear_deflection_mm=cfg['mesh_linear_deflection_mm'],
        angular_deflection_rad=cfg['mesh_angular_deflection_rad']) if cache else
        CadTopology.from_step(step, linear_deflection_mm=cfg['mesh_linear_deflection_mm'],
                             angular_deflection_rad=cfg['mesh_angular_deflection_rad']))
    dump_json(dest/'face_topology.json', topology.catalog)
    topology.save_mesh(dest/'cad_face_mesh.npz')
    print(f'Topology: {len(topology.faces)} faces; segmenting {len(raw)} original points', flush=True)
    dump_json(dest/'run_state.json', {'status': 'running', 'stage': 'segmentation',
              'source_points': len(raw), 'execution_allowed': False})
    before = time.perf_counter()
    from planning.fast_segmentation import MeshFaceSegmenter
    segmenter = MeshFaceSegmenter(topology, boundary_sampling_mm=scfg.get('boundary_sampling_mm', .4),
        boundary_cache_dir=None if not cache else ROOT/cache/'boundaries')
    previous = scfg.get('reuse_segmentation_dir')
    if previous:
        previous = ROOT/previous
        record = json.loads((previous/'segmentation_summary.json').read_text())
        if record['input_sha256'] != hashes or record['segmentation_config'] != scfg['segmentation'] or record['voxel_mm'] != mapping['scan_voxel_mm']:
            raise ValueError('Reused segmentation does not match inputs/configuration')
        arrays = np.load(previous/'segmentation_support.npz', allow_pickle=False)
        segmented = {**record, **{key: arrays[key] for key in arrays.files}}
        segmented['compact_segmentation'] = pd.DataFrame(record['compact_segmentation'])
        segmented['face_counts'] = {int(k):int(v) for k,v in record['face_counts'].items()}
        label_dir = previous
    else:
        segmented = stream_segment_scan(raw, transform, topology, scfg['segmentation'],
            mapping['scan_voxel_mm'], dest, scfg.get('segmentation_chunk_points', 100000),
            segmenter=segmenter, write_csv=scfg.get('point_diagnostics_csv', False))
        np.savez_compressed(dest/'segmentation_support.npz', **{k:segmented[k] for k in
            ['support_points','support_faces','support_source_indices','support_uv']})
        record = {k:v for k,v in segmented.items() if k not in
                  ['support_points','support_faces','support_source_indices','support_uv','compact_segmentation']}
        record.update(input_sha256=hashes, segmentation_config=scfg['segmentation'],
            voxel_mm=mapping['scan_voxel_mm'], compact_segmentation=segmented['compact_segmentation'].to_dict('records'))
        dump_json(dest/'segmentation_summary.json', record)
        label_dir = dest
    support_points, support_faces = segmented['support_points'], segmented['support_faces']
    timing = {'segmentation_seconds': time.perf_counter()-before}
    audit_count = int(scfg.get('exact_audit_samples', 400))
    if audit_count > 0:
        audit_ids = np.unique(np.linspace(0, len(raw)-1, min(audit_count,len(raw))).astype(np.int64))
        audit_rows, audit_summary = segmenter.audit(transform_points(xyz_of(raw[audit_ids]),transform),
            scfg['segmentation'], source_indices=audit_ids)
        audit_rows.to_csv(dest/'exact_face_audit.csv',index=False)
        dump_json(dest/'exact_face_audit.json',audit_summary)
        print('Exact STEP audit:',json.dumps(audit_summary),flush=True)
    print(f"Segmentation: {segmented['accepted_face_points']}/{len(raw)} accepted; building coverage", flush=True)
    dump_json(dest/'run_state.json', {'status': 'running', 'stage': 'coverage_and_mapping',
              'source_points': len(raw), 'execution_allowed': False})
    paths, coverage = build_surface_rasters(topology, segmented['compact_segmentation'], scfg,
        mapping['approach_outward_C'], support_points_C=support_points,
        support_face_ids=support_faces, support_uv=segmented['support_uv'])
    pd.DataFrame(coverage).to_csv(dest/'face_selection.csv',index=False)
    if not paths:
        # Preserve full point labels/catalog even if no face supports a path.
        dump_json(dest/'summary.json', {'status': 'no_supported_faces', 'source_points': len(raw),
                  'segmentation_status_counts': segmented['status_counts'],
                  'execution_allowed': False})
        dump_json(dest/'resolved_config.json', {'mapping': mapping, **{k:v for k,v in scfg.items() if k != 'mapping'}})
        _manifest(dest, hashes, transform_path, timing)
        dump_json(dest/'run_state.json', {'status': 'no_supported_faces', 'execution_allowed': False})
        return dest
    reference = pd.DataFrame([{'path_id': p['path_id'], 'sample_index': i, 'face_id': int(p['face_ids'][i]),
        'u': p['uv'][i,0], 'v': p['uv'][i,1], **dict(zip(['x_mm','y_mm','z_mm'], xyz))}
        for p in paths for i, xyz in enumerate(p['points_C'])])
    reference.to_csv(dest/'cad_coverage_reference.csv', index=False)
    dump_json(dest/'coverage_sampling.json', coverage)
    before = time.perf_counter()
    table = map_reference_paths(paths, support_points, mapping, support_face_ids=support_faces)
    boundaries = boundary_distances(topology, table)
    table = classify_surface_samples(table, boundaries, scfg)
    if scfg['process'].get('mode') == 'surface_following':
        rejected = table.valid & ~table.surface_following_eligible
        table.loc[rejected, 'valid'] = False
        table.loc[rejected, 'status'] = 'geometric_edge_or_roi_protection'
        # Unknown removal allowance must not remove supported contact geometry.
        # Conversely, rejected edge/ROI samples must split the exported strokes.
        from planning.scan_mapping import finalize_pose_segments
        table = finalize_pose_segments(table,mapping)
    table = assign_process_regions(table, topology, scfg)
    table.to_csv(dest/'surface_field.csv', index=False)
    checks_by_layer, counts = [], []
    main_cad = None
    for layer, depth in enumerate(scfg['process']['layer_depths_mm']):
        layer_dir = dest/f'layer_{layer:03d}'
        layer_dir.mkdir()
        targets = target_layer_table(table, depth, scfg)
        cad, scan, TC, TS = export_contact_poses(targets, transform, mapping)
        checks = validate_contact_poses(cad, scan, TC, TS, transform, mapping)
        targets.to_csv(layer_dir/'path_mapping_diagnostics.csv', index=False)
        cad.to_csv(layer_dir/'tool_axis_poses_CAD.csv', index=False)
        scan.to_csv(layer_dir/'tool_axis_poses_SCAN.csv', index=False)
        checks = validate_contact_poses(pd.read_csv(layer_dir/'tool_axis_poses_CAD.csv'),
            pd.read_csv(layer_dir/'tool_axis_poses_SCAN.csv'), TC, TS, transform, mapping)
        np.savez_compressed(layer_dir/'tool_axis_transforms.npz', T_C_contact=TC, T_S_contact=TS,
                            T_scan_to_CAD=transform)
        checks_by_layer.append(checks)
        if layer == 0:
            main_cad = cad
            contact_keys = ['path_id','sample_index','segment_id','face_id','u','v',
                'x_mm','y_mm','z_mm',*[f'tool_{axis}_{c}' for axis in 'xyz' for c in 'xyz'],
                'qx','qy','qz','qw','normal_source','position_reference','frame','execution_allowed']
            contact_keys = [key for key in contact_keys if key in cad]
            cad[contact_keys].to_csv(dest/'contact_poses_CAD.csv',index=False)
            scan[contact_keys].to_csv(dest/'contact_poses_SCAN.csv',index=False)
        counts.append({'layer': layer, 'depth_mm': depth, 'poses': len(cad),
                       'segments': int(cad.segment_id.nunique()), 'status': 'geometric_preview'})
    timing['mapping_boundary_and_export_seconds'] = time.perf_counter()-before
    timing['total_seconds'] = time.perf_counter()-start
    report = {'status': 'completed', 'source_points': len(raw), 'support_points': len(support_points),
              'accepted_face_points': segmented['accepted_face_points'],
              'segmentation_chunk_points': segmented['chunk_points'],
              'segmentation_status_counts': segmented['status_counts'],
              'segmentation_face_counts': {str(k):int(v) for k,v in segmented['face_counts'].items()},
              'reference_paths': len(paths), 'reference_samples': len(table),
              'mapped_samples': int(table.valid.sum()),
              'process_region_count': int(table.loc[table.region_id >= 0, 'region_id'].nunique()),
              'process_status_counts': {str(k):int(v) for k,v in table.process_status.value_counts().items()},
              'layers': counts, 'checks': checks_by_layer, 'timing': timing,
              'normal_mode': mapping['normal_mode'], 'execution_allowed': False,
              'segmentation_directory': str(label_dir.resolve()),
              'physical_accuracy_verified': False, 'area_coverage_certified': False,
              'robot_planning_included': False, 'machining_feedback_included': False}
    dump_json(dest/'summary.json', report)
    dump_json(dest/'resolved_config.json', {'mapping': mapping, **{k:v for k,v in scfg.items() if k != 'mapping'}})
    _manifest(dest, hashes, transform_path, timing)
    _report(dest, report)
    if not skip_visuals:
        from planning.real_presentation import write_real_deliverables
        write_real_deliverables(dest, raw, transform, topology, label_dir, main_cad, coverage, report)
    dump_json(dest/'run_state.json', {'status': 'completed', 'execution_allowed': False})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return dest


def run(config_path, surface_path, dest, skip_visuals=False):
    """Record failed/incomplete runs without overwriting completed run archives."""
    existing = Path(dest).resolve()
    was_nonempty = existing.exists() and any(existing.iterdir())
    try:
        return _run(config_path, surface_path, dest, skip_visuals)
    except Exception as error:
        state_path = Path(dest).resolve()/'run_state.json'
        if not was_nonempty and state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get('status') == 'running':
                dump_json(state_path, {**state, 'status': 'failed',
                          'error_type': type(error).__name__, 'error': str(error)})
        raise


def _manifest(dest, hashes, transform_path, timing):
    sources = list((ROOT/'planning').glob('*.py'))+[Path(__file__), ROOT/'surface_workflow.example.json',
                                                   ROOT/'cad_mapping.example.json', ROOT/'pipeline_common.py']
    dump_json(dest/'manifest.json', {'input_sha256': hashes, 'transform_path': str(transform_path),
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT).decode().strip(),
        'git_branch': subprocess.check_output(['git','branch','--show-current'], cwd=ROOT).decode().strip(),
        'working_tree_dirty': bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()),
        'python': platform.python_version(), 'numpy': np.__version__, 'timing': timing,
        'registration_reused': True, 'execution_allowed': False})


def _report(dest, report):
    lines = ['# CAD 型面分割、覆盖路径与刀轴映射', '',
             f"原始点 {report['source_points']}；确定面归属 {report['accepted_face_points']}。",
             f"覆盖参考路径 {report['reference_paths']} 条、样本 {report['reference_samples']}，有效映射 {report['mapped_samples']}。", '',
             f"法向模式：{report['normal_mode']}。所有输出均为 CAD/输入扫描坐标下的几何预览。", '',
             'scan_face_labels.csv 与原始 PLY 逐行对应，source_index 不改变；原始标量属性保留在原文件。',
             'face_topology.json、cad_face_mesh.npz 保留 STEP 面/真实共享边/三角形/顶点 UV 的来源。',
             'surface_field.csv 同时保留未支持、被拒绝、材料侧未确认及误差未知的位置。',
             'layer_000 是实际拟合基体接触路径；正深度层是沿实际法向的常量偏置，需明确配置。',
             '拟合曲面为局部二次/平面模型，不声称跨邻域全局 C1 连续。路径在缺口和姿态跳变处断开。',
             '行距采用采样曲面度量估计；不是全局严格覆盖或残留高度证明。',
             '误差为用户提供独立标准差的条件 RSS；不能由 ICP 残差推导绝对精度。',
             '边界保护使用配置接触半径及误差裕度；不代表完整工具包络碰撞验证。',
             '未涉及机器人基座标定、IK、设备执行、加工或复测闭环。', '',
             '## 分层输出', '', '| 层 | 目标深度 mm | 位姿 | 连续段 |', '|---|---:|---:|---:|']
    lines.extend(f"| {x['layer']} | {x['depth_mm']} | {x['poses']} | {x['segments']} |" for x in report['layers'])
    (dest/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--surface-config', type=Path, default=ROOT/'surface_workflow.example.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--skip-visuals', action='store_true')
    args = parser.parse_args()
    run(args.config, args.surface_config, args.run_dir, args.skip_visuals)


if __name__ == '__main__':
    main()
