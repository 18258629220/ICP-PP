"""Generate CAD-native curves, bounded scan projection, and contact-frame poses."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'output/.mplconfig'))
import argparse
import json
import platform
import subprocess
import time
import numpy as np
import pandas as pd
from pipeline_common import ROOT, assert_rigid, read_ply, xyz_of, transform_points, sha256, dump_json
from planning.scan_mapping import map_reference_paths, export_contact_poses, validate_contact_poses, validate_mapping_config


def summarize(table, poses, checks):
    valid = table.loc[table.valid]
    lengths = []
    for _, part in poses.groupby('segment_id', sort=False):
        lengths.append(float(np.linalg.norm(np.diff(part[['x_mm','y_mm','z_mm']], axis=0), axis=1).sum()))
    offsets = valid.offset_mm.to_numpy() if len(valid) else np.array([])
    return {
        'reference_path_count': int(table.path_id.nunique()), 'reference_samples': len(table),
        'accepted_samples': len(poses), 'accepted_sample_fraction': len(poses)/len(table),
        'continuous_segments': int(poses.segment_id.nunique()),
        'supported_path_length_mm': sum(lengths), 'segment_lengths_mm': lengths,
        'status_counts': {str(k):int(v) for k,v in table.status.value_counts().items()},
        'absolute_normal_offset_median_mm': float(np.median(np.abs(offsets))),
        'absolute_normal_offset_p95_mm': float(np.percentile(np.abs(offsets),95)),
        'absolute_normal_offset_max_mm': float(np.max(np.abs(offsets))),
        'local_fit_rms_median_mm': float(valid.fit_rms_mm.median()),
        'checks': checks, 'registration_accuracy_verified': False,
        'material_side_confirmed': False, 'tool_geometry_confirmed': False,
        'robot_base_transform_included': False, 'ik_included': False, 'execution_allowed': False,
        'mapping_scope': 'normal-line local substrate fit; no tangential edge correction or nonrigid warping',
        'normal_source': 'Exact STEP face normal; sign selected toward configured approach side',
        'position_reference': 'surface_contact_point',
        'sample_fraction_is_not_surface_area_coverage': True,
    }


def per_path_summary(paths, table, poses):
    rows = []
    for path in paths:
        part = poses[poses.path_id == path['path_id']]
        length = 0.
        for _, group in part.groupby('segment_id', sort=False):
            length += np.linalg.norm(np.diff(group[['x_mm','y_mm','z_mm']],axis=0),axis=1).sum()
        normals = path['normals_C']
        rows.append({'path_id':path['path_id'], 'face_id':int(path['face_ids'][0]),
                     'geometry_type':path['selection'].get('geometry_type','explicit'),
                     'reference_samples':int((table.path_id == path['path_id']).sum()),
                     'accepted_samples':len(part), 'segments':part.segment_id.nunique(),
                     'reference_length_mm':float(np.linalg.norm(np.diff(path['points_C'],axis=0),axis=1).sum()),
                     'accepted_length_mm':float(length),
                     'reference_normal_end_change_deg':float(np.degrees(np.arccos(np.clip(normals[0]@normals[-1],-1,1))))})
    return pd.DataFrame(rows)


def write_report(dest, cfg, report):
    r = report
    lines = ['# CAD 路径与扫描映射结果', '',
        '本次只处理 CAD、扫描与刀轴几何。位置参考点为表面接触点；全部长度单位 mm。', '',
        f"自动/手动生成 {r['reference_path_count']} 条 CAD 参考路径，采样 {r['reference_samples']} 点；",
        f"保留 {r['accepted_samples']} 个支持充分的位姿，拆为 {r['continuous_segments']} 个独立段，段内总长 {r['supported_path_length_mm']:.2f} mm。",
        f"采样点保留比例 {r['accepted_sample_fraction']:.1%}，不是工件表面覆盖率。", '',
        f"局部法向修正绝对值：中位 {r['absolute_normal_offset_median_mm']:.4f} mm，P95 {r['absolute_normal_offset_p95_mm']:.4f} mm，最大 {r['absolute_normal_offset_max_mm']:.4f} mm。",
        f"局部曲面拟合残差中位 {r['local_fit_rms_median_mm']:.4f} mm；数值位姿检查 {'通过' if r['checks']['passed'] else '失败'}。",
        '这些量描述局部几何贴合，不是独立测量的配准精度。', '',
        '## 文件', '', '| 文件 | 内容 |', '|---|---|',
        '| CAD_path_overview.png | CAD 上的原始路径、映射路径、法向与被拒绝样本 |',
        '| path_mapping_detail.png | 最长有效路径附近的 CAD / 扫描局部贴合与修正量 |',
        '| pose_axes_detail.png | 接触点及刀轴坐标系 X/Y/Z；Z 指向工件 |',
        '| CAD_scan_paths.html | 可离线旋转、缩放及隐藏图层的三维预览 |',
        '| cad_reference_path.csv | 所有 CAD 设计接触点及解析法向，包含未支持部分 |',
        '| path_mapping_diagnostics.csv | 全部采样点、有效性、拒绝原因、局部拟合指标 |',
        '| tool_axis_poses_CAD.csv | 有效接触点及刀轴完整姿态，CAD 坐标 |',
        '| tool_axis_poses_CAD_compact.csv | 精简位姿表：接触点、刀轴方向及四元数 |',
        '| tool_axis_poses_SCAN.csv | 相同位姿转换回原始输入扫描文件坐标 |',
        '| tool_axis_transforms.npz | T_C_contact、T_S_contact 及示例球心变换矩阵 |',
        '| selected_paths.json / face_catalog.json | 路径选择记录、STEP 面编号与候选信息 |',
        '| mapping_config.json / manifest.json / summary.json | 参数、输入与源码哈希、数值检查 |', '',
        '## 位姿的含义', '',
        '- `x_mm,y_mm,z_mm`：表面接触点，不是法兰或已标定 TCP。',
        '- `tool_z_x/y/z`：刀轴单位方向，工具 +Z 指向工件，等于朝工具一侧法向的负方向。',
        '- `tool_x_x/y/z`：路径前进方向在切平面上的投影；+Y 按右手系生成。',
        '- `qx,qy,qz,qw`：XYZW 四元数，主动旋转，将刀轴局部坐标映射到 CSV `frame` 所示坐标系。',
        '- 变换矩阵前三列是局部 X/Y/Z 轴，最后一列是接触点。`T_C_contact = T_scan_to_CAD @ T_S_contact`。',
        f"- `example_ball_center_*_mm`：仅按半径 {cfg['example_ball_radius_mm']} mm 球形磨头示例计算；球心 = 接触点 + 半径 × 朝工具侧法向。",
        '- `segment_id` 不同时必须断开；此处没有段间连接、抬刀或进退刀规划。',
        '- 正常 CSV 只包含有效连续段；被拒绝样本保存在诊断表，不能据此跨缺口连接。', '',
        '## 算法与限制', '',
        '路径在原始 STEP 的裁剪曲面上生成，法向从曲面直接计算；支持范围受扫描数据约束。',
        '沿 CAD 法向在有限邻域内拟合扫描局部曲面，使用鲁棒权重排除离群值，并拒绝多层、外推、缺失、噪声过大和法向分歧。',
        '主刀轴使用 CAD 法向先验；扫描拟合法向及夹角单独保存，未自动混合。',
        '映射后的接触点表示当前表面附近的拟合基体，不是去毛刺的最终去除终点；CAD 原始路径保留为设计参考。',
        '此版本只做法向局部修正，不修正横向轮廓错位；真实变形、孔边错配需额外处理。',
        '材料侧仍是 approach_outward_C 配置假设；开放 CAD 不能独立证明外侧。',
        '球心示例未验证刀具整体碰撞、可达性、标定、力控或去除深度；execution_allowed=False。',
        '复用已有刚性配准，未增加新的实体测量或精度认证；未加入机器人基坐标转换及 IK。', '',
        '## 样本状态', '', '| 状态 | 数量 |', '|---|---|']
    lines.extend(f'| {key} | {value} |' for key,value in r['status_counts'].items())
    (dest/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--mapping-config', type=Path, default=ROOT/'cad_mapping.example.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--skip-visuals', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    mcfg = json.loads(args.mapping_config.read_text())
    validate_mapping_config(mcfg)
    if not np.isfinite(mcfg['scan_voxel_mm']) or mcfg['scan_voxel_mm'] <= 0:
        raise ValueError('scan_voxel_mm must be positive')
    dest = args.run_dir.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Use a new empty run directory; previous results are preserved')
    dest.mkdir(parents=True, exist_ok=True)
    out = ROOT/cfg['output_dir']
    step_path, scan_path = ROOT/cfg['step_path'], ROOT/cfg['scan_path']
    T = np.load(out/'T_scan_to_CAD.npy')
    assert_rigid(T)
    registration = json.loads((out/'registration_quality.json').read_text())
    if not np.allclose(T, registration['selected_candidate']['T_final'], rtol=0, atol=1e-9):
        raise ValueError('Registration matrix differs from saved registration metadata')
    if cfg.get('units') != 'mm':
        raise ValueError('This pipeline requires mm')
    inputs = {'step': sha256(step_path), 'scan': sha256(scan_path),
              'transform': sha256(out/'T_scan_to_CAD.npy'), 'mesh': sha256(out/'cad_mesh_clean.ply')}
    for name, filename in [('step','cad_metadata.json'),('scan','scan_metadata.json')]:
        if inputs[name] != json.loads((out/filename).read_text())['sha256']:
            raise ValueError(f'{name} changed since registration; rerun upstream stages')
    start = time.perf_counter()
    import open3d as o3d
    raw, meta = read_ply(scan_path)
    if len(meta['elements']) != 1:
        raise ValueError('Scan must contain vertices only')
    xyz = xyz_of(raw)
    if not np.isfinite(xyz).all():
        raise ValueError('Nonfinite scan points')
    # Voxelize in saved input frame, then rigidly align; deterministic for this input.
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud = cloud.voxel_down_sample(mcfg['scan_voxel_mm'])
    scan_C = transform_points(np.asarray(cloud.points), T)
    del cloud, xyz
    timing = {'prepare_scan_seconds': time.perf_counter()-start}
    print('Scan support:', len(scan_C), 'voxel points', flush=True)
    from planning.cad_reference import build_reference_paths
    before = time.perf_counter()
    paths, catalog = build_reference_paths(step_path, scan_C, mcfg)
    timing['cad_reference_seconds'] = time.perf_counter()-before
    if not paths:
        raise ValueError('No CAD paths have adequate support; inspect selection parameters')
    reference = []
    for path in paths:
        for i,(p,n,f) in enumerate(zip(path['points_C'],path['normals_C'],path['face_ids'])):
            reference.append({'path_id':path['path_id'],'sample_index':i,'face_id':int(f),
                              **dict(zip(['x_mm','y_mm','z_mm'],p)),
                              **dict(zip(['normal_x','normal_y','normal_z'],n))})
    pd.DataFrame(reference).to_csv(dest/'cad_reference_path.csv',index=False)
    dump_json(dest/'selected_paths.json', [{'path_id':p['path_id'],'selection':p['selection']} for p in paths])
    dump_json(dest/'face_catalog.json',catalog)
    before = time.perf_counter()
    table = map_reference_paths(paths,scan_C,mcfg)
    # Always preserve rejected diagnostics, even if no complete pose can be exported.
    table.to_csv(dest/'path_mapping_diagnostics.csv',index=False)
    poses_C, poses_S, T_C, T_S = export_contact_poses(table,T,mcfg)
    keys = list(zip(poses_C.path_id,poses_C.sample_index))
    segment_lookup = dict(zip(keys,poses_C.segment_id))
    for i,row in table.iterrows():
        key = (row.path_id,row.sample_index)
        if row.valid and key not in segment_lookup:
            table.loc[i,['valid','status','segment_id']] = [False,'pose_discontinuity_isolated',-1]
            table.loc[i,['mapped_x_mm','mapped_y_mm','mapped_z_mm']] = np.nan
        elif key in segment_lookup:
            table.loc[i,'segment_id'] = segment_lookup[key]
    checks = validate_contact_poses(poses_C,poses_S,T_C,T_S,T,mcfg)
    table.to_csv(dest/'path_mapping_diagnostics.csv',index=False)
    poses_C.to_csv(dest/'tool_axis_poses_CAD.csv',index=False)
    poses_S.to_csv(dest/'tool_axis_poses_SCAN.csv',index=False)
    compact_keys = ['path_id','sample_index','segment_id','x_mm','y_mm','z_mm',
                    'tool_z_x','tool_z_y','tool_z_z','qx','qy','qz','qw',
                    'position_reference','frame','execution_allowed']
    poses_C[compact_keys].to_csv(dest/'tool_axis_poses_CAD_compact.csv',index=False)
    # Verify serialized CSVs, not just the arrays used to produce them.
    checks = validate_contact_poses(pd.read_csv(dest/'tool_axis_poses_CAD.csv'),
                                    pd.read_csv(dest/'tool_axis_poses_SCAN.csv'),T_C,T_S,T,mcfg)
    per_path_summary(paths,table,poses_C).to_csv(dest/'path_summary.csv',index=False)
    ball_C, ball_S = T_C.copy(), T_S.copy()
    ball_C[:,:3,3] = poses_C[[f'example_ball_center_{a}_mm' for a in 'xyz']]
    ball_S[:,:3,3] = poses_S[[f'example_ball_center_{a}_mm' for a in 'xyz']]
    np.savez_compressed(dest/'tool_axis_transforms.npz',T_C_contact=T_C,T_S_contact=T_S,
                        T_C_example_ball_center=ball_C,T_S_example_ball_center=ball_S,
                        path_id=poses_C.path_id.to_numpy(str),sample_index=poses_C.sample_index.to_numpy(int),
                        segment_id=poses_C.segment_id.to_numpy(int),T_scan_to_CAD=T)
    timing['mapping_and_pose_seconds'] = time.perf_counter()-before
    report = summarize(table,poses_C,checks)
    report['support_voxel_points'] = len(scan_C)
    dump_json(dest/'mapping_config.json',mcfg)
    before = time.perf_counter()
    if not args.skip_visuals:
        from planning.cad_mapping_visuals import write_mapping_visuals
        write_mapping_visuals(dest,out/'cad_mesh_clean.ply',scan_C,paths,table,mcfg,report)
    timing['visualization_seconds'] = time.perf_counter()-before
    timing['total_seconds'] = time.perf_counter()-start
    report['timing'] = timing
    write_report(dest,mcfg,report)
    dump_json(dest/'summary.json',report)
    source_paths = ['08_generate_cad_guided.py','cad_mapping.example.json',
                    'planning/scan_mapping.py','planning/cad_reference.py','planning/cad_mapping_visuals.py',
                    'pipeline_common.py','planning/geometry.py']
    manifest = {'input_sha256':inputs,'source_sha256':{p:sha256(ROOT/p) for p in source_paths},
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip(),
                'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()),
                'python':platform.python_version(),'numpy':np.__version__,
                'open3d':o3d.__version__, 'registration_reused':True,
                'scan_header':meta['header'],'robot_base_transform_included':False,
                'checks':checks}
    dump_json(dest/'manifest.json',manifest)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print('Outputs:',dest)


if __name__ == '__main__':
    main()
