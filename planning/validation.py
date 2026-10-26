"""M7 numerical checks and explicit separation from physical measurement validation."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from pipeline_common import assert_rigid, sha256, dump_json


def numerical_sign_check():
    import open3d as o3d
    from pipeline_common import closest_query
    mesh = o3d.geometry.TriangleMesh.create_box(10, 10, 10)
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    offset = np.array([-.999, -.2, 0., .2, .999], dtype=np.float32)
    points = np.column_stack([np.full(5, 5), np.full(5, 5), 10+offset]).astype(np.float32)
    result = closest_query(scene, points)
    delta = points-result['points']
    signed = np.einsum('ij,ij->i', delta, result['primitive_normals'])
    error = np.max(np.abs(signed-offset))
    return {'passed': bool(error < 2e-6), 'max_error_mm': float(error),
            'expected_mm': offset.tolist(), 'measured_mm': signed.tolist(),
            'scope': 'Synthetic known outward box normal; not actual material-side validation'}


def physical_gate(material_confirmed, independent_accuracy_verified, formal_process_tolerance):
    reasons = []
    if not material_confirmed:
        reasons.append('CAD material-side normals not independently confirmed')
    if not independent_accuracy_verified:
        reasons.append('Registration accuracy has no independent physical reference')
    if formal_process_tolerance is None:
        reasons.append('Deburring allowance/process tolerance is unspecified')
    return {'passed': not reasons, 'pending': reasons}


def exact_brep_check(step_path, frame, sample_count):
    """Compare sampled scan-to-mesh distances with OCCT trimmed B-rep distances.

    Conservative face AABBs prune only faces whose lower bound exceeds a known
    exact face distance. No nearest-triangle/STEP-face correspondence assumption.
    """
    import cadquery as cq
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    wp = cq.importers.importStep(str(step_path))
    faces = [f for shape in wp.vals() for f in shape.Faces()]
    bounds = []
    for face in faces:
        b = face.BoundingBox()
        bounds.append([b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax])
    bounds = np.asarray(bounds)
    # Spread deterministically across residual ranks and surface directions.
    eligible = frame[frame.quality_mask].sort_values('euclidean_distance_mm')
    selected = eligible.iloc[np.linspace(0, len(eligible)-1, min(sample_count, len(eligible))).astype(int)]
    rows = []
    for index, row in selected.iterrows():
        p = row[['scan_x_mm', 'scan_y_mm', 'scan_z_mm']].to_numpy(float)
        lower = np.linalg.norm(np.maximum(np.maximum(bounds[:, :3]-p, p-bounds[:, 3:]), 0), axis=1)
        vertex = cq.Vertex.makeVertex(*map(float, p))
        best, calls = np.inf, 0
        for face_index in np.argsort(lower):
            if lower[face_index] > best + 1e-5:
                break
            calc = BRepExtrema_DistShapeShape(vertex.wrapped, faces[int(face_index)].wrapped)
            if not calc.IsDone():
                raise RuntimeError('OCCT face-distance solver did not finish')
            best = min(best, calc.Value())
            calls += 1
        mesh_distance = float(row.euclidean_distance_mm)
        rows.append({'analysis_row': int(index), 'mesh_distance_mm': mesh_distance,
                     'brep_distance_mm': best, 'absolute_difference_mm': abs(best-mesh_distance),
                     'exact_face_queries': calls})
        print('B-rep check', len(rows), '/', len(selected), rows[-1], flush=True)
    return rows


def source_manifest(root):
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    # Include current new implementation as well, recording its content even before commit.
    files = set(p for p in tracked if p) | {str(p.relative_to(root)) for p in (root/'planning').glob('*.py')}
    files |= {p.name for p in root.glob('0[67]_*.py')} | {'planning.example.json'}
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
            'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip()),
            'source_sha256': {p: sha256(root/p) for p in sorted(files) if (root/p).is_file()}}


def run_m7(root, cfg, pcfg, run_dir):
    dest = run_dir/'m7'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'validation.json').exists():
        raise FileExistsError('Use a new run directory to preserve the previous M7 record')
    out = root/cfg['output_dir']
    frame = pd.read_csv(out/'deviation_points.csv')
    transform = np.load(out/'T_scan_to_CAD.npy')
    assert_rigid(transform)
    mesh_info = json.loads((out/'mesh_quality.json').read_text())
    registration = json.loads((out/'registration_quality.json').read_text())
    if not np.allclose(transform, registration['selected_candidate']['T_final'], atol=1e-9, rtol=0):
        raise ValueError('Saved transform and registration metadata differ')
    delta = frame[['scan_x_mm', 'scan_y_mm', 'scan_z_mm']].to_numpy()-frame[['cad_closest_x_mm', 'cad_closest_y_mm', 'cad_closest_z_mm']].to_numpy()
    normals = frame[['normal_x', 'normal_y', 'normal_z']].to_numpy()
    normal_error = float(np.max(np.abs(np.einsum('ij,ij->i', delta, normals)-frame.signed_deviation_mm)))
    distance_error = float(np.max(np.abs(np.linalg.norm(delta, axis=1)-frame.euclidean_distance_mm)))
    unit_error = float(np.max(np.abs(np.linalg.norm(normals, axis=1)-1)))
    numerical = numerical_sign_check()
    exact = exact_brep_check(root/cfg['step_path'], frame, pcfg['brep_samples'])
    pd.DataFrame(exact).to_csv(dest/'mesh_vs_step.csv', index=False)
    max_diff = max(r['absolute_difference_mm'] for r in exact)
    table = []
    for scope, mask in [('all', np.ones(len(frame), bool)), ('quality_filtered', frame.quality_mask.to_numpy(bool))]:
        distances = frame.loc[mask, 'euclidean_distance_mm'].to_numpy()
        for threshold in [.2, .3, .5, 1.]:
            table.append({'scope': scope, 'threshold_mm': threshold, 'fraction_below': float(np.mean(distances < threshold)),
                          'n': len(distances), 'meaning': 'Surface fit only; not registration accuracy'})
    pd.DataFrame(table).to_csv(dest/'threshold_sensitivity.csv', index=False)
    physical = physical_gate(pcfg['material_side_confirmed'], False, None)
    checks = {'known_sign_and_magnitude': numerical,
              'normal_unit_length': {'passed': unit_error < 1e-5, 'max_error': unit_error},
              'saved_signed_distance': {'passed': normal_error < .001, 'max_error_mm': normal_error},
              'saved_euclidean_distance': {'passed': distance_error < .001, 'max_error_mm': distance_error},
              'mesh_vs_original_step': {'passed': max_diff <= pcfg['mesh_brep_max_difference_mm'],
                                       'samples': len(exact), 'max_difference_mm': max_diff,
                                       'threshold_mm': pcfg['mesh_brep_max_difference_mm'],
                                       'scope': 'Sampled geometry representation agreement; not full-model certification'}}
    report = {'numerical_passed': all(c['passed'] for c in checks.values()), 'checks': checks,
              'physical_gate': physical, 'material_side_confirmed': pcfg['material_side_confirmed'],
              'mesh_watertight': mesh_info['watertight'], 'registration_target_mm': pcfg['registration_target_mm'],
              'registration_accuracy_verified': False, 'formal_removal_depth_mm': None,
              'registration_target_is_not_removal_depth': True,
              'source': source_manifest(root),
              'input_sha256': {name: sha256(out/name) for name in ['deviation_points.csv', 'T_scan_to_CAD.npy', 'cad_mesh_clean.ply']},
              'step_sha256': sha256(root/cfg['step_path']),
              'status': 'numerical_complete_physical_pending'}
    dump_json(dest/'validation.json', report)
    dump_json(run_dir/'planning_config.json', pcfg)
    lines = ['# M7 验证记录', '', f"数值验证：{'通过' if report['numerical_passed'] else '失败'}。实际材料侧、独立配准精度和正式加工余量仍待确认。", '',
             f"配准目标：{pcfg['registration_target_mm']} mm；这不是磨削深度，也不自动成为毛刺判据。", '',
             '| 检查 | 结果 |', '|---|---|']
    lines += [f"| {name} | {'PASS' if value['passed'] else 'FAIL'} |" for name, value in checks.items()]
    lines += ['', f"原 STEP 曲面抽查 {len(exact)} 点，网格距离与 B-rep 距离最大差 {max_diff:.6f} mm。",
              '本项验证几何表示的一致性，不能证明扫描仪或配准的绝对精度。', '',
              '合成几何可以验证正负号的计算；当前开放 CAD 的材料外侧无法仅凭这些文件独立证明。',
              'M8 允许生成明确标注假设的几何预览，禁止把本记录标为实机加工验收通过。']
    (dest/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    return report
