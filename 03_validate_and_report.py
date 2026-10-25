"""Check delivered artifact consistency; write a Chinese report from measured values."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import json
import numpy as np
import pandas as pd
from pipeline_common import ROOT,config,dump_json,read_ply,xyz_of,transform_points,assert_rigid

def main():
    cfg=config();out=ROOT/cfg['output_dir']
    load=lambda name:json.loads((out/name).read_text())
    T=np.load(out/'T_scan_to_CAD.npy'); assert_rigid(T)
    original,_=read_ply(ROOT/cfg['scan_path']);aligned,_=read_ply(out/'scan_aligned.ply')
    assert len(original)==len(aligned)
    for field in ['intensity','event_id','camera_column']:
        assert np.array_equal(original[field],aligned[field]),f'{field} changed'
    ids=np.random.default_rng(9).choice(len(original),10000,replace=False)
    p=xyz_of(original[ids]);expected=transform_points(p,T);actual=xyz_of(aligned[ids])
    error=float(np.max(np.abs(actual-expected)))
    assert error<1e-3
    full=np.load(out/'deviation_full_resolution.npy',mmap_mode='r')
    assert len(full)==len(original)
    from pipeline_common import closest_query
    from importlib import import_module
    pipeline=import_module('02_register_and_analyze')
    scene=pipeline.get_scene(pipeline.get_mesh(out));nearest=closest_query(scene,actual)
    distances=np.linalg.norm(actual-nearest['points'],axis=1)
    np.testing.assert_allclose(full['euclidean_distance_mm'][ids],distances,atol=1e-4)
    np.testing.assert_allclose(full['signed_normal_mm'][ids],np.einsum('ij,ij->i',actual-nearest['points'],nearest['primitive_normals']),atol=1e-4)
    validations={'rigid_transform_pass':True,'det_R':float(np.linalg.det(T[:3,:3])),
        'all_original_attributes_preserved':True,'point_count_preserved':len(original),
        'transformed_coordinate_max_error_mm':error,'full_resolution_row_correspondence_pass':True,
        'synthetic_tests':'see tests.log; known box +/-2 mm offsets and known rigid transform',
        'external_metrology_validation':False}
    dump_json(out/'validation_results.json',validations)
    s=pd.read_csv(out/'deviation_summary.csv');raw=s.iloc[0];good=s.iloc[1] if len(s)>1 else None
    q=load('registration_quality.json');a=load('analysis_quality.json');m=load('mesh_quality.json')
    cm=load('cad_metadata.json');sm=load('scan_metadata.json');coverage=load('coverage_summary.json')
    selected=q['selected_candidate'];fine=selected[f"icp_{cfg['icp_correspondence_mm'][-1]:g}mm"]
    sensitivity=load('registration_sensitivity.json') if (out/'registration_sensitivity.json').exists() else None
    if sensitivity and not np.allclose(sensitivity.get('reference_transform',T),T,atol=1e-8,rtol=0):
        sensitivity=None
    sensitivity_text='未运行额外的扰动敏感性检查。'
    if sensitivity:
        full_cases=[r for r in sensitivity['cases'] if r['type']=='perturbed_full_scan']
        half_cases=[r for r in sensitivity['cases'] if r['type']=='alternating_half_scan']
        sensitivity_text=(f"4 组初始旋转分量 ±2°、平移分量 ±2 mm 的扰动重新拟合，最大点位移 P95 为 {max(r['p95_displacement_mm'] for r in full_cases):.7f} mm；"
            f"两组交替半数扫描点独立拟合，最大点位移 P95 为 {max(r['p95_displacement_mm'] for r in half_cases):.5f} mm。"+
            (f"完整 CAD 与背面 ROI 初始化的最终点位移中位数为 {sensitivity['full_vs_back_roi_median_displacement_mm']:.7f} mm。" if sensitivity.get('full_vs_back_roi_median_displacement_mm') is not None else '')+
            "这些结果仅说明本次拟合的内部局部稳定性，不是绝对精度或测量不确定度。")
    rows=[]
    for _,r in s.iterrows():
        rows.append(f"| {r['scope']} | {int(r.n_points_analyzed):,} | {r.median_signed_deviation_mm:.4f} | {r.median_absolute_deviation_mm:.4f} | {r.p95_absolute_deviation_mm:.4f} | {r.euclidean_median_mm:.4f} | {r.euclidean_p95_mm:.4f} | {100*r.fraction_abs_dev_le_example_tolerance:.2f}% |")
    table='\n'.join(rows)
    candidate_lines=[]
    for r in load('registration_candidates.json'):
        score=f"{r['surface_distance']['median_mm']:.4f}" if 'surface_distance' in r else '未进入 ICP（内点不足）'
        candidate_lines.append(f"| {r['resolution_mm']:g} | {r['final_inliers']} | {score} |")
    candidates='\n'.join(candidate_lines)
    signs='封闭、绕序一致且有正体积，外法向拓扑检查通过' if m['outward_normals_verified'] else '未通过闭合外法向验证，正负值仅表示局部三角面法向方向，不能解释为多料/缺料'
    text=f'''# CAD–扫描点云配准与偏差分析报告

本报告由实际运行结果生成。完成方案 M1–M6 及 M7 的软件数值验证；状态为**工程初步分析**。没有独立物理基准、重复扫描、扫描仪精度、正式公差及机器人标定，尚不能签署计量验收或生成可执行打磨轨迹。

## 输入与几何检查

- 工作目录：`{ROOT}`；单位：mm。
- 扫描点数：{sm['count']:,}；有限点数：{sm['finite_count']:,}；17 字节交错顶点记录，XYZ 与 intensity / event_id / camera_column 按头解析。
- 扫描文件标记为 incomplete、provisional；输出保留全部原始点的顺序和属性。
- 扫描包围盒：{sm['bbox_min_mm']} → {sm['bbox_max_mm']}。
- CAD 包围盒：{cm['bbox_min_mm']} → {cm['bbox_max_mm']}。
- CAD 实体数：{cm['solids']}；面数：{cm['faces']}；B-rep 有效性：{cm['cad_valid']}。
- CAD 声明的模型不确定度：{cm['step_declared_uncertainty_mm']} mm；这不是扫描仪精度，也不是工艺公差。
- 网格：{m['vertices']:,} 顶点 / {m['triangles']:,} 三角面；绝对线性离散参数 {cfg['mesh_linear_deflection_mm']} mm，角度参数 {cfg['mesh_angular_deflection_rad']} rad。该参数不是经认证的全局网格误差上界。
- 法向状态：{signs}。

## 配准

扫描为 source，CAD 为 target，采用 KISS-Matcher + 两级 Huber 加权 Point-to-Plane ICP。KISS 输入分别居中，返回矩阵再组合回原始坐标；禁用不适用于当前毫米参数的默认噪声边界截断。最终粗配准 CAD 区域配置：`{json.dumps(cfg['kiss_target_roi'])}`（null 表示完整 CAD）。扫描采用统计离群点剔除后用于配准，全部原始点仍保留并参与完整偏差输出。没有人为指定稳定基准，刚体最佳拟合可能吸收一部分真实形变。

| KISS 分辨率 (mm) | final inliers | ICP 后点到三角面距离中位数 (mm) |
|---|---:|---:|
{candidates}

选用分辨率 {q['selected_resolution_mm']:g} mm 的候选；精 ICP 对应阈值 {cfg['icp_correspondence_mm'][-1]:g} mm，fitness={fine['fitness']:.6f}，有效对应点 RMSE={fine['inlier_rmse_mm']:.6f} mm。该 RMSE 仅统计阈值内的点到采样点对应，不能当作整个工件的误差。

配准门槛：fitness ≥ {cfg['quality_gate_min_fitness']} 且独立查询的三角面距离中位数 ≤ {cfg['quality_gate_max_median_distance_mm']} mm。这是工程筛查门槛，未替代独立基准测量。

变换方向：`p_C = R @ p_S + t`（列向量）；NumPy 行向量等价于 `p_C = p_S @ R.T + t`。

```text
{np.array2string(T,precision=10,suppress_small=True)}
```

候选间位移一致性见 `registration_quality.json`。原始 PLY 坐标已含暂定变换，本矩阵只从当前文件坐标映射到 CAD，不等同于相机外参或机器人手眼标定。

{sensitivity_text}

## 偏差与覆盖

局部法向偏差 `d = dot(p - q, n)`；同时报告真实最近距离 `||p - q||`。在三角形边缘，`|d|` 可能小于真实距离，二者不可混用。

偏差表以 {cfg['analysis_voxel_mm']:g} mm 体素质心为单位，属于体素计数统计，既非原始扫描密度加权，也非严格的 CAD 面积加权。完整 {len(original):,} 点的 NPY 按原始行号保存。

| 统计集合 | 点数 | 法向中位数 mm | 法向绝对值中位数 mm | 法向绝对值 P95 mm | 最近距离中位数 mm | 最近距离 P95 mm | 示例 ±1 mm 内 |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

质量掩码：最近距离 ≤ {cfg['quality_max_distance_mm']} mm、切向残差 ≤ {cfg['quality_max_tangential_mm']} mm、扫描/CAD 法向绝对余弦 ≥ {cfg['quality_min_normal_cosine']}。保留 {a['quality_retained']:,}/{a['analyzed_points']:,} 个点（{a['quality_fraction']:.2%}）。**全体统计始终保留**；过滤后的值不能用于证明整体达标。三角边缘接近标记仅作检查，包含普通三角化内部边，未直接用于剔除。

全体分析点的实际最近距离 RMSE={raw.euclidean_rmse_mm:.4f} mm，P99={raw.euclidean_p99_mm:.4f} mm。少量远离 CAD 的点会显著扩大尾部统计；这类点可能来自背景、夹具、错误对应或异常扫描，不能直接视为工件形变。

CAD 均匀面积采样点中，{coverage['fraction_within_2mm']:.2%} 距扫描体素点 ≤2 mm，{coverage['fraction_within_5mm']:.2%} ≤5 mm。这是覆盖近邻代理指标，受真实偏差影响；未覆盖区没有实测偏差，不能解释为凹陷或零误差。

1 mm **仅为方案示例阈值**。主要热图使用质量筛选点的对称 P98 色阶（至少 ±1 mm），灰色表示未通过质量筛选的点；另附全体点热图，使用全体点的对称 P98 色阶。所有完整极值保留在数据文件中。未将候选正偏差直接转成去除量或加工路径。

## 验证记录与使用限制

- 变换正交性、det(R)=+1、逆向往返检查通过；输出坐标 float32 最大舍入差 {error:.7f} mm。
- 全部原始属性及点数保留检查通过；完整偏差 NPY 随机抽取 10,000 行重新查询验证通过。
- 合成盒体已知 ±2 mm 法向偏差、边缘距离差异、已知刚体变换恢复、交错字段/大小端解析、截断文件拒绝，详见 `tests.log`。
- 未获得稳定定位基准、重复扫描或已知尺寸实测点，尚未完成 M7 的物理计量校验。
- M8/M9 需要机器人模型、工具几何、手眼/装夹外参、正式余量、公差及接触工艺参数，未生成虚构的 `target_pose.csv` 或机器人轨迹。

## 文件入口

- `registration_before_after.png`、`registration_projections.png`、`registration_detail.png`：整体叠加、投影与 CAD 边界范围放大图。
- `registration_3d.html`：离线可旋转 CAD/扫描叠加。
- `deviation_heatmap_3d.html`、`deviation_heatmap_xy.png`：质量筛选偏差视图（灰色为未通过筛选点）。
- `deviation_heatmap_all_points_3d.html`、`deviation_heatmap_projections.png`、`deviation_distribution.png`：全体点偏差视图。
- `cad_coverage_3d.html`：扫描覆盖代理视图。
- `deviation_points.csv` / `.npy`：1 mm 分析点、最近点、法向、偏差与质量标记。
- `deviation_full_resolution.npy`：每个原始扫描点的法向偏差、真实距离、最近三角面索引，逐行对应原 PLY。
- `T_scan_to_CAD.txt` / `.npy`、`scan_aligned.ply`：矩阵与保留属性的完整对齐点云。
- `config.json`、`software_versions.json`、`run.log`、`validation_results.json`：运行溯源。
'''
    (out/'REPORT_zh.md').write_text(text)
    print(text)
if __name__=='__main__': main()
