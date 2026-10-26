"""Offline review artifacts. Plotly is embedded; geometry remains local."""
import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pipeline_common import read_ply, xyz_of


def write_deliverables(out, run_dir, classified, poses, strokes, manifest):
    dest = run_dir/'m8'
    records, _ = read_ply(out/'cad_surface_sample.ply')
    ids = np.random.default_rng(7).choice(len(records), min(16000, len(records)), replace=False)
    cad = xyz_of(records[ids])
    candidate = classified[classified.candidate_mask]
    paths = [s['center'] for s in strokes]
    contacts = [s['contact'] for s in strokes]
    def joined(arrays):
        return np.concatenate([np.vstack([p, np.full((1, 3), np.nan)]) for p in arrays])
    line = joined(paths)
    touch = joined(contacts)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=cad[:, 0], y=cad[:, 1], z=cad[:, 2], mode='markers', name='CAD reference',
                              marker=dict(size=1, color='#8394a6', opacity=.22)))
    cp = candidate[['scan_x_mm','scan_y_mm','scan_z_mm']].to_numpy()
    fig.add_trace(go.Scatter3d(x=cp[:, 0], y=cp[:, 1], z=cp[:, 2], mode='markers', name='Candidate protrusions (unconfirmed)',
                              marker=dict(size=2.1, color='#df7549', opacity=.65)))
    fig.add_trace(go.Scatter3d(x=touch[:, 0], y=touch[:, 1], z=touch[:, 2], mode='lines', name='Nominal CAD contact curves',
                              line=dict(width=3, color='#2d9e77')))
    fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines', name='Ball-center contact strokes',
                              line=dict(width=5, color='#2166ac')))
    fig.add_trace(go.Scatter3d(x=poses.x_mm, y=poses.y_mm, z=poses.z_mm, mode='lines', name='Full order / lifts / transfers',
                              visible='legendonly', line=dict(width=2, color='#aaa', dash='dash')))
    cfg = manifest['config']
    fig.update_layout(title=f"M8 CAD-frame path preview — {len(strokes)} strokes; assumed Ø{cfg['tool_ball_diameter_mm']:g} mm ball tool",
                      template='plotly_white', height=820, margin=dict(l=0,r=0,b=0,t=60),
                      scene=dict(aspectmode='data', xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                                 camera=dict(eye=dict(x=1.2,y=-1.5,z=-1.5))), legend=dict(x=0,y=1))
    intro = f'''<div style="font:16px/1.65 sans-serif;max-width:1100px;margin:24px auto;padding:20px;background:#f5f7fa">
<h1>M7–M8：候选去毛刺路径预览</h1>
<p>已生成 {manifest['regions_with_paths']} 个区域、{manifest['strokes']} 段接触路径，接触路径总长 {manifest['contact_path_length_mm']:.1f} mm。
蓝线是球心轨迹，绿线是 CAD 名义接触点，橙点是疑似凸起。点击图例可显示抬刀与转场；拖动旋转、滚轮缩放。</p>
<p><b>这是 CAD 坐标系的几何预览，不能发送给机器人。</b>材料外侧、磨头和标定尚未确认。球头直径 {cfg['tool_ball_diameter_mm']} mm、行距 {cfg['raster_spacing_mm']} mm、采样 {cfg['path_sampling_mm']} mm 均为可修改的预览假设。
配准目标 {cfg['registration_target_mm']} mm 与候选阈值 {cfg['candidate_threshold_mm']} mm 分开记录；后者不代表已验证毛刺尺寸。没有指定实际去除深度、进给或接触力。</p>
</div>'''
    page = fig.to_html(include_plotlyjs=True, full_html=True)
    page = page.replace('<body>', '<body>'+intro, 1)
    (dest/'toolpath_preview.html').write_text(page)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig2, axes = plt.subplots(1, 3, figsize=(17, 6), layout='constrained')
    for ax, (i,j,name) in zip(axes, [(0,1,'XY'), (0,2,'XZ'), (1,2,'YZ')]):
        ax.scatter(cad[:,i],cad[:,j],s=.4,c='#a5b1bc',alpha=.3)
        ax.scatter(cp[:,i],cp[:,j],s=1.6,c='#df7549',alpha=.65,label='Candidate protrusions')
        for index, points in enumerate(paths):
            ax.plot(points[:,i],points[:,j],lw=.8,c='#2166ac',label='Ball-center contact path' if index==0 else None)
        ax.set(title=name,xlabel='XYZ'[i]+' (mm)',ylabel='XYZ'[j]+' (mm)')
        ax.set_xlim(cad[:,i].min()-8,cad[:,i].max()+8);ax.set_ylim(cad[:,j].min()-8,cad[:,j].max()+8)
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig2.suptitle('M8 offline geometry preview — provisional tool and material side; no robot execution',fontsize=13)
    fig2.savefig(dest/'toolpath_overview.png',dpi=160);plt.close(fig2)
    # Show the four largest regions with accepted paths; avoid a misleading dense global view.
    region_ids = sorted(set(s['region_id'] for s in strokes))[:4]
    fig3,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
    for ax,rid in zip(axes.flat,region_ids):
        region=classified[classified.region_id==rid]
        ax.scatter(region.cad_closest_x_mm,region.cad_closest_y_mm,s=15,c=region.local_excess_mm,cmap='YlOrRd',label='Local candidate samples')
        for s in strokes:
            if s['region_id']==rid:
                ax.plot(s['center'][:,0],s['center'][:,1],c='#2166ac',lw=1)
        center=region[['cad_closest_x_mm','cad_closest_y_mm']].median().to_numpy()
        ax.set(title=f'Region {rid}: 20 mm XY window',xlabel='X (mm)',ylabel='Y (mm)',
               xlim=(center[0]-10,center[0]+10),ylim=(center[1]-10,center[1]+10))
        ax.set_aspect('equal');ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(region_ids):]:ax.set_visible(False)
    fig3.suptitle('Candidate-region paths — orange: local protrusion; blue: ball center')
    fig3.savefig(dest/'toolpath_details.png',dpi=160);plt.close(fig3)
    summary = f'''# M7–M8 路径生成交付

## 当前结论

M7 数值检查通过；实际材料侧、独立配准精度和正式工艺余量尚未验证。M8 已生成可查看、可导出的 CAD 坐标系几何预览。

- 候选区域：{manifest['candidate_regions']} 个；有可用接触路径的区域：{manifest['regions_with_paths']} 个。
- 接触段：{manifest['strokes']} 段，名义球心接触路径总长 {manifest['contact_path_length_mm']:.3f} mm。
- 接触采样点：{manifest['contact_waypoints']}；含抬刀和转场的总位姿：{manifest['total_waypoints']}。
- 区域内候选采样点距接触路径采样点 1 mm 内的比例：{100*manifest['candidate_fraction_near_contact_path_1mm']:.2f}%。这只是点邻近比例，不是实际磨削覆盖率。

## 预览假设

球形工具直径 {cfg['tool_ball_diameter_mm']} mm、模型长度 {cfg['tool_length_mm']} mm；行距 {cfg['raster_spacing_mm']} mm、沿线采样 {cfg['path_sampling_mm']} mm。
接近外侧方向为 CAD 的 {cfg['approach_outward_C']}，基于本次扫描主要覆盖的一侧作可视化假设，不是已经验证的材料外侧。只规划这一侧、方向合适且有可信扫描对应的局部区域。

候选点同时满足法向正偏差 >{cfg['candidate_threshold_mm']} mm、局部突出量 ≥{cfg['local_prominence_mm']} mm，并有足够邻域与连续点支持。这些经验阈值用于生成待审查候选，不是对毛刺的真值判定。
配准目标 {cfg['registration_target_mm']} mm 并未用于设置磨削深度。当前空间区域敏感性与 1 mm 体素分辨率都限制细小毛刺的可靠识别；也不能把不生成路径的区域视为无毛刺。

## 生成方法

1. 保留每个分析点及其分类原因，排除不可靠对应、其他接近方向、宽区域偏移和孤立点。
2. 在三维空间按距离与法向连续性分区；投影区域由局部支持范围构造，保留孔洞与不连续区。
3. 沿每区主方向排列并用 Shapely 裁剪往复线，用 OpenCAMLib BallCutter 对原 CAD 三角面做 drop-cutter，求出球头接触位置与球心轨迹。
4. 检查接触仍对应原候选区域；在每个采样点和相邻中点检查球体到 CAD 的间隙，拒绝不连续高度和不满足条件的线段。
5. 各接触段之间先沿工具外侧抬起，再在高于全部 CAD（沿假设接近方向）的平面转场。顺序采用最近端点启发式，可反向，并非全局最优。

这不是完整连续碰撞证明；没有校验实际磨头杆、主轴、夹具、机器人、动态误差或原始毛刺实体。转场平面仅相对于 CAD 几何有间隙。
离散采样和直线插补中点采用 {cfg['geometry_tolerance_mm']} mm 的几何容差；允许的微小穿入也可能来自曲面弦线近似，不能全部解释为浮点误差。加工前需按正式工具与允许误差进一步加密、验证连续路径。

## 文件与字段

- `toolpath_preview.html`：可离线旋转查看；默认隐藏完整转场线，点击图例显示。
- `toolpath_overview.png` / `toolpath_details.png`：全局与区域放大图。
- `target_pose.csv`：路径顺序、球心 XYZ、四元数 XYZW、接触点、法向、区域/段编号及阶段。
- `candidate_regions.csv`：完整分析点、候选标签、区域、局部突出量与不采用原因。`analysis_row` 对应原 `deviation_points.csv` 行，不是原始 660 万点的行号。
- `region_planning.json` / `path_manifest.json`：每区接受/拒绝数量、参数、代码版本和输入哈希。
- `candidate_path_coverage.csv`：每个保留候选点到接触路径采样点的距离，明确保留尚未覆盖的候选。
- `artifact_checks.json`：有限坐标、四元数、步长、球心/接触位置一致性及禁止执行标记的检查。
- `../m7/validation.json` / `mesh_vs_step.csv`：M7 数值与原始 STEP 对照结果。

TCP 在这里定义为球心，不是机器人法兰；工具 +Z 指向工件。路径落在名义 CAD 接触包络上，未推导实际多道次去除过程。
`removal_depth_mm`、`feed_mm_s`、`force_N` 留空，所有行 `execution_allowed=False`。因此这是一份待工具/方向确认的几何路径，不是实机程序。
'''
    (dest/'REPORT_zh.md').write_text(summary)
