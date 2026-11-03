"""Offline, paired CAD/scan views of a few robot-base contact-frame paths."""
from pathlib import Path
import html
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {'cad': '#dc860d', 'scan': '#008b76', 'x': '#cc3434',
          'y': '#17884c', 'z': '#9636b8'}


def _broken_line(points, valid, segments, sample_indices=None):
    """A segment change or invalid sample always leaves a visible gap."""
    result = []
    previous = None
    for index, (point, accepted, segment) in enumerate(zip(points, valid, segments)):
        if not accepted or not np.isfinite(point).all():
            result.append([np.nan] * 3)
            previous = None
            continue
        if previous is not None and (segment != previous or
                (sample_indices is not None and sample_indices[index] != sample_indices[index-1]+1)):
            result.append([np.nan] * 3)
        result.append(point)
        previous = segment
    return np.asarray(result, float).reshape(-1, 3)


def _surface(surface, name, color):
    vertices = np.asarray(surface['vertices'], float).reshape(-1, 3)
    triangles = np.asarray(surface['triangles'], int).reshape(-1, 3)
    return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                     i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                     color=color, opacity=.57, name=name, hoverinfo='skip',
                     flatshading=False, lighting=dict(ambient=.7, diffuse=.6,
                     roughness=.8, specular=.1), showscale=False)


def _bounds(paths):
    points = []
    for path in paths:
        p = np.asarray(path['reference'], float)
        c = np.asarray(path['contact'], float)
        points.extend([p, c[np.isfinite(c).all(axis=1)]])
    points = np.vstack(points)
    low, high = points.min(axis=0), points.max(axis=0)
    length = max(float(p['axis_length_mm']) for p in paths)
    padding = np.maximum((high - low) * .12, 1.35 * length)
    return {f'{a}axis.range': [float(lo - pad), float(hi + pad)]
            for a, lo, hi, pad in zip('xyz', low, high, padding)}


def write_path_view_html(dest, bundle):
    """Write PATH_DETAILS.html; the bundle contains already validated BASE data.

    No geometry is transformed, fitted or shifted for this display. The +Z glyph
    is drawn with its arrowhead at contact so an inward vector stays visible.
    """
    if bundle['frame'] != 'ROBOT_BASE' or bundle['units'] != 'mm':
        raise ValueError('Path detail viewer requires ROBOT_BASE coordinates in mm')
    paths = bundle['paths']
    if not paths:
        raise ValueError('At least one supported path is required')
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}] * 2],
                        horizontal_spacing=.035,
                        subplot_titles=['CAD 设计曲面与参考路径', '实测点云曲面与接触路径'])

    def add(trace, column, index, layer):
        trace.update(meta={'path': index, 'layer': layer}, showlegend=False)
        fig.add_trace(trace, row=1, col=column)

    for index, path in enumerate(paths):
        reference = np.asarray(path['reference'], float)
        contact = np.asarray(path['contact'], float)
        valid = np.asarray(path['valid'], bool)
        segments = np.asarray(path['segment_ids'], int)
        add(_surface(path['cad_surface'], 'CAD 曲面', '#bfc7d2'), 1, index, 'surface')
        add(_surface(path['scan_surface'], '实测点云局部网格', '#87b6bc'), 2, index, 'surface')
        points = np.asarray(path['scan_points'], float).reshape(-1, 3)
        add(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
            marker=dict(size=1.5, color='#407c86', opacity=.45), hoverinfo='skip',
            name='原始观测支持点', visible=False), 2, index, 'points')
        reference_line = _broken_line(reference, np.ones(len(reference), bool),
                                      np.zeros(len(reference), int), path['sample_indices'])
        for column in (1, 2):
            add(go.Scatter3d(x=reference_line[:, 0], y=reference_line[:, 1], z=reference_line[:, 2],
                mode='lines', line=dict(color=COLORS['cad'], width=6, dash='dash'),
                hoverinfo='skip', name=path['label'] + ' CAD 参考路径'), column, index, 'cad')
        broken = _broken_line(contact, valid, segments, path['sample_indices'])
        add(go.Scatter3d(x=broken[:, 0], y=broken[:, 1], z=broken[:, 2], mode='lines',
            line=dict(color=COLORS['scan'], width=8), hoverinfo='skip',
            name=path['label'] + ' 实测接触路径'), 2, index, 'scan')
        accepted = np.flatnonzero(valid)
        custom = np.column_stack([np.asarray(path['sample_indices'])[accepted],
            np.full(len(accepted), path['face_id']), np.asarray(path['offset_mm'])[accepted],
            np.asarray(path['fit_rms_mm'])[accepted],
            np.asarray(path['axes'])[accepted].transpose(0, 2, 1).reshape(-1, 9)])
        q = contact[accepted]
        add(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode='markers',
            marker=dict(size=3.2, color=COLORS['scan']), customdata=custom,
            hovertemplate=('BASE (%{x:.4f}, %{y:.4f}, %{z:.4f}) mm<br>'
            '样本 %{customdata[0]:.0f} · CAD 面 %{customdata[1]:.0f}<br>'
            '沿 CAD 法向偏移 %{customdata[2]:.4f} mm<br>局部拟合 RMS %{customdata[3]:.4f} mm<br>'
            'X (%{customdata[4]:.3f}, %{customdata[5]:.3f}, %{customdata[6]:.3f})<br>'
            'Y (%{customdata[7]:.3f}, %{customdata[8]:.3f}, %{customdata[9]:.3f})<br>'
            '+Z (%{customdata[10]:.3f}, %{customdata[11]:.3f}, %{customdata[12]:.3f})'
            '<extra>' + html.escape(path['label']) + '</extra>')), 2, index, 'scan')
        rejected = reference[~valid]
        add(go.Scatter3d(x=rejected[:, 0], y=rejected[:, 1], z=rejected[:, 2], mode='markers',
            marker=dict(symbol='x', size=5, color='#cf3557'), name='不支持或已拒绝的参考样本',
            hoverinfo='skip'), 2, index, 'rejected')
        arrow_ids = np.asarray(path['arrow_indices'], int)
        length = float(path['axis_length_mm'])
        for column, origins, axes in [(1, reference, path['cad_axes']), (2, contact, path['axes'])]:
            matrices = np.asarray(axes, float)
            chosen = arrow_ids[np.isfinite(matrices[arrow_ids]).all(axis=(1, 2))]
            for axis_index, axis in enumerate('xyz'):
                vectors = matrices[chosen, :, axis_index]
                tips = origins[chosen] if axis == 'z' else origins[chosen] + length * vectors
                tails = origins[chosen] - length * vectors if axis == 'z' else origins[chosen]
                lines = np.full((len(chosen) * 3, 3), np.nan)
                lines[::3], lines[1::3] = tails, tips
                add(go.Scatter3d(x=lines[:, 0], y=lines[:, 1], z=lines[:, 2], mode='lines',
                    line=dict(color=COLORS[axis], width=5), hoverinfo='skip',
                    name='刀轴 ' + axis.upper()), column, index, axis)
                add(go.Cone(x=tips[:, 0], y=tips[:, 1], z=tips[:, 2],
                    u=vectors[:, 0], v=vectors[:, 1], w=vectors[:, 2],
                    anchor='tip', sizemode='absolute', sizeref=length * .24,
                    colorscale=[[0, COLORS[axis]], [1, COLORS[axis]]],
                    showscale=False, hoverinfo='skip', name='箭头'), column, index, axis)

    bounds = [_bounds(paths)] + [_bounds([p]) for p in paths]
    camera = paths[0].get('camera', {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}})
    if 'eye' not in camera:
        camera = {'eye': camera}
    scene = dict(aspectmode='data', camera=camera, bgcolor='#fbfcfd',
        xaxis=dict(title='BASE X / mm', gridcolor='#e6ebf0', showspikes=False),
        yaxis=dict(title='BASE Y / mm', gridcolor='#e6ebf0', showspikes=False),
        zaxis=dict(title='BASE Z / mm', gridcolor='#e6ebf0', showspikes=False))
    fig.update_layout(template='plotly_white', height=760, scene=scene, scene2=scene,
        margin=dict(l=4, r=4, t=45, b=5), font=dict(family='Arial, sans-serif', size=12),
        hoverlabel=dict(bgcolor='white'), uirevision='contact-frame-detail')
    for key, value in bounds[0].items():
        fig.update_layout(**{f'scene.{key}': value, f'scene2.{key}': value})
    plot = fig.to_html(full_html=False, include_plotlyjs=True, div_id='detail-plot',
        config={'displaylogo': False, 'scrollZoom': True, 'responsive': True,
                'toImageButtonOptions': {'format': 'png', 'filename': 'contact_path_details', 'scale': 2}})
    cards = ''.join('<article><b>' + html.escape(p['label']) + ' · CAD 面 ' + str(int(p['face_id']))
        + '</b><small>' + html.escape(p['path_id']) + '</small><span>'
        + f"{p['stats']['accepted_samples']} / {p['stats']['total_samples']} 有效点 · "
        + f"{p['stats']['length_mm']:.1f} mm · 刀轴变化 {p['stats']['normal_sweep_deg']:.1f}°"
        + '</span></article>' for p in paths)
    options = f'<option value="-1">{len(paths)} 条路径一起看</option>' + ''.join(
        f'<option value="{i}">{html.escape(p["label"])} · 面 {int(p["face_id"])}</option>'
        for i, p in enumerate(paths))
    layers = [('surface', '曲面', True), ('cad', 'CAD 参考路径', True), ('scan', '实测接触路径', True),
              ('x', 'X 切向', True), ('y', 'Y 横向', True), ('z', '+Z 刀轴', True),
              ('points', '原始支持点', False), ('rejected', '拒绝样本', True)]
    checks = ''.join(f'<label><input type="checkbox" data-layer="{key}" '
        + ('checked' if enabled else '') + f'>{name}</label>' for key, name, enabled in layers)
    cameras = [camera] + [p.get('camera', camera) for p in paths]
    cameras = [c if 'eye' in c else {'eye': c} for c in cameras]
    title = 'CAD → 实测曲面：路径与刀轴位姿对照'
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f4f7;color:#213043;font:14px Arial,"Microsoft YaHei",sans-serif}
main{max-width:1800px;margin:auto;padding:24px 30px}h1{font-size:25px;margin:0 0 9px}p{line-height:1.7;margin:4px 0;color:#526477}
.badge{display:inline-block;background:#e2eee9;color:#176750;padding:4px 10px;border-radius:20px;font-size:12px;margin-left:12px;vertical-align:middle}
.cards{display:flex;gap:12px;margin:18px 0}.cards article{flex:1;background:white;border:1px solid #dde5ed;border-radius:9px;padding:13px 16px}
small{display:block;color:#6e7c8c;margin:6px 0;font-size:11px}article span{font-size:12px;color:#485c70}
.bar{background:white;padding:12px 16px;border:1px solid #dde5ed;border-radius:9px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
select,button{border:1px solid #bbc8d7;border-radius:5px;padding:7px 10px;background:white;color:#24394e;font-size:13px}button{cursor:pointer}
.layers{display:flex;gap:13px;flex-wrap:wrap}.layers label{cursor:pointer;white-space:nowrap;font-size:12px}.layers input{vertical-align:middle}
.plot{border:1px solid #dde5ed;border-radius:10px;background:white;margin-top:12px;overflow:hidden}.legend{display:flex;gap:22px;flex-wrap:wrap;margin:12px 2px;font-size:13px}
.legend i{display:inline-block;width:22px;border-top:3px solid;vertical-align:middle;margin-right:6px}.footer{font-size:12px;margin-top:8px}
@media(max-width:850px){main{padding:15px 10px}.cards{flex-direction:column}.bar{gap:10px}h1{font-size:21px}}
</style><main><h1>__TITLE__<span class="badge">ROBOT_BASE · mm</span></h1>
<p>左侧看 CAD 设计位置与姿态，右侧看实测接触位置与姿态。拖动旋转、滚轮缩放；两侧相机同步。选择单条路径可放大查看完整三轴。</p>
<section class="cards">__CARDS__</section><div class="bar"><label>路径 <select id="path">__OPTIONS__</select></label>
<button id="reset">恢复推荐视角</button><label>曲面透明度 <input id="opacity" type="range" min="0" max="1" step="0.05" value="0.57"></label>
<div class="layers">__CHECKS__</div></div><section class="plot">__PLOT__</section>
<div class="legend"><span><i style="border-color:#dc860d;border-top-style:dashed"></i>CAD 参考路径</span><span><i style="border-color:#008b76"></i>实测接触路径</span>
<span><i style="border-color:#cc3434"></i>X 沿路径切向</span><span><i style="border-color:#17884c"></i>Y 右手系横轴</span><span><i style="border-color:#9636b8"></i>+Z 朝向表面</span></div>
<p id="scale" class="footer"></p><p class="footer">接触点与参考路径保持实际尺度，未放大两者偏差。紫色 +Z 箭头从表面外侧指向接触点，以免埋入曲面；方向等于导出位姿的 +Z。实测曲面为观测点的局部显示网格，缺口保持断开。</p>
</main><script>
const gd=document.getElementById('detail-plot'), boxes=[...document.querySelectorAll('[data-layer]')];
const bounds=__BOUNDS__, cameras=__CAMERAS__, lengths=__LENGTHS__;
let syncing=false;
function active(){return Number(document.getElementById('path').value)}
function view(){const n=active()+1, update={};for(const s of ['scene','scene2']){for(const [key,val] of Object.entries(bounds[n]))update[s+'.'+key]=val;update[s+'.camera']=cameras[n]}
 Plotly.relayout(gd,update);document.getElementById('scale').textContent='三轴箭头显示长度：'+(active()<0?lengths.map((v,i)=>'P'+(i+1)+' '+v.toFixed(1)+' mm').join('；'):lengths[active()].toFixed(1)+' mm')+'。此长度仅用于辨认方向。';}
function layers(){const enabled=Object.fromEntries(boxes.map(b=>[b.dataset.layer,b.checked]));const n=active();
 Plotly.restyle(gd,{visible:gd.data.map(t=>(n<0||t.meta.path===n)&&!!enabled[t.meta.layer])});}
document.getElementById('path').addEventListener('change',()=>{layers();view()});
document.getElementById('reset').addEventListener('click',view);boxes.forEach(b=>b.addEventListener('change',layers));
document.getElementById('opacity').addEventListener('input',e=>{const ids=gd.data.map((t,i)=>t.meta.layer==='surface'?i:-1).filter(i=>i>=0);Plotly.restyle(gd,{opacity:Number(e.target.value)},ids)});
gd.on('plotly_relayout',event=>{if(syncing)return;let source=null,cam=null;for(const s of ['scene','scene2'])if(event[s+'.camera']){source=s;cam=event[s+'.camera'];break}
 if(!cam)return;syncing=true;Plotly.relayout(gd,{[(source==='scene'?'scene2':'scene')+'.camera']:cam}).then(()=>{syncing=false})});
layers();view();
</script></html>'''
    for marker, value in {'TITLE': title, 'CARDS': cards, 'OPTIONS': options, 'CHECKS': checks,
        'PLOT': plot, 'BOUNDS': json.dumps(bounds), 'CAMERAS': json.dumps(cameras),
        'LENGTHS': json.dumps([float(p['axis_length_mm']) for p in paths])}.items():
        page = page.replace('__' + marker + '__', value)
    result = Path(dest) / 'PATH_DETAILS.html'
    result.write_text(page, encoding='utf-8')
    return result
