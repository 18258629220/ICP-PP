"""Robot-base scan segmentation and contact paths, without a tool model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from pipeline_common import assert_rigid, transform_points, dump_json


def _color(face_id):
    import colorsys
    return np.array(colorsys.hsv_to_rgb((int(face_id)*.61803398875)%1., .68, .88))


def _write_labeled_cloud(path, raw, labels, selected_faces):
    """Keep original robot-base XYZ and every scalar in source row order."""
    original = raw.dtype
    reserved = {'cad_face_id','planned_surface','red','green','blue'}
    if reserved.intersection(original.names):
        raise ValueError('Input already has label/color fields; refuse ambiguous overwrite')
    dtype = np.dtype([(name,original.fields[name][0].newbyteorder('<')) for name in original.names]
                     +[(name,'<i4') for name in ['cad_face_id']]
                     +[('planned_surface','u1'),('red','u1'),('green','u1'),('blue','u1')])
    ply_types = {('f',4):'float',('f',8):'double',('i',1):'char',('i',2):'short',
                 ('i',4):'int',('u',1):'uchar',('u',2):'ushort',('u',4):'uint'}
    header = ['ply','format binary_little_endian 1.0',
              'comment frame ROBOT_BASE; original scan XYZ unchanged; units mm',
              'comment Source row order and original scalar attributes preserved',
              f'element vertex {len(raw)}']
    for name in dtype.names:
        field = dtype.fields[name][0]
        header.append(f'property {ply_types[(field.kind,field.itemsize)]} {name}')
    with Path(path).open('wb') as stream:
        stream.write(('\n'.join(header+['end_header'])+'\n').encode('ascii'))
        for start in range(0,len(raw),200000):
            stop = min(start+200000,len(raw))
            block = np.empty(stop-start,dtype=dtype)
            for name in original.names:
                block[name] = raw[name][start:stop]
            face = np.asarray(labels[start:stop])
            block['cad_face_id'] = face
            block['planned_surface'] = np.isin(face,selected_faces)
            rgb = np.full((len(face),3), [150,158,167], dtype=np.uint8)
            rgb[face < 0] = [210,100,70]
            for fid in np.intersect1d(np.unique(face), selected_faces):
                rgb[face == fid] = np.round(255*_color(fid)).astype(np.uint8)
            for j,name in enumerate(['red','green','blue']):
                block[name] = rgb[:,j]
            block.tofile(stream)


def _stroke_lines(poses, maximum=150000):
    """Retain segment breaks; thin display only if necessary."""
    if poses is None or not len(poses):
        return np.empty((0,3))
    stride = max(1,int(np.ceil(len(poses)/maximum)))
    parts = []
    for _, group in poses.groupby('segment_id',sort=False):
        p = group[['x_mm','y_mm','z_mm']].to_numpy(float)
        ids = np.unique(np.r_[np.arange(0,len(p),stride),len(p)-1])
        parts.extend([p[ids], np.full((1,3),np.nan)])
    return np.vstack(parts)


def _validate_base_poses(poses, dest):
    """Reject mixed frames and malformed axes before drawing a plausible plot."""
    if poses is None:
        raise ValueError('ROBOT_BASE contact pose table is required')
    if 'frame' not in poses or not poses.frame.eq('ROBOT_BASE').all():
        raise ValueError('Presentation requires poses.frame=ROBOT_BASE; convert once before calling')
    keys = ['x_mm','y_mm','z_mm','qx','qy','qz','qw']
    axes = [f'tool_{axis}_{component}' for axis in 'xyz' for component in 'xyz']
    if not set(keys+axes).issubset(poses.columns):
        raise ValueError('Contact pose position, axes and XYZW quaternion columns are required')
    values = poses[keys+axes].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite robot-base contact poses')
    if len(poses):
        matrices = np.stack([poses[[f'tool_{axis}_{c}' for c in 'xyz']].to_numpy(float)
                             for axis in 'xyz'],axis=2)
        np.testing.assert_allclose(np.swapaxes(matrices,1,2)@matrices,
                                   np.broadcast_to(np.eye(3),matrices.shape),atol=1e-8,rtol=0)
        np.testing.assert_allclose(np.linalg.det(matrices),1.,atol=1e-8,rtol=0)
        q = poses[['qx','qy','qz','qw']].to_numpy(float)
        np.testing.assert_allclose(np.linalg.norm(q,axis=1),1.,atol=1e-8,rtol=0)
        np.testing.assert_allclose(Rotation.from_quat(q).as_matrix(),matrices,atol=1e-8,rtol=0)
    exported = Path(dest)/'contact_poses_BASE.csv'
    if exported.exists():
        table = pd.read_csv(exported)
        if len(table) != len(poses) or 'frame' not in table or not table.frame.eq('ROBOT_BASE').all():
            raise ValueError('Existing BASE CSV disagrees with presentation poses')
        np.testing.assert_allclose(table[keys+axes],values,atol=1e-8,rtol=0)


def _base_view(poses):
    """Look toward observed surfaces; this changes only the camera."""
    if len(poses):
        outward = -poses[[f'tool_z_{a}' for a in 'xyz']].to_numpy(float).mean(axis=0)
    else:
        outward = np.array([0.,0.,1.])
    if np.linalg.norm(outward) < 1e-8:
        outward = np.array([0.,0.,1.])
    outward /= np.linalg.norm(outward)
    side = np.cross(outward,[0.,0.,1.])
    if np.linalg.norm(side) < 1e-8:
        side = np.array([1.,0.,0.])
    side /= np.linalg.norm(side)
    across = np.cross(outward,side)
    eye = 1.7*outward+.7*side+.4*across
    elev = float(np.clip(np.degrees(np.arcsin(eye[2]/np.linalg.norm(eye))),-60.,60.))
    azim = float(np.degrees(np.arctan2(eye[1],eye[0])))
    return eye,elev,azim


def write_real_deliverables(dest, raw, transform, topology, label_dir, poses, coverage, report,
                            *, output_frame='ROBOT_BASE'):
    """Write a robot-base view; poses must already be converted exactly once.

    ``transform`` remains the legacy base-to-CAD registration. Only CAD overlay
    vertices use its inverse. Scan XYZ are the original user-confirmed base
    coordinates. Contact CSV/matrix export is owned by the calling workflow.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import plotly.graph_objects as go
    dest, label_dir = Path(dest), Path(label_dir)
    if output_frame != 'ROBOT_BASE' or report.get('output_frame','ROBOT_BASE') != 'ROBOT_BASE':
        raise ValueError('Real deliverables support only ROBOT_BASE output')
    if report.get('units','mm') != 'mm':
        raise ValueError('Robot-base presentation requires mm')
    assert_rigid(transform)
    _validate_base_poses(poses,dest)
    labels = np.load(label_dir/'face_labels.npy',mmap_mode='r')
    if labels.shape != (len(raw),) or not np.issubdtype(labels.dtype,np.integer):
        raise ValueError('Face labels must align with every original scan row')
    if np.any(labels < -1):
        raise ValueError('Incomplete face labels cannot be presented as a completed scan')
    selected = np.array(sorted(poses.face_id.unique()),int) if poses is not None and len(poses) else np.array([],int)
    _write_labeled_cloud(dest/'scan_segmented_BASE.ply', raw, labels, selected)
    ids = np.linspace(0,len(raw)-1,min(90000,len(raw))).astype(np.int64)
    points = np.column_stack([raw[a][ids] for a in 'xyz']).astype(float)
    faces = np.asarray(labels[ids])
    strokes = _stroke_lines(poses)
    valid = faces >= 0
    planned = np.isin(faces,selected)
    figure = go.Figure()
    nominal = transform_points(topology.vertices[np.linspace(0,len(topology.vertices)-1,
        min(50000,len(topology.vertices))).astype(int)],np.linalg.inv(transform))
    figure.add_trace(go.Scatter3d(x=nominal[:,0],y=nominal[:,1],z=nominal[:,2],mode='markers',
        name='CAD 定位到机器人基坐标（显示抽样）',visible='legendonly',
        marker=dict(size=1.,color='#8b919a',opacity=.3),hoverinfo='skip'))
    for mask,name,color,size in [
        (valid & ~planned,'其余已归属型面','#a4adb8',1.4),
        (~valid,'边界、未匹配或不确定点','#d26446',1.6)]:
        q=points[mask]
        figure.add_trace(go.Scatter3d(x=q[:,0],y=q[:,1],z=q[:,2],mode='markers',name=name,
            marker=dict(size=size,color=color,opacity=.45),hoverinfo='skip',
            visible='legendonly' if name.startswith('边界') else True))
    q=points[planned]
    colors = ['rgb('+','.join(map(str,np.round(255*_color(i)).astype(int)))+')' for i in faces[planned]]
    figure.add_trace(go.Scatter3d(x=q[:,0],y=q[:,1],z=q[:,2],mode='markers',name='规划型面（按CAD面分色）',
        marker=dict(size=1.9,color=colors,opacity=.85),customdata=faces[planned],
        hovertemplate='CAD面 %{customdata}<br>BASE (%{x:.2f}, %{y:.2f}, %{z:.2f}) mm<extra></extra>'))
    figure.add_trace(go.Scatter3d(x=strokes[:,0],y=strokes[:,1],z=strokes[:,2],mode='lines',name='随形接触路径',
        line=dict(color='#142fa0',width=4),hoverinfo='skip'))
    if poses is not None and len(poses):
        arrow_rows=poses.iloc[np.linspace(0,len(poses)-1,min(70,len(poses))).astype(int)]
        p=arrow_rows[['x_mm','y_mm','z_mm']].to_numpy(float)
        z=arrow_rows[[f'tool_z_{a}' for a in 'xyz']].to_numpy(float)
        arrow = np.full((len(p)*3,3),np.nan);arrow[::3]=p-4*z;arrow[1::3]=p
        figure.add_trace(go.Scatter3d(x=arrow[:,0],y=arrow[:,1],z=arrow[:,2],mode='lines',
            name='刀轴 +Z 指向接触点',line=dict(color='#9b1c8c',width=6),hoverinfo='skip'))
        figure.add_trace(go.Cone(x=p[:,0],y=p[:,1],z=p[:,2],u=z[:,0],v=z[:,1],w=z[:,2],
            anchor='tip',sizemode='absolute',sizeref=1.2,showscale=False,
            colorscale=[[0,'#9b1c8c'],[1,'#9b1c8c']],name='刀轴方向箭头',showlegend=False,hoverinfo='skip'))
    eye,elev,azim = _base_view(poses)
    figure.update_layout(title='真实工件：机器人基坐标系下的型面分割与随形接触路径<br><sup>分色=型面；蓝线=接触路径；紫箭头=刀轴。全部坐标为 ROBOT_BASE，单位 mm。</sup>',
        template='plotly_white',height=900,scene=dict(aspectmode='data',xaxis_title='BASE X / mm',
        yaxis_title='BASE Y / mm',zaxis_title='BASE Z / mm',camera=dict(eye=dict(zip('xyz',eye)))),
        margin=dict(l=0,r=0,b=0,t=90))
    figure.write_html(dest/'REAL_WORKPIECE.html',include_plotlyjs=True,config={'displaylogo':False,'scrollZoom':True})
    registration = go.Figure(figure)
    for index,trace in enumerate(registration.data):
        trace.visible = True if index == 0 else 'legendonly'
    registration.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],
        mode='markers',name='原始扫描：机器人基坐标，XYZ未变换',
        marker=dict(size=1.5,color='#1675b7',opacity=.5),hoverinfo='skip'))
    registration.update_layout(title='机器人基坐标系中的CAD定位结果<br><sup>蓝点=原始基坐标扫描；灰色=由配准逆变换定位的CAD。扫描点保持不动。</sup>')
    registration.write_html(dest/'registration_BASE.html',include_plotlyjs=True,
                            config={'displaylogo':False,'scrollZoom':True})

    fig=plt.figure(figsize=(16,8),layout='constrained')
    center_points=points[valid] if valid.any() else points
    low,high=np.quantile(center_points,[.001,.999],axis=0); span=np.maximum(high-low,1.);mid=(low+high)/2
    # Display only: all source rows, including unassigned outliers, remain in PLY.
    in_view=valid & np.all((points >= low-.03*span) & (points <= high+.03*span),axis=1)
    displayed=np.flatnonzero(in_view)[::2]
    for number in [1,2]:
        ax=fig.add_subplot(1,2,number,projection='3d')
        show=displayed
        colors=np.tile([.65,.68,.72],(len(show),1))
        for fid in selected:
            colors[faces[show] == fid]=_color(fid)
        ax.scatter(*points[show].T,s=.4,c=colors,alpha=.55,depthshade=False,rasterized=True)
        if number==2 and len(strokes):
            ax.plot(*strokes.T,color='#102b91',lw=.65)
        for setter,c,d in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid,span):setter(c-.54*d,c+.54*d)
        # Matplotlib normalizes its aspect array in place; keep shared limits.
        ax.set_box_aspect(span.copy(),zoom=.85);ax.view_init(elev=elev,azim=azim)
        ax.set(xlabel='BASE X / mm',ylabel='BASE Y / mm',zlabel='BASE Z / mm',
               title='Robot-base scan — face segmentation' if number==1 else 'Robot-base surface contact paths')
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=8);ax.grid(alpha=.15)
    fig.suptitle('Real workpiece in ROBOT_BASE: segmentation and contact-axis planning',fontsize=16)
    fig.savefig(dest/'real_overview.png',dpi=180,facecolor='white');plt.close(fig)
    face_rows=[]
    for fid in selected:
        group=poses[poses.face_id==fid]
        length=sum(np.linalg.norm(np.diff(g[['x_mm','y_mm','z_mm']].to_numpy(),axis=0),axis=1).sum()
                   for _,g in group.groupby('segment_id',sort=False))
        face_rows.append({'face_id':int(fid),'geometry_type':topology.catalog['faces'][int(fid)]['geometry_type'],
            'poses':len(group),'strokes':int(group.segment_id.nunique()),'contact_length_mm':float(length)})
    pd.DataFrame(face_rows).to_csv(dest/'planned_faces.csv',index=False)
    contact_length=sum(x['contact_length_mm'] for x in face_rows)
    audit_path=dest/'exact_face_audit.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    result={'source_points':len(raw),'assigned_points':report['accepted_face_points'],
        'planned_faces':len(selected),'contact_poses':0 if poses is None else len(poses),
        'contact_strokes':0 if poses is None else int(poses.segment_id.nunique()),
        'contact_length_mm':contact_length,'exact_audit':audit,'tool_model_included':False,
        'output_frame':'ROBOT_BASE','units':'mm',
        'scan_xyz_preserved':True,'scan_frame_source':'user_confirmed_robot_base',
        'base_calibration_accuracy_verified':False,
        'execution_allowed':False,'physical_accuracy_verified':False}
    dump_json(dest/'FINAL_RESULT.json',result)
    (dest/'FINAL_RESULT_zh.md').write_text('\n'.join([
        '# 真实工件：机器人基坐标系下的型面分割与接触路径', '',
        '本目录只使用本次真实CAD和扫描；没有人工平板/圆柱。',
        '全部主要结果使用 ROBOT_BASE，长度单位 mm。原始扫描 XYZ 原值保留，不再对扫描重复配准变换。',
        'CAD 定位使用 T_CAD_to_BASE = inverse(T_BASE_to_CAD)，只有 CAD 模型从局部坐标转换到基坐标。',
        f'原始点 {len(raw):,}；通过型面归属检查 {report["accepted_face_points"]:,}。',
        f'有有效路径的主要CAD面 {len(selected)}；接触位姿 {result["contact_poses"]:,}；独立接触段 {result["contact_strokes"]:,}。',
        f'接触段累计长度 {contact_length/1000:.3f} m；段之间未规划加工连接。', '',
        '- REAL_WORKPIECE.html：基坐标下旋转查看分割、已定位CAD、接触路径与刀轴。',
        '- registration_BASE.html：原始基坐标扫描保持不动，检查CAD定位叠加。',
        '- real_overview.png：基坐标下的分割与路径静态总览。',
        '- path_details/PATH_DETAILS.html：三条代表路径的CAD/实测曲面对照，可放大查看完整刀轴；同目录含静态细节图。',
        '- scan_segmented_BASE.ply：全分辨率基坐标分割点云；XYZ、原属性和点序保留。',
        '- contact_poses_BASE.csv：基坐标接触点XYZ、刀轴三轴和XYZW四元数。',
        '- contact_transforms_BASE.npz：T_B_contact；列向量约定，前三列为刀轴X/Y/Z，末列为接触点。',
        '- planned_faces.csv：每个有路径型面的类型、段数和路径长度。',
        '- exact_face_audit.csv/json：网格型面归属与精确STEP抽查对照。', '',
        '分割全点采用保留CAD面来源的网格；边界、竞争和未离散面附近拒绝自动归属。',
        '型面参数与精确CAD裁剪仍在CAD局部坐标计算，最终位置与方向统一映射到机器人基坐标；缺测处保持断开。',
        '工具+Z朝向表面的符号沿用当前几何配置，不代表已完成工具、TCP或材料侧标定。',
        '未指定磨头和去除深度，本次只输出接触点与刀轴几何。',
        '抽查一致性和拟合残差不是独立测量精度，不能据此认证≤0.2mm。',
        '材料侧/实际设备定位仍需验证；未生成设备可执行加工指令。', '']) )
    if len(poses):
        from planning.path_visualization import write_path_details
        write_path_details(dest, dest/'path_details')
