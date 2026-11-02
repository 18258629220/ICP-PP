"""Real scan segmentation and surface-contact paths, without a tool model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pipeline_common import xyz_of, transform_points, dump_json


def _color(face_id):
    import colorsys
    return np.array(colorsys.hsv_to_rgb((int(face_id)*.61803398875)%1., .68, .88))


def _write_labeled_cloud(path, raw, transform, labels, selected_faces):
    """One output row per source row; preserve every original scalar attribute."""
    original = raw.dtype
    reserved = {'cad_face_id','planned_surface','red','green','blue'}
    if reserved.intersection(original.names):
        raise ValueError('Input already has label/color fields; refuse ambiguous overwrite')
    dtype = np.dtype([(name,original.fields[name][0].newbyteorder('<')) for name in original.names]
                     +[(name,'<i4') for name in ['cad_face_id']]
                     +[('planned_surface','u1'),('red','u1'),('green','u1'),('blue','u1')])
    ply_types = {('f',4):'float',('f',8):'double',('i',1):'char',('i',2):'short',
                 ('i',4):'int',('u',1):'uchar',('u',2):'ushort',('u',4):'uint'}
    header = ['ply','format binary_little_endian 1.0','comment Real scan aligned to CAD, mm; source row order preserved',
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
            points = transform_points(xyz_of(raw[start:stop]),transform)
            for j,axis in enumerate('xyz'):
                block[axis] = points[:,j]
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


def write_real_deliverables(dest, raw, transform, topology, label_dir, poses, coverage, report):
    """Write a compact explanatory page plus full-resolution colored PLY."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    dest, label_dir = Path(dest), Path(label_dir)
    labels = np.load(label_dir/'face_labels.npy',mmap_mode='r')
    selected = np.array(sorted(poses.face_id.unique()),int) if poses is not None and len(poses) else np.array([],int)
    _write_labeled_cloud(dest/'scan_segmented_CAD.ply', raw, transform, labels, selected)
    ids = np.linspace(0,len(raw)-1,min(90000,len(raw))).astype(np.int64)
    points = transform_points(xyz_of(raw[ids]), transform)
    faces = np.asarray(labels[ids])
    strokes = _stroke_lines(poses)
    valid = faces >= 0
    planned = np.isin(faces,selected)
    figure = go.Figure()
    nominal = topology.vertices[np.linspace(0,len(topology.vertices)-1,
        min(50000,len(topology.vertices))).astype(int)]
    figure.add_trace(go.Scatter3d(x=nominal[:,0],y=nominal[:,1],z=nominal[:,2],mode='markers',
        name='CAD 标准形状（显示抽样）',visible='legendonly',
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
        hovertemplate='CAD面 %{customdata}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f}) mm<extra></extra>'))
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
    figure.update_layout(title='真实工件：CAD型面分割与随形接触路径<br><sup>分色=型面；蓝线=接触路径；紫线=刀轴。显示抽样，完整结果保存在PLY/CSV。</sup>',
        template='plotly_white',height=900,scene=dict(aspectmode='data',xaxis_title='CAD X / mm',
        yaxis_title='CAD Y / mm',zaxis_title='CAD Z / mm',camera=dict(eye=dict(x=1.4,y=-1.5,z=-1.3))),
        margin=dict(l=0,r=0,b=0,t=90))
    figure.write_html(dest/'REAL_WORKPIECE.html',include_plotlyjs=True,config={'displaylogo':False,'scrollZoom':True})

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
        ax.set_box_aspect(span);ax.view_init(elev=-30,azim=-55)
        ax.set(xlabel='CAD X / mm',ylabel='CAD Y / mm',zlabel='CAD Z / mm',
               title='Real scan — CAD face segmentation' if number==1 else 'Surface-following contact paths')
        ax.tick_params(labelsize=7);ax.grid(alpha=.15)
    fig.suptitle('Real workpiece: face segmentation and contact-axis planning',fontsize=16)
    fig.savefig(dest/'real_overview.png',dpi=180,facecolor='white');plt.close(fig)
    if poses is not None and len(poses):
        curved=[]
        for _, group in poses.groupby('segment_id',sort=False):
            if len(group)<10:
                continue
            normal=group[[f'tool_z_{a}' for a in 'xyz']].to_numpy(float)
            spread=float(np.max(np.degrees(np.arccos(np.clip(normal@normal[len(normal)//2],-1,1)))))
            curved.append((spread,group))
        group=max(curved,key=lambda item:item[0])[1] if curved else poses.iloc[:min(30,len(poses))]
        p=group[['x_mm','y_mm','z_mm']].to_numpy(float)
        sample=np.unique(np.linspace(0,len(p)-1,min(12,len(p))).astype(int))
        fig=plt.figure(figsize=(10,8),layout='constrained');ax=fig.add_subplot(111,projection='3d')
        ax.plot(*p.T,color='#142fa0',lw=2,label='Surface contact path')
        for axis,color in [('x','#c43232'),('y','#188b44'),('z','#9b1c8c')]:
            vectors=group[[f'tool_{axis}_{a}' for a in 'xyz']].to_numpy(float)[sample]
            ax.quiver(*p[sample].T,*vectors.T,length=2.,normalize=True,color=color,
                      arrow_length_ratio=.25,label=f'Tool {axis.upper()}')
        lo,hi=p.min(0)-2.5,p.max(0)+2.5
        for setter,a,b in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],lo,hi):setter(a,b)
        ax.set_box_aspect(np.maximum(hi-lo,1.));ax.view_init(elev=25,azim=-55)
        ax.set(xlabel='CAD X / mm',ylabel='CAD Y / mm',zlabel='CAD Z / mm',
               title='Contact-frame poses on a curved region\nPurple +Z points into the surface; origin is the contact point')
        ax.legend(loc='upper left');fig.savefig(dest/'contact_axes_detail.png',dpi=180,facecolor='white');plt.close(fig)
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
        'execution_allowed':False,'physical_accuracy_verified':False}
    dump_json(dest/'FINAL_RESULT.json',result)
    (dest/'FINAL_RESULT_zh.md').write_text('\n'.join([
        '# 真实工件：型面分割与接触路径', '',
        '本目录只使用本次真实CAD和扫描；没有人工平板/圆柱。',
        f'原始点 {len(raw):,}；通过型面归属检查 {report["accepted_face_points"]:,}。',
        f'有有效路径的主要CAD面 {len(selected)}；接触位姿 {result["contact_poses"]:,}；独立接触段 {result["contact_strokes"]:,}。',
        f'接触段累计长度 {contact_length/1000:.3f} m；段之间未规划加工连接。', '',
        '- REAL_WORKPIECE.html：旋转查看分割、接触路径与刀轴，可切换图层。',
        '- real_overview.png：分割与路径静态总览。',
        '- scan_segmented_CAD.ply：全分辨率分割点云；原始属性和点序保留。',
        '- contact_poses_CAD.csv / contact_poses_SCAN.csv：接触点XYZ、刀轴三轴和XYZW四元数。',
        '- planned_faces.csv：每个有路径型面的类型、段数和路径长度。',
        '- exact_face_audit.csv/json：网格型面归属与精确STEP抽查对照。', '',
        '分割全点采用保留CAD面来源的网格；边界、竞争和未离散面附近拒绝自动归属。',
        '路径定义在精确CAD裁剪面上，实际接触点和法向由同面扫描局部拟合；缺测处保持断开。',
        '未指定磨头和去除深度，本次只输出接触点与刀轴几何。',
        '抽查一致性和拟合残差不是独立测量精度，不能据此认证≤0.2mm。',
        '材料侧/实际设备定位仍需验证；未生成设备可执行加工指令。', '']) )
