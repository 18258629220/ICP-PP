"""Offline interactive HTML and static views; plots never imply unseen coverage."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import open3d as o3d
from pipeline_common import ROOT,read_ply,subset,transform_points,assert_rigid

CAD_COLOR='#718096'; SCAN_COLOR='#ea7016'

def scatter3(p,name,color,opacity=.6,size=1.3):
    return go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',name=name,
        marker=dict(size=size,color=color,opacity=opacity))

def layout(title,frame='CAD'):
    return dict(title=title,template='plotly_white',margin=dict(l=0,r=0,b=0,t=75),
        scene=dict(xaxis_title=f'{frame}_X (mm)',yaxis_title=f'{frame}_Y (mm)',zaxis_title=f'{frame}_Z (mm)',aspectmode='data'),
        legend=dict(x=0,y=1),height=780)

def equal3d(ax,points,frame='CAD'):
    lo=points.min(axis=0); hi=points.max(axis=0); center=(lo+hi)/2; span=max(hi-lo)/2
    ax.set_xlim(center[0]-span,center[0]+span); ax.set_ylim(center[1]-span,center[1]+span)
    ax.set_zlim(center[2]-span,center[2]+span); ax.set_box_aspect([1,1,1])
    ax.set_xlabel(f'{frame}_X (mm)'); ax.set_ylabel(f'{frame}_Y (mm)'); ax.set_zlabel(f'{frame}_Z (mm)')
    ax.view_init(elev=25,azim=-60)

def plots(cfg,out):
    T=np.load(out/'T_scan_to_CAD.npy'); assert_rigid(T)
    in_base=cfg.get('scan_frame')=='ROBOT_BASE'
    frame='BASE' if in_base else 'CAD'
    cad=np.asarray(o3d.io.read_point_cloud(str(out/'cad_surface_sample.ply')).points)
    c=subset(cad,cfg['vis_max_points'])
    raw,_=read_ply(ROOT/cfg['scan_path']); ids=np.random.default_rng(7).choice(len(raw),min(len(raw),cfg['vis_max_points']),replace=False)
    # Read original XYZ directly: the BASE scan must not be rounded or transformed again.
    before=np.column_stack([raw[a][ids] for a in 'xyz']).astype(float)
    before=before[np.isfinite(before).all(axis=1)]
    after=before if in_base else transform_points(before,T)
    cad_display=transform_points(c,np.linalg.inv(T)) if in_base else c
    cad_bounds=transform_points(cad,np.linalg.inv(T)) if in_base else cad
    scan_name='Original scan — ROBOT_BASE' if in_base else 'Aligned scan — CAD'
    registration_title=('ROBOT_BASE: original scan fixed; CAD located by T_CAD_to_BASE'
                        if in_base else 'CAD audit: CAD / aligned scan')
    fig=go.Figure([scatter3(cad_display,'Located CAD' if in_base else 'CAD surface',CAD_COLOR,.35),
                   scatter3(after,scan_name,SCAN_COLOR,.8)])
    fig.update_layout(**layout(registration_title,frame))
    fig.write_html(out/'registration_3d.html',include_plotlyjs=True)
    fig=plt.figure(figsize=(14,7),layout='constrained')
    titles=('Original scan — ROBOT_BASE','CAD located in ROBOT_BASE') if in_base else ('Before registration — input / CAD coordinates','After registration — CAD audit')
    for i,p,title in [(1,before,titles[0]),(2,after,titles[1])]:
        ax=fig.add_subplot(1,2,i,projection='3d')
        if not in_base or i==2:
            ax.scatter(*cad_display.T,s=.15,c=CAD_COLOR,alpha=.25,label='Located CAD' if in_base else 'CAD')
        ax.scatter(*p.T,s=.2,c=SCAN_COLOR,alpha=.5,label='Scan')
        equal3d(ax,np.vstack([cad_display,p]) if not in_base or i==2 else p,
                frame if in_base or i==2 else 'INPUT / CAD')
        ax.set_title(title); ax.legend(markerscale=12)
    fig.savefig(out/'registration_before_after.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    for ax,(i,j,name) in zip(axes,[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
        ax.scatter(cad_display[:,i],cad_display[:,j],s=.2,c=CAD_COLOR,alpha=.3,label='Located CAD' if in_base else 'CAD')
        ax.scatter(after[:,i],after[:,j],s=.25,c=SCAN_COLOR,alpha=.5,label=scan_name)
        ax.set(xlabel=f'{frame}_{"XYZ"[i]} (mm)',ylabel=f'{frame}_{"XYZ"[j]} (mm)',title=f'{name} — {frame}')
        ax.set_aspect('equal',adjustable='datalim'); ax.legend(markerscale=10); ax.grid(alpha=.15)
    fig.savefig(out/'registration_projections.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,(i,j,name) in zip(axes,[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
        ax.scatter(cad_display[:,i],cad_display[:,j],s=.25,c=CAD_COLOR,alpha=.4,label='Located CAD' if in_base else 'CAD')
        ax.scatter(after[:,i],after[:,j],s=.4,c=SCAN_COLOR,alpha=.55,label=scan_name)
        ax.set(xlabel=f'{frame}_{"XYZ"[i]} (mm)',ylabel=f'{frame}_{"XYZ"[j]} (mm)',title=f'{name} — {frame}',
               xlim=(cad_bounds[:,i].min()-5,cad_bounds[:,i].max()+5),ylim=(cad_bounds[:,j].min()-5,cad_bounds[:,j].max()+5))
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.65,.08),ncol=2,markerscale=8)
    fig.suptitle(f'{frame}: located CAD bounds detail — out-of-bounds points visible in overview')
    fig.savefig(out/'registration_detail.png',dpi=180);plt.close(fig)
    # Deviation tables retain CAD-local geometry for internal numerical review.
    audit_link='<br><sup>CAD coordinates: internal audit; <a href="registration_3d.html">open primary registration view</a></sup>'
    df=pd.read_csv(out/'deviation_points.csv'); selected=df.sample(min(cfg['vis_max_points'],len(df)),random_state=7)
    p=selected[['scan_x_mm','scan_y_mm','scan_z_mm']].to_numpy(); d=selected.signed_deviation_mm.to_numpy()
    lim=max(.1,float(np.percentile(np.abs(df.signed_deviation_mm),98)))
    heat=go.Figure([scatter3(c,'CAD reference',CAD_COLOR,.15)])
    heat.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',name='Measured scan',
        customdata=selected[['signed_deviation_mm','euclidean_distance_mm','quality_mask']].to_numpy(),
        hovertemplate='X=%{x:.2f}, Y=%{y:.2f}, Z=%{z:.2f}<br>Normal deviation=%{customdata[0]:.3f} mm<br>Surface distance=%{customdata[1]:.3f} mm<br>Quality mask=%{customdata[2]}<extra></extra>',
        marker=dict(size=2,color=d,colorscale='RdBu',reversescale=True,cmin=-lim,cmax=lim,
            colorbar=dict(title='Normal deviation<br>(mm)'))))
    mesh_info=json.loads((out/'mesh_quality.json').read_text())
    sign='outward mesh normals verified' if mesh_info['outward_normals_verified'] else 'SIGN UNVERIFIED: local triangle normals only'
    heat.update_layout(**layout(f'CAD audit: local normal deviation — {sign}<br><sup>Symmetric color clipping: ±{lim:.2f} mm; example tolerance only; no robot approval</sup>'+audit_link))
    heat.write_html(out/'deviation_heatmap_all_points_3d.html',include_plotlyjs=True)
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    for ax,(i,j,name) in zip(axes,[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
        ax.scatter(c[:,i],c[:,j],s=.2,c=CAD_COLOR,alpha=.15)
        sc=ax.scatter(p[:,i],p[:,j],c=d,s=.6,cmap='RdBu_r',vmin=-lim,vmax=lim)
        ax.set(xlabel=f'CAD_{"XYZ"[i]} (mm)',ylabel=f'CAD_{"XYZ"[j]} (mm)',title=f'{name} — CAD audit')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.15)
    fig.colorbar(sc,ax=axes,label='Local normal deviation (mm; clipped colors)',shrink=.65)
    fig.suptitle('CAD internal audit: measured scan regions only — '+sign,fontsize=12)
    fig.savefig(out/'deviation_heatmap_projections.png',dpi=180);plt.close(fig)
    good=selected.quality_mask.to_numpy(dtype=bool)
    quality_dev=df.loc[df.quality_mask,'signed_deviation_mm'].to_numpy()
    lim=max(cfg['example_tolerance_mm'],float(np.percentile(np.abs(quality_dev),98))) if len(quality_dev) else lim
    heat=go.Figure([scatter3(c,'CAD reference',CAD_COLOR,.12),scatter3(p[~good],'Rejected quality / uncertain', '#777777',.35)])
    heat.add_trace(go.Scatter3d(x=p[good,0],y=p[good,1],z=p[good,2],mode='markers',name='Quality-filtered measurements',
        customdata=selected.loc[good,['signed_deviation_mm','euclidean_distance_mm']].to_numpy(),
        hovertemplate='X=%{x:.2f}, Y=%{y:.2f}, Z=%{z:.2f}<br>Normal deviation=%{customdata[0]:.3f} mm<br>Surface distance=%{customdata[1]:.3f} mm<extra></extra>',
        marker=dict(size=2,color=d[good],colorscale='RdBu',reversescale=True,cmin=-lim,cmax=lim,colorbar=dict(title='Local normal<br>deviation (mm)'))))
    heat.update_layout(**layout(f'CAD audit: quality-filtered local deviations — {sign}<br><sup>Gray: rejected quality; colors clipped at ±{lim:.2f} mm; full unfiltered data in companion view</sup>'+audit_link))
    heat.write_html(out/'deviation_heatmap_3d.html',include_plotlyjs=True)
    fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
    ax.scatter(p[~good,0],p[~good,1],s=.7,c='#777777',alpha=.3,label='Rejected quality / uncertain')
    sc=ax.scatter(p[good,0],p[good,1],c=d[good],s=.9,cmap='RdBu_r',vmin=-lim,vmax=lim)
    ax.set(xlabel='CAD_X (mm)',ylabel='CAD_Y (mm)',title=f'CAD audit: XY quality-filtered local normal deviation\nColors clipped at ±{lim:.2f} mm; gray = rejected quality; {sign}')
    ax.legend(loc='lower right',markerscale=5)
    ax.set_aspect('equal',adjustable='datalim');fig.colorbar(sc,ax=ax,label='Local normal deviation (mm)')
    fig.savefig(out/'deviation_heatmap_xy.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    axes[0].hist(df.signed_deviation_mm,bins=160,range=(-lim,lim),color='#387aa5')
    axes[0].set(xlabel='Local normal deviation (mm)',ylabel='1 mm voxel count',title=f'All-voxel distribution, central ±{lim:.2f} mm shown')
    tolerance=cfg['example_tolerance_mm']
    for v in [-tolerance,tolerance]: axes[0].axvline(v,color='#d34f36',linestyle='--',label='Example ±1 mm' if v>0 else None)
    axes[0].legend()
    for label,key in [('Surface distance','euclidean_distance_mm'),('|Normal projection|','absolute_deviation_mm')]:
        vals=np.sort(df[key].to_numpy()); axes[1].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=label)
    axes[1].set(xlabel='Distance (mm)',ylabel='Cumulative fraction',xlim=(0,5),ylim=(0,1),title=f'All voxels; {100*np.mean(df.euclidean_distance_mm>5):.2f}% have distance >5 mm')
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(out/'deviation_distribution.png',dpi=180);plt.close(fig)
    cover=np.load(out/'cad_coverage.npz'); ids=np.random.default_rng(7).choice(len(cover['points']),min(len(cover['points']),cfg['vis_max_points']),replace=False)
    p=cover['points'][ids]; distances=cover['nearest_scan_distance_mm'][ids]
    fig=go.Figure([go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',
        marker=dict(size=2,color=np.minimum(distances,10),colorscale='Viridis',cmin=0,cmax=10,
        colorbar=dict(title='CAD to scan<br>distance (mm, capped 10)')))])
    fig.update_layout(**layout('CAD audit: scan-coverage proxy — far regions have no measured deviation'+audit_link))
    fig.write_html(out/'cad_coverage_3d.html',include_plotlyjs=True)
