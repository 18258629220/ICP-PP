"""Additional diagnostics of the saved transform; does not replace baseline outputs."""
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'output/.mplconfig'))
import json
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import cKDTree
from pipeline_common import ROOT,config,dump_json,read_ply,xyz_of,cloud,transform_points,closest_query,distance_stats
from importlib import import_module
cfg=config();out=ROOT/cfg['output_dir'];dest=out/'assessment';dest.mkdir(exist_ok=True)
df=pd.read_csv(out/'deviation_points.csv');p=df[['scan_x_mm','scan_y_mm','scan_z_mm']].to_numpy()
n=df[['normal_x','normal_y','normal_z']].to_numpy();q=df[['cad_closest_x_mm','cad_closest_y_mm','cad_closest_z_mm']].to_numpy()
d=df.euclidean_distance_mm.to_numpy();signed=df.signed_deviation_mm.to_numpy();good=df.quality_mask.to_numpy(bool)
rows=[]
for name,mask in [('all',np.ones(len(df),bool)),('quality_filtered',good),('distance_le_5',d<=5),('distance_gt_5',d>5),
    ('quality_horizontal_abs_nz_ge_0.9',good & (np.abs(n[:,2])>=.9)),
    ('quality_oblique_or_side_abs_nz_lt_0.9',good & (np.abs(n[:,2])<.9))]:
    row={'scope':name,**distance_stats(d[mask]),'signed_median_mm':float(np.median(signed[mask]))}
    for t in [.1,.2,.5,1,2,5,10,50,100]:row[f'fraction_distance_le_{t:g}mm']=float(np.mean(d[mask]<=t))
    rows.append(row)
pd.DataFrame(rows).to_csv(dest/'distance_strata.csv',index=False)
fail_dist=d>5;fail_tangent=df.tangential_residual_mm.to_numpy()>.2;fail_norm=df.normal_agreement_abs_cosine.to_numpy()<.7
mask_info={'n':len(df),'quality_rejected':int((~good).sum()),'distance_gt_5':int(fail_dist.sum()),
  'tangent_gt_0_2':int(fail_tangent.sum()),'normal_cos_lt_0_7':int(fail_norm.sum()),
  'distance_gt_5_squared_error_fraction':float(np.sum(d[fail_dist]**2)/np.sum(d**2)),
  'quality_horizontal_fraction':float(np.mean(np.abs(n[good,2])>=.9)),
  'distance_to_normal_difference_p99_mm':float(np.percentile(d-np.abs(signed),99))}
dump_json(dest/'quality_breakdown.json',mask_info)
print('quality',mask_info,flush=True)
# Spatial clusters of extreme residuals, retained as unclassified scan points.
far=p[d>5]; farcloud=cloud(far);labels=np.array(farcloud.cluster_dbscan(eps=6,min_points=10))
clusters=[]
for lab in np.unique(labels):
    mask=labels==lab;pts=far[mask];dd=d[d>5][mask]
    clusters.append({'label':int(lab),'n':int(mask.sum()),'bbox_min_mm':pts.min(0).tolist(),
      'bbox_max_mm':pts.max(0).tolist(),'median_distance_mm':float(np.median(dd)),'max_distance_mm':float(dd.max())})
clusters.sort(key=lambda a:-a['n']);dump_json(dest/'far_point_clusters.json',clusters)
# Approximate point-to-plane local geometric sensitivity, with rotation scaled by RMS radius.
mask=good & (d<=2);pp=p[mask];nn=n[mask];center=pp.mean(0);radius=float(np.sqrt(np.mean(np.sum((pp-center)**2,axis=1))))
J=np.column_stack([np.cross((pp-center)/radius,nn),nn]);w=np.minimum(1,1/np.maximum(np.abs(signed[mask]),1e-12))
H=(J.T*w)@J/len(J);eigen,vec=np.linalg.eigh(H)
normal_moment=nn.T@nn/len(nn)
obs={'points':len(pp),'rms_radius_mm':radius,'center_mm':center.tolist(),
 'scaled_jacobian_information_eigenvalues':eigen.tolist(),'condition_ratio_max_min':float(eigen[-1]/eigen[0]),
 'weakest_mode_rot_scaled_then_translation':vec[:,0].tolist(),
 'mean_normal_squared_components_xyz':np.diag(normal_moment).tolist(),
 'definition':'Approximate local geometry only; centered rotations scaled by RMS radius; not a calibrated covariance'}
dump_json(dest/'geometric_sensitivity.json',obs);print('geometry',obs,flush=True)
# Raw-point statistics: density-weighted, unlike 1mm voxel statistics.
full=np.load(out/'deviation_full_resolution.npy',mmap_mode='r');fr=full['euclidean_distance_mm']
dump_json(dest/'full_resolution_stats.json',distance_stats(fr)|{'max_distance_mm':float(np.max(fr)),
    'fraction_gt_5mm':float(np.mean(fr>5)),'density_weighted':True})
# Small-to-large error location plots and CDF, using all analysis voxels.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
colors=['#2b8cbe','#7bccc4','#fed976','#fd8d3c','#bd0026']
thresholds=[0,.2,1,2,5,np.inf]
for ax,(i,j,title) in zip(axes,[(0,1,'XY residual bands'),(0,2,'XZ residual bands'),(1,2,'YZ residual bands')]):
    for k in range(5):
        mask=(d>thresholds[k])&(d<=thresholds[k+1]);lab=f'{thresholds[k]:g}–{thresholds[k+1]:g} mm' if k<4 else '>5 mm'
        ax.scatter(p[mask,i],p[mask,j],s=.5,color=colors[k],alpha=.65,label=lab,rasterized=True)
    ax.set(xlabel='XYZ'[i]+' (mm)',ylabel='XYZ'[j]+' (mm)',title=title);ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.15)
handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,ncol=5,loc='outside lower center',markerscale=6)
fig.savefig(dest/'residual_bands.png',dpi=170);plt.close(fig)
# Exclude one spatial quadrant at a time. Starting pose is the saved solution in CAD coordinates.
T=np.load(out/'T_scan_to_CAD.npy');src=o3d.io.read_point_cloud(str(out/'scan_icp_filtered.ply'));src.transform(T)
s=np.asarray(src.points);cad=o3d.io.read_point_cloud(str(out/'cad_surface_sample.ply')).voxel_down_sample(cfg['icp_cad_voxel_mm']);cad.normalize_normals()
reg=o3d.pipelines.registration;pipeline=import_module('02_register_and_analyze');scene=pipeline.get_scene(pipeline.get_mesh(out))
xcut=float(np.median(s[:,0]));ycut=float(np.median(s[:,1]));sg=(s[:,0]>=xcut).astype(int)+2*(s[:,1]>=ycut).astype(int)
eg=(p[:,0]>=xcut).astype(int)+2*(p[:,1]>=ycut).astype(int)
results=[]
for quadrant in range(4):
    train=cloud(s[sg!=quadrant]);delta=np.eye(4)
    for corr in [5.,2.]:
        fit=reg.registration_icp(train,cad,corr,delta,
          reg.TransformationEstimationPointToPlane(reg.HuberLoss(1)),
          reg.ICPConvergenceCriteria(relative_fitness=1e-7,relative_rmse=1e-7,max_iteration=100))
        delta=fit.transformation
    evalp=transform_points(p,delta);displacement=np.linalg.norm(evalp-p,axis=1)
    ans=closest_query(scene,evalp);dist=np.linalg.norm(evalp-ans['points'],axis=1);held=(eg==quadrant)
    r={'held_quadrant':quadrant,'n_training':len(train.points),'translation_delta_mm':delta[:3,3].tolist(),
       'rotation_delta_degrees':float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))),
       'median_displacement_mm':float(np.median(displacement)), 'p95_displacement_mm':float(np.percentile(displacement,95)),
       'heldout_baseline':distance_stats(d[held]),'heldout_refit':distance_stats(dist[held]),'delta_T':delta.tolist()}
    results.append(r);print('spatial holdout',r,flush=True)
dump_json(dest/'spatial_holdout.json',{'xcut_mm':xcut,'ycut_mm':ycut,'results':results,'independent_scan':False})
print('ASSESSMENT COMPLETE',flush=True)
