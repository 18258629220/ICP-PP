"""Diagnostics for failed full-CAD matching: one-sided CAD views and scan components."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import json
import time
import numpy as np
import open3d as o3d
import kiss_matcher as km
from scipy.spatial import cKDTree
from pipeline_common import ROOT,cloud,transform_points,dump_json
out=ROOT/'output'
source=o3d.io.read_point_cloud(str(out/'scan_downsampled_filtered.ply'))
cad=o3d.io.read_point_cloud(str(out/'cad_surface_sample.ply'))
s=np.asarray(source.points);c=np.asarray(cad.points)
labels=np.array(source.cluster_dbscan(eps=6,min_points=10,print_progress=False))
unique,counts=np.unique(labels,return_counts=True);print('Scan components',list(zip(unique.tolist(),counts.tolist())),flush=True)
main_label=unique[unique>=0][np.argmax(counts[unique>=0])]
sources={'all':s,'main':s[labels==main_label]}
o3d.io.write_point_cloud(str(out/'scan_main_component.ply'),cloud(sources['main']))
targets={'full':c,'back_z_le_15':c[c[:,2]<=15],'front_z_ge_35':c[c[:,2]>=35]}
for name,cam in [('visible_back',[0,0,-5000]),('visible_front',[0,0,5000])]:
 _,ids=cad.hidden_point_removal(cam,50000)
 targets[name]=c[ids];o3d.io.write_point_cloud(str(out/f'cad_{name}.ply'),cloud(c[ids]));print(name,len(ids),flush=True)
rows=[];tree=cKDTree(c)
for sn,sp in sources.items():
 for tn,tp in targets.items():
  for resolution in [1.,2.,3.,5.]:
   src=np.asarray(cloud(sp).voxel_down_sample(resolution).points,dtype=np.float32)
   tgt=np.asarray(cloud(tp).voxel_down_sample(resolution).points,dtype=np.float32)
   # Explicit centering changes numeric conditioning; transforms are composed back to original frames.
   sc=src.mean(0);tc=tgt.mean(0)
   matcher=km.KISSMatcher(km.KISSMatcherConfig(resolution,enable_noise_bound_clamping=False))
   r=matcher.estimate(src-sc,tgt-tc);T=np.eye(4);T[:3,:3]=r.rotation;T[:3,3]=np.asarray(r.translation).reshape(3)+tc-T[:3,:3]@sc
   count=int(matcher.get_num_final_inliers());p=transform_points(s,T);dist=tree.query(p,workers=4)[0]
   row={'source':sn,'target':tn,'resolution_mm':resolution,'inliers':count,'median_mm':float(np.median(dist)),
         'fraction_le_5mm':float(np.mean(dist<=5)),'T':T.tolist()};rows.append(row)
   print({k:v for k,v in row.items() if k!='T'},flush=True)
   dump_json(out/'kiss_visibility_experiments.json',rows)
