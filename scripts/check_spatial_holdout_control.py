import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
import open3d as o3d
from pipeline_common import config,transform_points,dump_json
out=root/config()['output_dir'];(out/'assessment').mkdir(exist_ok=True);T=np.load(out/'T_scan_to_CAD.npy')
src=o3d.io.read_point_cloud(str(out/'scan_icp_filtered.ply'));src.transform(T)
tgt=o3d.io.read_point_cloud(str(out/'cad_surface_sample.ply')).voxel_down_sample(.5);tgt.normalize_normals()
reg=o3d.pipelines.registration;delta=np.eye(4)
for corr in [5.,2.]:
 fit=reg.registration_icp(src,tgt,corr,delta,reg.TransformationEstimationPointToPlane(reg.HuberLoss(1)),reg.ICPConvergenceCriteria(relative_fitness=1e-7,relative_rmse=1e-7,max_iteration=100));delta=fit.transformation
p=np.asarray(src.points);dist=np.linalg.norm(transform_points(p,delta)-p,axis=1)
r={'control_full_scan_median_displacement_mm':float(np.median(dist)),'p95_displacement_mm':float(np.percentile(dist,95)),'fitness':fit.fitness,'rmse_mm':fit.inlier_rmse,'delta_T':delta.tolist()}
print(r);dump_json(out/'assessment/spatial_control.json',r)
assert r['p95_displacement_mm']<1e-3
