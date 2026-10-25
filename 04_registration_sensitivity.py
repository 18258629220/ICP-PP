"""Internal local-stability checks; do not represent independent physical calibration."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import open3d as o3d
from pipeline_common import ROOT,config,cloud,subset,transform_points,dump_json
cfg=config();out=ROOT/cfg['output_dir'];T=np.load(out/'T_scan_to_CAD.npy')
source=o3d.io.read_point_cloud(str(out/'scan_icp_filtered.ply'))
cad=o3d.io.read_point_cloud(str(out/'cad_surface_sample.ply'));target=cad.voxel_down_sample(cfg['icp_cad_voxel_mm']);target.normalize_normals()
p=np.asarray(source.points);evaluation=subset(p,20000,19);rng=np.random.default_rng(19)
reg=o3d.pipelines.registration;rows=[]
for j in range(6):
    delta=np.eye(4);delta[:3,:3]=o3d.geometry.get_rotation_matrix_from_xyz(rng.uniform(-2,2,3)*np.pi/180)
    delta[:3,3]=rng.uniform(-2,2,3)
    initial=delta@T
    src=source if j<4 else cloud(p[np.arange(len(p))%2==(j-4)])
    fit_T=initial
    for corr in cfg['icp_correspondence_mm']:
        fit=reg.registration_icp(src,target,corr,fit_T,reg.TransformationEstimationPointToPlane(reg.HuberLoss(cfg['icp_robust_huber_mm'])),reg.ICPConvergenceCriteria(relative_fitness=1e-7,relative_rmse=1e-7,max_iteration=100))
        fit_T=fit.transformation
    displacement=np.linalg.norm(transform_points(evaluation,fit_T)-transform_points(evaluation,T),axis=1)
    row={'case':j,'type':'perturbed_full_scan' if j<4 else 'alternating_half_scan',
        'fitness':fit.fitness,'rmse_mm':fit.inlier_rmse,
        'median_displacement_mm':float(np.median(displacement)),
        'p95_displacement_mm':float(np.percentile(displacement,95)),'T':fit_T.tolist()}
    rows.append(row);print({k:v for k,v in row.items() if k!='T'},flush=True)
roi_median=None
if (out/'diagnostics/T_back_roi.npy').exists():
    back=np.load(out/'diagnostics/T_back_roi.npy')
    roi_displacement=np.linalg.norm(transform_points(evaluation,back)-transform_points(evaluation,T),axis=1)
    roi_median=float(np.median(roi_displacement))
dump_json(out/'registration_sensitivity.json',{'reference_transform':T.tolist(),'physical_accuracy_not_established':True,'perturbation_rotation_component_degrees':2,'perturbation_translation_component_mm':2,'cases':rows,'full_vs_back_roi_median_displacement_mm':roi_median})
