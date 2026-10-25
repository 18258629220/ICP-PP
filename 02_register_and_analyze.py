"""Run measured registration candidates, gated analysis, and offline plots.
Stages can be rerun independently: --stage prepare/register/analyze/plots/all.
"""
import argparse
import gc
import importlib.metadata
import logging
import os
from pathlib import Path
import time
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'output/.mplconfig'))
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree
from pipeline_common import (ROOT,config,dump_json,log_setup,sha256,read_ply,xyz_of,
    cloud,subset,transform_points,assert_rigid,write_aligned_ply,closest_query,distance_stats)


def get_mesh(out):
    mesh=o3d.io.read_triangle_mesh(str(out/'cad_mesh_clean.ply'))
    mesh.compute_triangle_normals()
    return mesh

def get_scene(mesh):
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene

def prepare(cfg,out):
    import trimesh
    records,meta=read_ply(ROOT/cfg['scan_path']); xyz=xyz_of(records)
    finite=np.isfinite(xyz).all(axis=1)
    meta.update({'sha256':sha256(ROOT/cfg['scan_path']),'finite_count':int(finite.sum()),
                 'nonfinite_count':int((~finite).sum()),'units':'mm',
                 'bbox_min_mm':xyz[finite].min(axis=0).tolist(),
                 'bbox_max_mm':xyz[finite].max(axis=0).tolist(),
                 'quantiles_mm':np.percentile(xyz[finite],[0.5,50,99.5],axis=0).tolist()})
    dump_json(out/'scan_metadata.json',meta)
    logging.info('Scan header and bounds: %s',meta)
    raw=cloud(xyz[finite]); del xyz,records; gc.collect()
    for voxel,name in [(cfg['registration_voxel_mm'],'scan_downsampled'),
                        (cfg['icp_scan_voxel_mm'],'scan_icp')]:
        down=raw.voxel_down_sample(voxel)
        o3d.io.write_point_cloud(str(out/f'{name}.ply'),down)
        clean,ids=down.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.5)
        o3d.io.write_point_cloud(str(out/f'{name}_filtered.ply'),clean)
        logging.info('%s: voxel=%g mm, before SOR=%d, retained=%d',name,voxel,len(down.points),len(clean.points))
    del raw,down,clean; gc.collect()
    tm=trimesh.load_mesh(out/'cad_mesh.stl',process=True)
    # Do not fill holes or modify engineering surfaces; inspect orientation instead.
    removed_degenerate=int(np.sum(tm.area_faces<1e-12))
    tm.update_faces(tm.area_faces>=1e-12)
    tm.remove_unreferenced_vertices()
    mesh_meta={'vertices':len(tm.vertices),'triangles':len(tm.faces),
        'watertight':bool(tm.is_watertight),'winding_consistent':bool(tm.is_winding_consistent),
        'signed_volume_mm3':float(tm.volume),'surface_area_mm2':float(tm.area),
        'bbox_mm':tm.bounds.tolist(),'degenerate_faces':int(np.sum(tm.area_faces<1e-12)),
        'removed_zero_area_faces':removed_degenerate}
    mesh_meta['outward_normals_verified']=bool(tm.is_watertight and tm.is_winding_consistent and tm.volume>0)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices),o3d.utility.Vector3iVector(tm.faces))
    mesh.compute_triangle_normals(); mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out/'cad_mesh_clean.ply'),mesh)
    dump_json(out/'mesh_quality.json',mesh_meta); logging.info('Mesh topology: %s',mesh_meta)
    o3d.utility.random.seed(cfg['random_seed'])
    cad=mesh.sample_points_uniformly(number_of_points=cfg['cad_sample_count'],use_triangle_normal=True)
    o3d.io.write_point_cloud(str(out/'cad_surface_sample.ply'),cad)
    dump_json(out/'software_versions.json',{k:importlib.metadata.version(k) for k in
              ['numpy','scipy','open3d','kiss-matcher','cadquery','trimesh','pandas','plotly','matplotlib']})


def register(cfg,out):
    import kiss_matcher as km
    cad=o3d.io.read_point_cloud(str(out/'cad_surface_sample.ply'))
    coarse_cad=cad
    if cfg.get('kiss_target_roi'):
        roi=cfg['kiss_target_roi']; pts=np.asarray(cad.points)
        coarse_cad=cloud(pts[pts[:,roi['axis']]<=roi['max_mm']])
        o3d.io.write_point_cloud(str(out/'cad_coarse_roi.ply'),coarse_cad)
    source=o3d.io.read_point_cloud(str(out/'scan_downsampled_filtered.ply'))
    fine_source=o3d.io.read_point_cloud(str(out/'scan_icp_filtered.ply'))
    target=cad.voxel_down_sample(cfg['icp_cad_voxel_mm']); target.normalize_normals()
    eval_points=np.asarray(o3d.io.read_point_cloud(str(out/'scan_icp.ply')).points)
    mesh=get_mesh(out); scene=get_scene(mesh)
    evaluation=subset(eval_points,80000,cfg['random_seed']+1)
    result_rows=[]; candidates=[]
    reg=o3d.pipelines.registration
    for resolution in cfg['kiss_resolutions_mm']:
        src=source.voxel_down_sample(resolution); tgt=coarse_cad.voxel_down_sample(resolution)
        logging.info('KISS start: resolution=%g, source=%d target=%d',resolution,len(src.points),len(tgt.points))
        start=time.time(); matcher=km.KISSMatcher(km.KISSMatcherConfig(resolution,enable_noise_bound_clamping=cfg['kiss_enable_noise_bound_clamping']))
        sp=np.asarray(src.points,dtype=np.float32);tp=np.asarray(tgt.points,dtype=np.float32)
        sc=sp.mean(axis=0) if cfg['kiss_center_inputs'] else np.zeros(3)
        tc=tp.mean(axis=0) if cfg['kiss_center_inputs'] else np.zeros(3)
        result=matcher.estimate(sp-sc,tp-tc)
        T=np.eye(4); T[:3,:3]=result.rotation
        T[:3,3]=np.asarray(result.translation).reshape(3)+tc-T[:3,:3]@sc
        assert_rigid(T)
        row={'resolution_mm':resolution,'rotation_inliers':int(matcher.get_num_rotation_inliers()),
             'final_inliers':int(matcher.get_num_final_inliers()),'kiss_seconds':time.time()-start,
             'T_kiss':T.tolist()}
        tag=f'{resolution:g}mm'; np.savetxt(out/f'T_kiss_{tag}.txt',T,fmt='%.12f')
        logging.info('KISS candidate: %s',row)
        if row['final_inliers']<5:
            row['accepted_for_icp']=False; result_rows.append(row)
            dump_json(out/'registration_candidates.json',result_rows)
            continue
        row['accepted_for_icp']=True
        for max_corr in cfg['icp_correspondence_mm']:
            fit=reg.registration_icp(fine_source,target,max_corr,T,
                reg.TransformationEstimationPointToPlane(reg.HuberLoss(cfg['icp_robust_huber_mm'])),
                reg.ICPConvergenceCriteria(relative_fitness=1e-7,relative_rmse=1e-7,max_iteration=100))
            T=np.array(fit.transformation); assert_rigid(T)
            row[f'icp_{max_corr:g}mm']={'fitness':fit.fitness,'inlier_rmse_mm':fit.inlier_rmse}
            logging.info('ICP %s max_corr=%g fitness=%.6f RMSE=%.6f',tag,max_corr,fit.fitness,fit.inlier_rmse)
        p=transform_points(evaluation,T); ans=closest_query(scene,p)
        dist=np.linalg.norm(p-ans['points'],axis=1)
        row['surface_distance']=distance_stats(dist); row['T_final']=T.tolist()
        candidates.append((np.median(dist),T,row))
        result_rows.append(row)
        dump_json(out/'registration_candidates.json',result_rows)
    if not candidates:
        dump_json(out/'registration_quality.json',{'passed':False,'reason':'No KISS candidate >= 5 inliers'})
        raise RuntimeError('No valid coarse registration. Analysis deliberately stopped.')
    _,T,best=min(candidates,key=lambda x:x[0]); assert_rigid(T)
    metrics=best['surface_distance']; fine=best[f"icp_{cfg['icp_correspondence_mm'][-1]:g}mm"]
    passed=(fine['fitness']>=cfg['quality_gate_min_fitness'] and
            metrics['median_mm']<=cfg['quality_gate_max_median_distance_mm'])
    pairwise=[]
    for _,other,row in candidates:
        delta=transform_points(evaluation,other)-transform_points(evaluation,T)
        pairwise.append({'resolution_mm':row['resolution_mm'],
            'median_point_displacement_mm':float(np.median(np.linalg.norm(delta,axis=1)))})
    quality={'passed':bool(passed),'gate_type':'engineering screening; not metrology certification',
        'selected_resolution_mm':best['resolution_mm'],'selected_candidate':best,
        'candidate_agreement':pairwise,'sample_count_for_surface_validation':len(evaluation),
        'source':'scan','target':'CAD','transform_direction':'p_C = R @ p_S + t',
        'stable_datum_provided':False,'registration_regions':'all SOR-retained scan with Huber weighting',
        'independent_external_validation':False}
    dump_json(out/'registration_quality.json',quality)
    if not passed: raise RuntimeError('Registration gate failed; inspect candidates before deviation analysis')
    np.savetxt(out/'T_scan_to_CAD.txt',T,fmt='%.12f'); np.save(out/'T_scan_to_CAD.npy',T)
    write_aligned_ply(ROOT/cfg['scan_path'],out/'scan_aligned.ply',T)
    dump_json(out/'config.json',cfg|{'T_scan_to_CAD':T.tolist()})
    logging.info('Selected registration %g mm: %s',best['resolution_mm'],metrics)


def analyze(cfg,out):
    import json
    quality=json.loads((out/'registration_quality.json').read_text())
    if not quality['passed']: raise RuntimeError('Registration must pass before analysis')
    T=np.load(out/'T_scan_to_CAD.npy'); mesh=get_mesh(out); scene=get_scene(mesh)
    records,_=read_ply(out/'scan_aligned.ply'); xyz=xyz_of(records)
    finite=np.isfinite(xyz).all(axis=1)
    pc=cloud(xyz[finite]).voxel_down_sample(cfg['analysis_voxel_mm']); del xyz,records; gc.collect()
    p=np.asarray(pc.points,dtype=np.float32)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4.0,max_nn=40))
    scan_normals=np.asarray(pc.normals)
    ans=closest_query(scene,p); q=ans['points']; n=ans['primitive_normals']
    n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    delta=p-q; signed=np.einsum('ij,ij->i',delta,n); dist=np.linalg.norm(delta,axis=1)
    tangential=np.sqrt(np.maximum(dist**2-signed**2,0))
    cosine=np.abs(np.einsum('ij,ij->i',scan_normals,n))
    uv=ans['primitive_uvs']; bary=np.column_stack([1-uv.sum(axis=1),uv])
    # Edge proximity is a flag; triangulation-internal edges are not all CAD feature edges.
    edge=bary.min(axis=1)<cfg['quality_edge_barycentric_margin']
    good=(dist<=cfg['quality_max_distance_mm']) & (tangential<=cfg['quality_max_tangential_mm']) & (cosine>=cfg['quality_min_normal_cosine'])
    mesh_info=json.loads((out/'mesh_quality.json').read_text()); outward=mesh_info['outward_normals_verified']
    df=pd.DataFrame({'scan_x_mm':p[:,0],'scan_y_mm':p[:,1],'scan_z_mm':p[:,2],
        'cad_closest_x_mm':q[:,0],'cad_closest_y_mm':q[:,1],'cad_closest_z_mm':q[:,2],
        'normal_x':n[:,0],'normal_y':n[:,1],'normal_z':n[:,2],
        'signed_deviation_mm':signed,'absolute_deviation_mm':np.abs(signed),
        'euclidean_distance_mm':dist,'tangential_residual_mm':tangential,
        'normal_agreement_abs_cosine':cosine,'triangle_id':ans['primitive_ids'],
        'near_triangle_edge':edge,'quality_mask':good})
    df.to_csv(out/'deviation_points.csv',index=False,float_format='%.7g')
    np.save(out/'deviation_points.npy',df.to_records(index=False))
    # Full resolution float32 deviation values, 1:1 with original input rows; no large CSV.
    raw,_=read_ply(out/'scan_aligned.ply')
    full=np.lib.format.open_memmap(out/'deviation_full_resolution.npy',mode='w+',dtype=np.dtype([
        ('signed_normal_mm','<f4'),('euclidean_distance_mm','<f4'),('triangle_id','<u4')]),shape=(len(raw),))
    for i in range(0,len(raw),100000):
        x=xyz_of(raw[i:i+100000]); valid=np.isfinite(x).all(axis=1)
        block=full[i:i+len(x)]; block['signed_normal_mm']=np.nan; block['euclidean_distance_mm']=np.nan
        block['triangle_id']=np.iinfo(np.uint32).max
        if not valid.any():continue
        a=closest_query(scene,x[valid]); d=x[valid]-a['points']
        block['signed_normal_mm'][valid]=np.einsum('ij,ij->i',d,a['primitive_normals'])
        block['euclidean_distance_mm'][valid]=np.linalg.norm(d,axis=1)
        block['triangle_id'][valid]=a['primitive_ids']
        if i%1000000==0: logging.info('Full-resolution deviation: %d/%d',min(i+100000,len(raw)),len(raw))
    full.flush()
    rows=[]
    for name,mask in [('all_1mm_voxels',np.ones(len(p),bool)),('quality_filtered_1mm_voxels',good)]:
        d=signed[mask]; ad=np.abs(d); e=dist[mask]; tol=cfg['example_tolerance_mm']
        if not len(d):continue
        rows.append({'scope':name,'n_points_analyzed':len(d),'mean_signed_deviation_mm':np.mean(d),
          'median_signed_deviation_mm':np.median(d),'median_absolute_deviation_mm':np.median(ad),
          'mean_absolute_deviation_mm':np.mean(ad),'rmse_signed_deviation_mm':np.sqrt(np.mean(d**2)),
          'p90_absolute_deviation_mm':np.percentile(ad,90),'p95_absolute_deviation_mm':np.percentile(ad,95),
          'p99_absolute_deviation_mm':np.percentile(ad,99),
          'fraction_abs_dev_le_example_tolerance':np.mean(ad<=tol),
          'fraction_signed_dev_gt_example_tolerance':np.mean(d>tol),
          'fraction_signed_dev_lt_minus_example_tolerance':np.mean(d< -tol),
          'example_tolerance_mm':tol, 'outward_normals_verified':outward,
          **{'euclidean_'+k:v for k,v in distance_stats(e).items()}})
    pd.DataFrame(rows).to_csv(out/'deviation_summary.csv',index=False)
    # CAD coverage is a proximity proxy, not an unseen-surface deviation measurement.
    cad=o3d.io.read_point_cloud(str(out/'cad_surface_sample.ply')); c=np.asarray(cad.points)
    coverage_dist=cKDTree(p).query(c,workers=4)[0]
    np.savez_compressed(out/'cad_coverage.npz',points=c.astype(np.float32),nearest_scan_distance_mm=coverage_dist.astype(np.float32))
    coverage={'definition':'fraction of uniform CAD area samples within radius of scan voxel centroids',
        'fraction_within_2mm':float(np.mean(coverage_dist<=2)),
        'fraction_within_5mm':float(np.mean(coverage_dist<=5)),
        'missing_regions_have_no_measured_deviation':True}
    dump_json(out/'coverage_summary.json',coverage)
    dump_json(out/'analysis_quality.json',{'analyzed_points':len(p),'quality_retained':int(good.sum()),
        'quality_fraction':float(good.mean()),'outward_normals_verified':outward,
        'near_triangle_edge_fraction':float(edge.mean()),'full_resolution_count':len(raw),
        'signed_definition':'dot(scan - CAD closest point, CAD triangle normal)',
        'full_resolution_row_mapping':'same row and attributes as original PLY; NaN for nonfinite XYZ',
        'tolerance_is_example_only':True,'approved_for_robot_execution':False})
    logging.info('Analysis complete; %d voxel points, %d quality retained; coverage=%s',len(p),good.sum(),coverage)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['prepare','register','analyze','plots','all'],default='all')
    args=parser.parse_args(); cfg=config(); out=ROOT/cfg['output_dir']; out.mkdir(exist_ok=True); log_setup(out)
    for name,fn in [('prepare',prepare),('register',register),('analyze',analyze)]:
        if args.stage in [name,'all']: fn(cfg,out)
    if args.stage in ['plots','all']:
        from visualization import plots
        plots(cfg,out)
if __name__=='__main__': main()
