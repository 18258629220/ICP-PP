"""Checks that catch metrology-relevant IO, transform and sign errors."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from pipeline_common import read_ply,xyz_of,transform_points,write_aligned_ply,assert_rigid,closest_query

def make_ply(path,endian='<',count=3,truncate=False):
    dtype=np.dtype([(n,endian+t) for n,t in [('x','f4'),('y','f4'),('z','f4'),('intensity','u1'),('event_id','u2'),('camera_column','u2')]])
    rows=np.zeros(count,dtype=dtype)
    rows['x']=np.arange(count)+1; rows['y']=17; rows['z']=-3
    rows['intensity']=255; rows['event_id']=4123; rows['camera_column']=1234
    fmt='binary_little_endian' if endian=='<' else 'binary_big_endian'
    header=f'ply\nformat {fmt} 1.0\nelement vertex {count}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar intensity\nproperty ushort event_id\nproperty ushort camera_column\nend_header\n'
    path.write_bytes(header.encode()+rows.tobytes()[:(-1 if truncate else None)])
    return rows

@pytest.mark.parametrize('endian',['<','>'])
def test_interleaved_ply_and_transform_preserves_attributes(tmp_path,endian):
    src=tmp_path/'in.ply'; rows=make_ply(src,endian)
    loaded,meta=read_ply(src); assert meta['stride_bytes']==17
    np.testing.assert_allclose(xyz_of(loaded),[[1,17,-3],[2,17,-3],[3,17,-3]])
    T=np.array([[0,-1,0,10],[1,0,0,20],[0,0,1,30],[0,0,0,1.]])
    assert_rigid(T); dst=tmp_path/'out.ply'; write_aligned_ply(src,dst,T,chunk_size=2)
    result,_=read_ply(dst)
    np.testing.assert_allclose(xyz_of(result),[[-7,21,27],[-7,22,27],[-7,23,27]])
    np.testing.assert_allclose(transform_points(xyz_of(result),np.linalg.inv(T)),xyz_of(loaded),atol=1e-6)
    for name in ['intensity','event_id','camera_column']: np.testing.assert_array_equal(result[name],rows[name])

def test_truncation_rejected(tmp_path):
    src=tmp_path/'in.ply';make_ply(src,truncate=True)
    with pytest.raises(ValueError,match='Truncated'): read_ply(src)

def test_box_known_signed_offsets_and_edge_distance():
    import open3d as o3d
    mesh=o3d.geometry.TriangleMesh.create_box(10,10,10)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    points=np.array([[5,5,12],[5,5,8],[12,12,5]],np.float32)
    ans=closest_query(scene,points); delta=points-ans['points']
    normal=np.einsum('ij,ij->i',delta,ans['primitive_normals'])
    np.testing.assert_allclose(normal[:2],[2,-2],atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(delta,axis=1),[2,2,np.sqrt(8)],atol=1e-6)
    assert abs(normal[2]) < np.linalg.norm(delta[2])

def test_point_to_plane_recovers_known_transform():
    import open3d as o3d
    from pipeline_common import cloud
    mesh=o3d.geometry.TriangleMesh.create_box(10,15,23)
    o3d.utility.random.seed(11); target=mesh.sample_points_uniformly(8000,use_triangle_normal=True)
    T=np.eye(4);T[:3,:3]=o3d.geometry.get_rotation_matrix_from_xyz([.02,-.03,.01]);T[:3,3]=[.2,-.3,.4]
    source=cloud(transform_points(np.asarray(target.points),np.linalg.inv(T)))
    reg=o3d.pipelines.registration
    fit=reg.registration_icp(source,target,2,np.eye(4),reg.TransformationEstimationPointToPlane(),reg.ICPConvergenceCriteria(max_iteration=100))
    np.testing.assert_allclose(fit.transformation,T,atol=1e-5)
