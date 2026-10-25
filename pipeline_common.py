"""Shared IO and geometric conventions: p_C = p_S @ R.T + t, all in mm."""
from pathlib import Path
import hashlib
import json
import logging
import numpy as np

ROOT = Path(__file__).resolve().parent
PLY_TYPES = {'char':'i1','int8':'i1','uchar':'u1','uint8':'u1',
             'short':'i2','int16':'i2','ushort':'u2','uint16':'u2',
             'int':'i4','int32':'i4','uint':'u4','uint32':'u4',
             'float':'f4','float32':'f4','double':'f8','float64':'f8'}

def config():
    return json.loads((ROOT/'config.json').read_text())

def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def log_setup(out):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(out/'run.log'), logging.StreamHandler()], force=True)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_ply(path):
    """Strict packed scalar binary PLY; reject unsupported layouts instead of guessing."""
    path=Path(path)
    with path.open('rb') as f:
        lines=[]
        while True:
            line=f.readline()
            if not line or f.tell()>1024*1024: raise ValueError('Incomplete/oversized PLY header')
            lines.append(line.decode('ascii').strip())
            if lines[-1]=='end_header': break
        offset=f.tell()
    if lines[0]!='ply': raise ValueError('Not a PLY file')
    formats=[s.split()[1] for s in lines if s.startswith('format ')]
    if formats not in [['binary_little_endian'],['binary_big_endian']]:
        raise ValueError('Only scalar binary PLY is supported')
    endian='<' if formats[0]=='binary_little_endian' else '>'
    props=[]; elements=[]; current=None; count=None
    for line in lines:
        a=line.split()
        if a[0]=='element':
            current=a[1]; elements.append((current,int(a[2])))
            if current=='vertex': count=int(a[2])
        elif a[0]=='property' and current=='vertex':
            if len(a)!=3 or a[1] not in PLY_TYPES: raise ValueError('Unsupported vertex property')
            props.append((a[2],endian+PLY_TYPES[a[1]]))
    if not elements or elements[0][0]!='vertex': raise ValueError('Vertex must be first element')
    if count is None or count<=0: raise ValueError('No vertices')
    dtype=np.dtype(props,align=False)
    if not all(n in dtype.names for n in ('x','y','z')): raise ValueError('Missing XYZ')
    expected=offset+count*dtype.itemsize
    if path.stat().st_size<expected: raise ValueError('Truncated PLY data')
    data=np.memmap(path,mode='r',dtype=dtype,offset=offset,shape=(count,))
    return data, {'header':lines,'offset':offset,'count':count,'stride_bytes':dtype.itemsize,
                  'properties':dtype.descr,'elements':elements,'path':str(path)}

def xyz_of(records):
    return np.column_stack([records[n] for n in ('x','y','z')]).astype(np.float32)

def transform_points(points,T):
    return np.asarray(points) @ T[:3,:3].T + T[:3,3]

def assert_rigid(T):
    assert T.shape==(4,4) and np.isfinite(T).all()
    np.testing.assert_allclose(T[3],[0,0,0,1],atol=1e-8)
    np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)
    assert abs(np.linalg.det(T[:3,:3])-1)<1e-5

def write_aligned_ply(source,path,T,chunk_size=100000):
    records,meta=read_ply(source)
    if len(meta['elements'])!=1: raise ValueError('Aligned writer only supports vertex-only PLY')
    lines=[s for s in meta['header'] if s!='end_header']
    lines+=['comment Transformed to CAD frame; XYZ in mm; all source scalar attributes preserved',
            'comment T_scan_to_CAD maps original scan XYZ into this frame','end_header']
    with Path(path).open('wb') as f:
        f.write(('\n'.join(lines)+'\n').encode('ascii'))
        for i in range(0,len(records),chunk_size):
            block=records[i:i+chunk_size].copy()
            p=transform_points(xyz_of(block),T)
            for k,n in enumerate(('x','y','z')): block[n]=p[:,k]
            block.tofile(f)

def cloud(points):
    import open3d as o3d
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points,dtype=np.float64)))

def subset(points,n,seed=7):
    if len(points)<=n:return np.asarray(points)
    return np.asarray(points)[np.random.default_rng(seed).choice(len(points),n,replace=False)]

def closest_query(scene,points,chunk_size=100000):
    import open3d as o3d
    results={k:[] for k in ('points','primitive_normals','primitive_ids','primitive_uvs')}
    for i in range(0,len(points),chunk_size):
        result=scene.compute_closest_points(o3d.core.Tensor(np.asarray(points[i:i+chunk_size],dtype=np.float32)),nthreads=4)
        for k in results: results[k].append(result[k].numpy())
    return {k:np.concatenate(v) for k,v in results.items()}

def distance_stats(dist):
    return {'n':len(dist),'median_mm':float(np.median(dist)),
      'rmse_mm':float(np.sqrt(np.mean(dist**2))),'p90_mm':float(np.percentile(dist,90)),
      'p95_mm':float(np.percentile(dist,95)),'p99_mm':float(np.percentile(dist,99)),
      'fraction_le_1mm':float(np.mean(dist<=1)), 'fraction_le_2mm':float(np.mean(dist<=2)),
      'fraction_le_5mm':float(np.mean(dist<=5))}
