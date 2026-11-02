"""Create a deterministic synthetic CAD/scan case; no private inputs or ICP run.

The saved identity transform is synthetic ground truth, not a registration
accuracy claim. Every generated scan row has an independent analytic position
and normal in ground_truth.csv. All lengths are millimetres.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_common import dump_json, read_ply, sha256, xyz_of


def _write_scan(path, points):
    dtype = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
                      ('intensity', 'u1'), ('event_id', '<u2'),
                      ('camera_column', '<u2')])
    records = np.zeros(len(points), dtype=dtype)
    for column, axis in enumerate('xyz'):
        records[axis] = points[:, column]
    records['intensity'] = 200
    records['event_id'] = np.arange(len(points)) % 65536
    records['camera_column'] = np.arange(len(points)) % 1024
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment Synthetic analytic surface; XYZ in mm\n'
              f'element vertex {len(points)}\n'
              'property double x\nproperty double y\nproperty double z\n'
              'property uchar intensity\nproperty ushort event_id\n'
              'property ushort camera_column\nend_header\n')
    path.write_bytes(header.encode('ascii')+records.tobytes())


def create_case(output_dir, sampling_mm=.3):
    """Write a holed plate and cylinder with analytic shape deviations.

    Refuse a nonempty directory so independent runs remain comparable.
    """
    if not np.isfinite(sampling_mm) or sampling_mm <= 0 or sampling_mm > 1.:
        raise ValueError('sampling_mm must be finite, positive and <=1 mm')
    dest = Path(output_dir).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Use a new empty synthetic case directory')
    dest.mkdir(parents=True, exist_ok=True)
    import cadquery as cq
    import open3d as o3d
    import pandas as pd
    from planning.cad_reference import load_faces

    plate = cq.Workplane('XY').rect(20, 16).circle(2).extrude(2).val()
    cylinder = cq.Workplane('XY').center(20, 0).circle(4).extrude(8).val()
    compound = cq.Compound.makeCompound([plate, cylinder])
    step = dest/'model.step'
    cq.exporters.export(compound, str(step))
    # Identify truth IDs after STEP import, matching the production convention.
    faces = load_faces(step)
    plate_face_id = next(i for i, f in enumerate(faces)
                         if f.geomType() == 'PLANE' and abs(f.Center().z-2) < 1e-7
                         and abs(f.Center().x) < 1)
    cylinder_face_id = next(i for i, f in enumerate(faces)
                            if f.geomType() == 'CYLINDER' and f.Center().x > 15)

    gx = np.linspace(-9.6, 9.6, int(np.ceil(19.2/sampling_mm))+1)
    gy = np.linspace(-7.6, 7.6, int(np.ceil(15.2/sampling_mm))+1)
    x, y = np.meshgrid(gx, gy)
    x, y = x.ravel(), y.ravel()
    # Leave a finite observation gap around the exact CAD hole.
    keep = np.hypot(x, y) >= 2.15
    x, y = x[keep], y[keep]
    d = .15+.015*x+.008*y+.0008*x*x
    plate_nominal = np.column_stack([x, y, np.full(len(x), 2.)])
    plate_points = plate_nominal+np.column_stack([np.zeros((len(x), 2)), d])
    plate_normals = np.column_stack([-(.015+.0016*x), np.full(len(x), -.008), np.ones(len(x))])
    plate_normals /= np.linalg.norm(plate_normals, axis=1)[:, None]

    theta = np.linspace(-1.05, 1.05, int(np.ceil(8.4/sampling_mm))+1)
    z = np.linspace(.5, 7.5, int(np.ceil(7/sampling_mm))+1)
    theta, z = np.meshgrid(theta, z)
    theta, z = theta.ravel(), z.ravel()
    c, s = np.cos(theta), np.sin(theta)
    cd = .12+.04*np.sin(2*theta)+.005*(z-4)
    r, drdt = 4+cd, .08*np.cos(2*theta)
    cylinder_nominal = np.column_stack([20+4*c, 4*s, z])
    cylinder_points = np.column_stack([20+r*c, r*s, z])
    du = np.column_stack([drdt*c-r*s, drdt*s+r*c, np.zeros(len(z))])
    dv = np.column_stack([.005*c, .005*s, np.ones(len(z))])
    cylinder_normals = np.cross(du, dv)
    cylinder_normals /= np.linalg.norm(cylinder_normals, axis=1)[:, None]

    points = np.vstack([plate_points, cylinder_points])
    nominal = np.vstack([plate_nominal, cylinder_nominal])
    normals = np.vstack([plate_normals, cylinder_normals])
    face_ids = np.r_[np.full(len(plate_points), plate_face_id),
                     np.full(len(cylinder_points), cylinder_face_id)]
    scan = dest/'scan.ply'
    _write_scan(scan, points)
    loaded, _ = read_ply(scan)
    np.testing.assert_allclose(xyz_of(loaded), points, atol=2e-6)
    truth = pd.DataFrame({'source_index': np.arange(len(points)), 'face_id': face_ids,
                          'analytic_normal_offset_mm': np.r_[d, cd]})
    for j, axis in enumerate('xyz'):
        truth[f'nominal_{axis}_mm'] = nominal[:, j]
        truth[f'scan_{axis}_mm'] = points[:, j]
        truth[f'actual_normal_{axis}'] = normals[:, j]
    truth.to_csv(dest/'ground_truth.csv', index=False)

    upstream = dest/'upstream'
    upstream.mkdir()
    transform = np.eye(4)
    np.save(upstream/'T_scan_to_CAD.npy', transform)
    dump_json(upstream/'registration_quality.json', {
        'selected_candidate': {'T_final': transform.tolist()},
        'synthetic_identity_ground_truth': True,
        'registration_algorithm_executed': False})
    dump_json(upstream/'cad_metadata.json', {'sha256': sha256(step), 'units': 'mm',
                                            'synthetic': True})
    dump_json(upstream/'scan_metadata.json', {'sha256': sha256(scan), 'units': 'mm',
                                             'synthetic': True, 'count': len(points)})
    vertices, triangles = compound.tessellate(.05, .1)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([v.toTuple() for v in vertices]),
                                   o3d.utility.Vector3iVector(triangles))
    mesh.compute_triangle_normals()
    if not o3d.io.write_triangle_mesh(str(upstream/'cad_mesh_clean.ply'), mesh, write_ascii=False):
        raise RuntimeError('Failed writing synthetic mesh')
    config = json.loads((ROOT/'config.example.json').read_text())
    config.update(step_path=str(step), scan_path=str(scan), output_dir=str(upstream))
    dump_json(dest/'config.json', config)
    mapping = json.loads((ROOT/'cad_mapping.example.json').read_text())
    mapping.update(approach_outward_C=[0, 0, 1], scan_voxel_mm=.25,
                   normal_mode='scan', reference_support_distance_mm=.8,
                   max_normal_offset_mm=.8, max_support_gap_mm=.5)
    dump_json(dest/'mapping_config.json', mapping)
    surface = json.loads((ROOT/'surface_workflow.example.json').read_text())
    surface['mapping'].update(normal_mode='scan', approach_outward_C=[0,0,1],
        scan_voxel_mm=.25, scan_sigma_mm=0., registration_sigma_mm=0.)
    surface['coverage'].update(face_ids=[plate_face_id, cylinder_face_id])
    # Exact synthetic geometry/identity are known here, unlike actual metrology.
    # Positive layers are previews with analytic material-side truth only.
    surface['process'].update(material_side_confirmed=True, boundary_position_bound_mm=0.,
                              layer_depths_mm=[0., .05, .1])
    dump_json(dest/'surface_config.json', surface)
    manifest = {
        'synthetic': True, 'seed': None, 'randomness_used': False,
        'units': 'mm', 'scan_points': len(points), 'sampling_mm': sampling_mm,
        'plate_face_id': plate_face_id, 'cylinder_face_id': cylinder_face_id,
        'plate_points': len(plate_points), 'cylinder_points': len(cylinder_points),
        'plate_height_model': 'z=2+.15+.015*x+.008*y+.0008*x^2',
        'cylinder_radius_model': 'r=4+.12+.04*sin(2*theta)+.005*(z-4)',
        'expected_segmentation': 'interior accepted points match ground_truth.face_id; trim/boundary ambiguity may reject',
        'expected_mapping': 'plate quadratic fit matches height and derivative; cylinder local polynomial is approximate',
        'registration_algorithm_executed': False, 'physical_accuracy_verified': False,
        'input_sha256': {'step': sha256(step), 'scan': sha256(scan)},
    }
    dump_json(dest/'case_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sampling-mm', type=float, default=.3)
    args = parser.parse_args()
    manifest = create_case(args.output_dir, args.sampling_mm)
    print(json.dumps({'output_dir': str(args.output_dir.resolve()), **manifest}, indent=2))


if __name__ == '__main__':
    main()
