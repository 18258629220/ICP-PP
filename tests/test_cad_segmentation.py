"""CAD provenance and trimmed-face assignment against synthetic known geometry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
cq = pytest.importorskip('cadquery')
from planning.cad_topology import CadTopology
from planning.segmentation import segment_points, FaceBoundsIndex, segmentation_config


def test_true_shared_edges_and_mesh_face_uv_provenance(tmp_path):
    solid = cq.Workplane('XY').circle(5).extrude(4).val().translate((13, -7, 2))
    topology = CadTopology(solid.Faces(), linear_deflection_mm=.02)
    assert len(topology.catalog['faces']) == 3
    assert len(topology.triangles) > 10*len(topology.faces)
    assert topology.catalog['relative_deflection'] is False
    assert len(topology.catalog['adjacency']) == 2  # circular rim edges; periodic seam is self-adjacent
    assert np.all(topology.vertex_face_ids[topology.triangles] == topology.triangle_face_ids[:, None])
    for vertex, uv, face_id in zip(topology.vertices, topology.vertex_uv, topology.vertex_face_ids):
        exact = topology.faces[face_id].positionAt(*uv).toTuple()
        assert np.allclose(vertex, exact, atol=1e-8)
    for face_id, face in enumerate(topology.faces):
        tri = topology.triangles[topology.triangle_face_ids == face_id][0]
        a, b, c = topology.vertices[tri]
        normal = np.asarray(face.normalAt(tuple((a+b+c)/3)).toTuple())
        assert np.dot(np.cross(b-a, c-a), normal) > 0
    path = tmp_path/'mesh.npz'
    topology.save_mesh(path)
    with np.load(path) as data:
        assert np.array_equal(data['triangle_face_ids'], topology.triangle_face_ids)
        assert np.array_equal(data['vertex_uv'], topology.vertex_uv)
    # Coincidence is not a shared topological edge.
    face = cq.Workplane('XY').rect(5, 4).extrude(1).faces('>Z').val()
    topology = CadTopology([face, face.copy()])
    assert not topology.catalog['adjacency']


def test_exact_plane_projection_preserves_rows_sign_and_source_indices():
    solid = cq.Workplane('XY').box(10, 8, 2).val()
    topology = CadTopology(solid.Faces())
    points = np.array([[0, 0, 1.2], [0, 0, -1.3], [0, 0, 10], [np.nan, 0, 0]])
    result = segment_points(points, topology, source_indices=np.array([42, 9, 99, 12]))
    assert result.source_index.tolist() == [42, 9, 99, 12]
    assert result.valid.tolist() == [True, True, False, False]
    assert result.loc[0, 'signed_deviation_mm'] == pytest.approx(.2)
    assert result.loc[1, 'signed_deviation_mm'] == pytest.approx(.3)
    for i in (0, 1):
        row = result.iloc[i]
        q = topology.faces[row.face_id].positionAt(row.u, row.v).toTuple()
        assert np.allclose(q, row[['projected_x_mm', 'projected_y_mm', 'projected_z_mm']].to_numpy(float))
        assert row.boundary_distance_mm == pytest.approx(4.)
    assert result.loc[2, 'status'] == 'no_candidate'
    assert result.loc[3, 'status'] == 'nonfinite_point'


def test_hole_and_trim_extension_are_unknown_instead_of_snapping_to_rim():
    face = cq.Workplane('XY').rect(10, 8).circle(.2).extrude(1).faces('>Z').val()
    topology = CadTopology([face])
    result = segment_points([[0,0,1.05],[.24,0,1.05],[1,0,1.05],[5.1,0,1.05]], topology,
                            {'max_distance_mm': .4, 'boundary_margin_mm': .1})
    assert not result.loc[0, 'valid']
    assert result.loc[0, 'candidate_face_id'] == -1
    assert result.loc[1, 'status'] == 'boundary_uncertain'
    assert result.loc[1, 'boundary_distance_mm'] == pytest.approx(.04, abs=1e-7)
    assert result.loc[2, 'valid']
    assert not result.loc[3, 'valid']


def test_thin_wall_competing_faces_require_unambiguous_or_oriented_evidence():
    solid = cq.Workplane('XY').box(10,8,.4).val()
    topology = CadTopology(solid.Faces())
    points = [[0,0,0],[0,0,.25],[0,0,-.25]]
    unknown = segment_points(points, topology, {'max_distance_mm': .5})
    assert unknown.loc[0, 'status'] == 'ambiguous_faces'
    assert unknown.loc[0, 'face_id'] == -1
    assert unknown.valid.tolist() == [False, True, True]
    oriented = segment_points([[0,0,0]], topology,
                              {'max_distance_mm': .5, 'normal_orientation_known': True},
                              normals_C=[[0,0,1]])
    assert oriented.loc[0, 'valid']
    assert oriented.loc[0, 'normal_z'] == pytest.approx(1.)
    unoriented = segment_points([[0,0,0]], topology, {'max_distance_mm': .5}, normals_C=[[0,0,1]])
    assert unoriented.loc[0, 'status'] == 'ambiguous_faces'


def test_periodic_cylinder_seam_is_not_a_physical_boundary():
    cylinder = cq.Workplane('XY').circle(5).extrude(8).faces('%CYLINDER').val()
    topology = CadTopology([cylinder])
    result = segment_points([[5.1, 0, 4]], topology)
    assert result.loc[0, 'valid']
    assert result.loc[0, 'boundary_distance_mm'] == pytest.approx(4.)
    assert result.loc[0, 'normal_x'] == pytest.approx(1.)


def test_bvh_candidates_match_conservative_aabb_distances_and_overflow_rejects():
    rng = np.random.default_rng(11)
    low = rng.uniform(-20,20,(101,3))
    high = low+rng.uniform(.1,10,(101,3))
    bounds = np.column_stack([low, high]); index = FaceBoundsIndex(bounds)
    for p in rng.uniform(-20,20,(30,3)):
        distances = np.linalg.norm(np.maximum(np.maximum(low-p,p-high),0),axis=1)
        assert index.query(p,2.) == np.flatnonzero(distances <= 2.).tolist()
    topology = CadTopology(cq.Workplane('XY').box(2,2,2).val().Faces())
    result = segment_points([[0,0,0]],topology,{'max_distance_mm':2.,'max_candidates':1})
    assert result.loc[0,'status'] == 'candidate_overflow'
    assert result.loc[0,'exact_projection_count'] == 0


@pytest.mark.parametrize('cfg', [{'max_distance_mm':0},{'ambiguity_margin_mm':-.1},
                                {'max_candidates':0},{'max_normal_angle_deg':np.nan}])
def test_invalid_segmentation_config_rejected(cfg):
    with pytest.raises(ValueError):
        segmentation_config(cfg)
