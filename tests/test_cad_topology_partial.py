"""Incomplete face tessellation must not erase exact geometry or shift IDs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
cq = pytest.importorskip('cadquery')
import planning.cad_topology as module
from planning.cad_topology import CadTopology
from planning.segmentation import segment_points


def test_partial_mesh_preserves_face_ids_uv_and_exact_assignment(monkeypatch, tmp_path):
    faces = cq.Workplane('XY').box(10,8,2).val().Faces()
    original = module._triangulate_face
    missing_id = 2
    def triangulate(face, linear, angular):
        if face.isSame(faces[missing_id]):
            return {'status': 'missing_triangulation'}
        return original(face,linear,angular)
    monkeypatch.setattr(module,'_triangulate_face',triangulate)
    with pytest.warns(RuntimeWarning, match='tessellation is partial'):
        topology = CadTopology(faces)
    assert topology.catalog['mesh_status'] == 'partial'
    assert topology.catalog['meshed_face_count'] == 5
    assert topology.catalog['unmeshed_face_count'] == 1
    assert topology.catalog['exact_faces_preserved'] is True
    assert len(topology.faces) == 6
    assert len(topology.catalog['faces']) == 6
    assert topology.catalog['faces'][missing_id]['triangle_count'] == 0
    assert topology.catalog['faces'][missing_id]['mesh_status'] == 'missing_triangulation'
    assert topology.catalog['mesh_warnings'] == [{'face_id':2,'mesh_status':'missing_triangulation',
        'geometry_type':'PLANE','area_mm2':20.,'cad_valid':True}]
    assert set(topology.triangle_face_ids) == {0,1,3,4,5}
    assert np.all(topology.vertex_face_ids[topology.triangles] == topology.triangle_face_ids[:,None])
    assert topology.triangles.min() == 0
    assert topology.triangles.max() < len(topology.vertices)
    for p, uv, fid in zip(topology.vertices,topology.vertex_uv,topology.vertex_face_ids):
        assert np.allclose(p,topology.faces[fid].positionAt(*uv).toTuple())
    # Exact segmentation remains available on the face missing from the mesh.
    face = faces[missing_id]
    q = np.array(face.Center().toTuple()); n = np.array(face.normalAt(tuple(q)).toTuple())
    label = segment_points([q+.2*n],topology)
    assert label.loc[0,'valid']
    assert label.loc[0,'face_id'] == missing_id
    assert label.loc[0,'signed_deviation_mm'] == pytest.approx(.2)
    topology.save_mesh(tmp_path/'partial.npz')
    with np.load(tmp_path/'partial.npz') as saved:
        assert str(saved['mesh_status']) == 'partial'
        assert saved['unmeshed_face_ids'].tolist() == [missing_id]
        assert saved['exact_face_count'] == 6
