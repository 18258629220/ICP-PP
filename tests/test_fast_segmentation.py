"""One compact safeguard for batched labels: trims, layers and source order."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
cq = pytest.importorskip('cadquery')
pytest.importorskip('open3d')
from planning.cad_topology import CadTopology
from planning.fast_segmentation import MeshFaceSegmenter


def test_mesh_labels_keep_holes_layers_and_original_source_order():
    face = cq.Workplane('XY').rect(10,8).circle(.5).extrude(1).faces('>Z').val()
    topology = CadTopology([face])
    segmenter = MeshFaceSegmenter(topology,boundary_sampling_mm=.1,mesh_error_margin_mm=.01)
    points = [[2,1,1.1],[0,0,1.1],[4.99,0,1.1],[1,2,1.2]]
    labels = segmenter.segment(points,{'max_distance_mm':.4},source_indices=np.array([7,3,90,11]))
    assert labels.source_index.tolist() == [7,3,90,11]
    assert labels.valid.tolist() == [True,False,False,True]
    assert labels.face_id.tolist() == [0,-1,-1,0]
    assert labels.geometry_method.eq('face_preserving_mesh').all()
    assert labels.exact_projection_count.eq(0).all()
    np.testing.assert_allclose(labels.loc[[0,3],'signed_deviation_mm'],[.1,.2],atol=1e-6)
    wall = CadTopology(cq.Workplane('XY').box(10,8,.4).val().Faces())
    segmenter = MeshFaceSegmenter(wall,boundary_sampling_mm=.2,mesh_error_margin_mm=.01)
    labels = segmenter.segment([[0,0,0],[0,0,.3]],{'max_distance_mm':.5})
    assert not labels.loc[0,'valid']
    assert labels.loc[0,'status'] == 'mesh_face_competition'
    assert labels.loc[1,'valid']
