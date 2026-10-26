"""M8 candidate selection and non-executable pose export checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
pytest.importorskip("shapely")
pytest.importorskip("opencamlib")
from planning.geometry import ocl_frame

def test_local_protrusion_rejects_uniform_offset():
    import pandas as pd
    from planning.candidates import classify
    import json
    cfg=json.loads((Path(__file__).resolve().parents[1]/'planning.example.json').read_text())
    x,y=np.meshgrid(np.arange(-5,5.1,.5),np.arange(-5,5.1,.5));x=x.ravel();y=y.ravel()
    frame=pd.DataFrame({'cad_closest_x_mm':x,'cad_closest_y_mm':y,'cad_closest_z_mm':0.,
                        'normal_x':0.,'normal_y':0.,'normal_z':-1.,'signed_deviation_mm':.8,'quality_mask':True})
    result=classify(frame,cfg)
    assert not result.candidate_mask.any()
    frame['signed_deviation_mm']=.1+.8*np.exp(-(x*x+y*y)/1.5)
    result=classify(frame,cfg)
    assert result.candidate_mask.any()
    assert np.all(np.sqrt(x[result.candidate_mask]**2+y[result.candidate_mask]**2)<2)


def test_export_preserves_clearance_and_keeps_execution_disabled():
    from planning.paths import export_poses
    import json
    cfg=json.loads((Path(__file__).resolve().parents[1]/'planning.example.json').read_text())
    stroke={'center':np.array([[0,0,-2.],[1,0,-2.]]),'contact':np.array([[0,0,0.],[1,0,0.]]),
            'normal':np.array([[0,0,-1.],[0,0,-1.]]),'analysis_rows':np.array([4,5]),
            'clearance':np.zeros(2),'region_id':0}
    poses=export_poses([stroke],ocl_frame([0,0,-1]),10.,cfg)
    assert poses.iloc[0].z_mm == -10
    assert poses.iloc[-1].z_mm == -10
    assert (poses.phase=='nominal_contact').sum()==2
    assert not poses.execution_allowed.any()
    assert poses.removal_depth_mm.isna().all()
    assert np.max(np.linalg.norm(np.diff(poses[['x_mm','y_mm','z_mm']].to_numpy(),axis=0),axis=1))<=1.00001


def test_occluding_surface_is_not_mistaken_for_requested_lower_region():
    import json,pandas as pd
    from planning.paths import project_region
    cfg=json.loads((Path(__file__).resolve().parents[1]/'planning.example.json').read_text())
    class TwoPlanes:
        def crop(self,*args):
            p=np.array([[-10,-10,0],[10,-10,0],[10,10,0],[-10,10,0]],float)
            bottom=p[[[0,1,2],[0,2,3]]]
            return np.concatenate([bottom,bottom+[0,0,4.]])
    x,y=np.meshgrid(np.arange(-2,3),np.arange(-2,3));n=x.size
    region=pd.DataFrame({'cad_closest_x_mm':x.ravel(),'cad_closest_y_mm':y.ravel(),
                         'cad_closest_z_mm':0.,'normal_x':0.,'normal_y':0.,'normal_z':1.,'analysis_row':np.arange(n)})
    strokes,stats=project_region(region,TwoPlanes(),None,np.eye(3),0.,cfg)
    assert not strokes
    assert stats['accepted_waypoints']==0


def test_export_validation_rejects_bad_quaternion():
    from planning.paths import export_poses,validate_exported_poses
    import json
    cfg=json.loads((Path(__file__).resolve().parents[1]/'planning.example.json').read_text())
    stroke={'center':np.array([[0,0,-2.],[1,0,-2.]]),'contact':np.array([[0,0,0.],[1,0,0.]]),
            'normal':np.array([[0,0,-1.],[0,0,-1.]]),'analysis_rows':np.array([4,5]),
            'clearance':np.zeros(2),'region_id':0}
    poses=export_poses([stroke],ocl_frame([0,0,-1]),10.,cfg)
    assert validate_exported_poses(poses,cfg)['execution_disabled']
    poses.loc[0,'qw']=4
    with pytest.raises(ValueError,match='quaternion'):
        validate_exported_poses(poses,cfg)
