"""Critical position/orientation checks for user-confirmed robot-base scans."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation
from planning.frames import (cad_to_base_transform, contact_poses_in_base,
                             transform_frame_table, validate_base_contact_poses)


def _cad():
    frame = pd.DataFrame({'path_id': ['known']*3, 'sample_index': [0,1,2],
        'segment_id': [0]*3, 'face_id': [7]*3, 'frame': ['CAD']*3,
        'x_mm': [1.,2.,3.], 'y_mm': [2.]*3, 'z_mm': [3.]*3,
        'position_reference': ['surface_contact_point']*3, 'execution_allowed': [False]*3,
        'normal_x': [0.]*3, 'normal_y': [0.]*3, 'normal_z': [-1.]*3})
    for axis_index, axis in enumerate('xyz'):
        for component_index, component in enumerate('xyz'):
            frame[f'tool_{axis}_{component}'] = float(axis_index == component_index)
    frame[['qx','qy','qz','qw']] = np.tile([0.,0.,0.,1.],(3,1))
    return frame


def test_known_cad_to_base_moves_position_rotates_axis_without_translation():
    base_from_cad = np.eye(4)
    base_from_cad[:3,:3] = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
    base_from_cad[:3,3] = [10.,20.,30.]
    transform = np.linalg.inv(base_from_cad)
    cad = _cad()
    base, matrices = contact_poses_in_base(cad,transform)
    np.testing.assert_allclose(base[['x_mm','y_mm','z_mm']],[[8.,21.,33.],[8.,22.,33.],[8.,23.,33.]])
    np.testing.assert_allclose(base[['tool_x_x','tool_x_y','tool_x_z']],np.tile([0.,1.,0.],(3,1)))
    np.testing.assert_allclose(base[['normal_x','normal_y','normal_z']],np.tile([0.,0.,-1.],(3,1)))
    np.testing.assert_allclose(Rotation.from_quat(base[['qx','qy','qz','qw']]).as_matrix(),matrices[:,:3,:3])
    np.testing.assert_allclose(cad_to_base_transform(transform),base_from_cad)
    assert base.frame.eq('ROBOT_BASE').all()
    assert not any('ball' in key for key in base)
    assert validate_base_contact_poses(cad,base,transform)['passed']
    with pytest.raises(ValueError,match='source frame'):
        contact_poses_in_base(base,transform)
    cad['cad_x_mm'],cad['cad_y_mm'],cad['cad_z_mm'] = cad.x_mm,cad.y_mm,cad.z_mm
    moved = transform_frame_table(cad,base_from_cad)
    np.testing.assert_allclose(moved[['reference_x_mm','reference_y_mm','reference_z_mm']],base[['x_mm','y_mm','z_mm']])
    assert 'cad_x_mm' not in moved


def test_base_validation_detects_corrupt_pose_and_handles_empty_exports():
    cad = _cad()
    base,_ = contact_poses_in_base(cad,np.eye(4))
    broken=base.copy();broken.loc[0,'x_mm']+=1.
    with pytest.raises(ValueError,match='validation failed'):
        validate_base_contact_poses(cad,broken,np.eye(4))
    empty,transforms = contact_poses_in_base(cad.iloc[:0],np.eye(4))
    assert transforms.shape == (0,4,4)
    assert validate_base_contact_poses(cad.iloc[:0],empty,np.eye(4))['empty']
