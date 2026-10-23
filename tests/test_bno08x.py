"""Regression checks for the robot's confirmed BNO08x mounting."""
import time
from threading import Lock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mini_bdx_runtime.raw_imu import Imu


@pytest.mark.parametrize('upside_down,expected', [(False, [2, -1, 3]), (True, [2, 1, -3])])
def test_robot_frame_vectors_match_orientation(upside_down, expected):
    reader = Imu.__new__(Imu)
    reader._set_mounting(upside_down)
    reader.x_offset = 0
    rotation = Rotation.from_euler('xyz', [.3, -.5, .8])
    data = reader._transform_sample([1, 2, 3], [1, 2, 3], rotation.as_quat())
    np.testing.assert_allclose(data['gyro'], expected)
    np.testing.assert_allclose(data['accelero'], expected)
    q = data['quaternion']
    world_from_robot = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
    np.testing.assert_allclose(world_from_robot @ reader._mount, rotation.as_matrix(), atol=1e-14)
    assert np.linalg.norm(q) == pytest.approx(1)


@pytest.mark.parametrize('quat', [(0, 0, 0, 0), (float('nan'), 0, 0, 1), (None, 0, 0, 1)])
def test_invalid_sensor_orientation_is_rejected(quat):
    reader = Imu.__new__(Imu)
    reader._set_mounting(False)
    reader.x_offset = 0
    with pytest.raises(ValueError):
        reader._transform_sample([0, 0, 0], [0, 0, 9.81], quat)


def test_stale_readings_are_rejected():
    reader = Imu.__new__(Imu)
    reader.sampling_freq = 50
    reader._lock = Lock()
    reader._last_sample_at = time.monotonic() - 2
    with pytest.raises(RuntimeError, match='stale'):
        reader.get_data()
