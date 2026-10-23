"""Fused BNO08x orientation with the legacy scalar-last interface."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from mini_bdx_runtime.raw_imu import Imu as RawImu


class Imu(RawImu):
    def __init__(self, sampling_freq, user_pitch_bias=0, calibrate=False, upside_down=True):
        self.pitch_bias = user_pitch_bias
        super().__init__(sampling_freq, user_pitch_bias, calibrate, upside_down)

    def get_data(self, euler=False, mat=False):
        quat = super().get_data()["quaternion"]
        angles = R.from_quat([*quat[1:], quat[0]]).as_euler("xyz")
        angles[1] -= np.deg2rad(self.pitch_bias)
        orientation = R.from_euler("xyz", angles)
        if euler:
            return orientation.as_euler("xyz")
        if mat:
            return orientation.as_matrix()
        return orientation.as_quat()
