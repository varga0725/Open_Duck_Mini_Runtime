"""Hardware-independent settings for the 101D walking policy."""
import json
from pathlib import Path

import numpy as np

JOINT_NAMES = (
    'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
    'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
    'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
)
OBS_BLOCKS = (
    ('gyro', 3), ('accelerometer', 3), ('commands', 7), ('joint_pos', 14),
    ('joint_vel', 14), ('last_action', 14), ('last_last_action', 14),
    ('last_last_last_action', 14), ('motor_targets', 14), ('feet_contacts', 2),
    ('imitation_phase', 2),
)


class PolicySettings:
    def __init__(self, path):
        self.path = Path(path).resolve()
        data = json.loads(self.path.read_text())
        if data['actuator_names'] != list(JOINT_NAMES):
            raise ValueError('Policy actuator order does not match the 14-joint runtime')
        if [block[1] for block in data['obs_101_layout']] != [n for _, n in OBS_BLOCKS]:
            raise ValueError('Policy observation layout does not match the 101D runtime')
        self.default_pose = np.asarray(data['default_pose'], dtype=float)
        if self.default_pose.shape != (14,) or not np.isfinite(self.default_pose).all():
            raise ValueError('default_pose must contain 14 finite values')
        for key in ('action_scale', 'control_hz', 'max_motor_velocity_rad_s'):
            value = float(data[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
            setattr(self, key, value)
        self.period = data['phase']['nb_steps_in_period']
        if type(self.period) is not int or self.period <= 0:
            raise ValueError('nb_steps_in_period must be a positive integer')
        # This bundle explicitly describes the training-time acceleration bias.
        # Keep it local to policy observations, rather than changing IMU telemetry.
        accel_block = data['obs_101_layout'][1][0]
        if accel_block != 'accelerometer (+1.3 on x, as mujoco_infer.py)':
            raise ValueError('Unsupported policy accelerometer convention')
        self.accel_x_bias = 1.3
        limits = data.get('actuator_ctrlrange_rad')
        self.actuator_ctrlrange = None
        if limits is not None:
            if set(limits) != set(JOINT_NAMES):
                raise ValueError('actuator_ctrlrange_rad must name all 14 policy joints')
            self.actuator_ctrlrange = np.asarray([limits[n] for n in JOINT_NAMES], dtype=float)
            if (self.actuator_ctrlrange.shape != (14, 2)
                    or not np.isfinite(self.actuator_ctrlrange).all()
                    or np.any(self.actuator_ctrlrange[:, 0] >= self.actuator_ctrlrange[:, 1])):
                raise ValueError('actuator_ctrlrange_rad must contain finite ordered bounds')

    def actuator_targets(self, policy_targets):
        """Match MuJoCo's ctrlrange clamp without altering recurrent policy state."""
        targets = np.array(policy_targets, dtype=float, copy=True)
        if targets.shape != (14,) or not np.isfinite(targets).all():
            raise ValueError('Actuator targets must contain 14 finite values')
        if self.actuator_ctrlrange is not None:
            targets = np.clip(targets, self.actuator_ctrlrange[:, 0], self.actuator_ctrlrange[:, 1])
        return targets

    def targets(self, action, previous):
        action = np.asarray(action, dtype=float)
        previous = np.asarray(previous, dtype=float)
        if (action.shape != (14,) or previous.shape != (14,)
                or not np.isfinite(action).all() or not np.isfinite(previous).all()):
            raise ValueError('Action and previous targets must contain 14 finite values')
        limit = self.max_motor_velocity_rad_s / self.control_hz
        desired = self.default_pose + self.action_scale * action
        return previous + np.clip(desired - previous, -limit, limit)

    def phase(self, tick):
        angle = 2 * np.pi * (tick % self.period) / self.period
        return np.array([np.cos(angle), np.sin(angle)])
