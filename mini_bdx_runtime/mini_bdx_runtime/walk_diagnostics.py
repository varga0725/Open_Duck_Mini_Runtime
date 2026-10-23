"""Bounded, in-memory policy diagnostics; disk writes only when walking exits."""
from collections import deque
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

STAGES = ('observation', 'policy', 'motor_write', 'telemetry')


class WalkDiagnostics:
    def __init__(self, path, control_hz, model_path, capacity=3000):
        self.path = Path(path)
        self.rows = deque(maxlen=capacity)
        self.metadata = {
            'format': 'open-duck-policy-diagnostics-v2',
            'motor_targets_semantics': 'actual actuator command after ctrlrange clamp',
            'policy_motor_targets_semantics': 'rate-limited targets before actuator ctrlrange clamp',
            'wall_start_time': time.time(),
            'monotonic_start_time': time.perf_counter(),
            'control_hz': control_hz,
            'model_path': str(model_path),
            'stage_names': STAGES,
            'observation_layout': 'gyro3,biased_accel3,commands7,joint_pos_minus_default14,'
                'joint_vel_times_0.05_14,action_history42,motor_targets14,contacts2,phase2',
            'quaternion_order': 'wxyz',
        }
        self.total_ticks = 0

    def append(self, timestamp, obs, action, motor_targets, quaternion, stages,
               policy_motor_targets=None):
        self.rows.append((timestamp, np.array(obs, dtype=np.float32, copy=True),
                          np.array(action, dtype=np.float32, copy=True),
                          np.array(motor_targets, dtype=np.float32, copy=True),
                          np.array(quaternion, dtype=np.float32, copy=True), tuple(stages),
                          np.array(motor_targets if policy_motor_targets is None else policy_motor_targets,
                                   dtype=np.float32, copy=True)))
        self.total_ticks += 1

    def save(self):
        if not self.rows:
            return
        columns = list(zip(*self.rows))
        self.metadata['total_ticks'] = self.total_ticks
        self.metadata['retained_ticks'] = len(self.rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix='.walk-', suffix='.npz')
        try:
            with os.fdopen(fd, 'wb') as output:
                np.savez_compressed(
                    output, timestamp_monotonic=np.asarray(columns[0]),
                    obs=np.asarray(columns[1]), action=np.asarray(columns[2]),
                    motor_targets=np.asarray(columns[3]), quaternion=np.asarray(columns[4]),
                    stage_seconds=np.asarray(columns[5]),
                    policy_motor_targets=np.asarray(columns[6]),
                    metadata=np.array(json.dumps(self.metadata)),
                )
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
