import json
import numpy as np
from mini_bdx_runtime.walk_diagnostics import WalkDiagnostics


def test_diagnostics_keep_true_action_and_targets_in_bounded_buffer(tmp_path):
    path = tmp_path / 'walk' / 'latest.npz'
    recorder = WalkDiagnostics(path, 50, '/policy.onnx', capacity=2)
    obs = np.zeros(101)
    for i in range(3):
        obs[0] = i
        recorder.append(i*.02, obs, np.full(14, i*.1), np.full(14, i*.2),
                        [1, 0, 0, 0], [.004, .001, .007, .001],
                        policy_motor_targets=np.full(14, i*.3))
    assert not path.exists(), 'recording must not write disk in the control loop'
    obs[:] = 99
    recorder.save()
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_allclose(data['timestamp_monotonic'], [.02, .04])
        np.testing.assert_allclose(data['obs'][:, 0], [1, 2])
        np.testing.assert_allclose(data['action'][:, 0], [.1, .2])
        np.testing.assert_allclose(data['motor_targets'][:, 0], [.2, .4])
        np.testing.assert_allclose(data['policy_motor_targets'][:, 0], [.3, .6])
        meta = json.loads(str(data['metadata']))
        assert meta['total_ticks'] == 3 and meta['retained_ticks'] == 2
    assert not list(path.parent.glob('.walk-*'))
