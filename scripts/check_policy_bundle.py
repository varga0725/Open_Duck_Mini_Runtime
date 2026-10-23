"""Check an exported policy against golden data without opening robot hardware."""
import argparse
import json
from pathlib import Path

import numpy as np

from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.policy_settings import OBS_BLOCKS, PolicySettings


def check_bundle(bundle, observation_path=None):
    bundle = Path(bundle).resolve()
    settings = PolicySettings(bundle / 'runtime_settings.json')
    policy = OnnxInfer(str(bundle / 'policy.onnx'), awd=True)
    with np.load(bundle / 'golden.npz', allow_pickle=False) as golden:
        obs, expected, targets = (golden[k] for k in ('obs', 'action', 'motor_targets'))
        if obs.ndim != 2 or obs.shape[1] != 101 or len(obs) == 0:
            raise ValueError('Expected nonempty golden observations with shape [ticks, 101]')
        if expected.shape != (len(obs), 14) or targets.shape != expected.shape:
            raise ValueError('Golden action/target shapes do not match observations')
        if not all(np.isfinite(a).all() for a in (obs, expected, targets)):
            raise ValueError('Golden data contains nonfinite values')
        actual = np.array([policy.infer(o.astype(np.float32)) for o in obs])
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
        previous = settings.default_pose.copy()
        reconstructed = []
        for action in actual:
            previous = settings.targets(action, previous)
            reconstructed.append(previous)
        reconstructed = np.asarray(reconstructed)
        np.testing.assert_allclose(reconstructed, targets, atol=1e-5, rtol=0)
        phases = np.array([settings.phase(i + 1) for i in range(len(obs))])
        np.testing.assert_allclose(phases, obs[:, -2:], atol=1e-5, rtol=0)
        report = {
            'ticks': len(obs),
            'max_action_error': float(np.max(np.abs(actual - expected))),
            'max_target_error': float(np.max(np.abs(reconstructed - targets))),
            'control_hz': settings.control_hz,
            'phase_period': settings.period,
        }
        if observation_path:
            recorded = np.load(observation_path, allow_pickle=False)
            if recorded.ndim == 2:
                recorded = recorded[0]
            if recorded.shape != (101,) or not np.isfinite(recorded).all():
                raise ValueError('Expected a finite 101D observation (.npy)')
            offset = 0
            report['observation_comparison'] = {}
            for name, size in OBS_BLOCKS:
                sl = slice(offset, offset + size)
                report['observation_comparison'][name] = {
                    'recorded': recorded[sl].tolist(),
                    'golden_tick_0': obs[0, sl].tolist(),
                    'max_abs_difference': float(np.max(np.abs(recorded[sl] - obs[0, sl]))),
                }
                offset += size
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--observation', type=Path, help='Compare a captured .npy observation by block')
    args = parser.parse_args()
    print(json.dumps(check_bundle(args.bundle, args.observation), indent=2))
