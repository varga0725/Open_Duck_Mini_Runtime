"""Verify the exported training contract through the production observation/loop code."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mini_bdx_runtime.policy_settings import JOINT_NAMES, PolicySettings
from mini_bdx_runtime.walk_timing import LoopTimer

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'policies/robo_duck_walk_orient48M'


def runner_methods():
    tree = ast.parse((ROOT / 'scripts/v2_rl_walk_mujoco.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RLWalk')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('get_obs', 'run', 'start')]
    namespace = {
        'np': np,
        'time': SimpleNamespace(time=lambda: 1., perf_counter=lambda: 1., sleep=lambda _: None),
        'LoopTimer': lambda hz: LoopTimer(hz, clock=lambda: 1., sleep=lambda _: None),
        'signal': SimpleNamespace(SIGTERM=15, signal=lambda *args: None),
        'walk_pause': SimpleNamespace(read=lambda default: default, clear=Mock()),
        'local_telemetry': SimpleNamespace(clear=Mock(), write_snapshot=Mock()),
        'make_action_dict': lambda targets, names: dict(zip(names, targets)),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<runner>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize("cloud", [False, True])
def test_production_observations_and_loop_match_all_golden_ticks(cloud):
    settings = PolicySettings(BUNDLE / 'runtime_settings.json')
    with np.load(BUNDLE / 'golden.npz', allow_pickle=False) as data:
        golden, actions, targets = (data[k] for k in ('obs', 'action', 'motor_targets'))
    cursor = [0]
    robot = SimpleNamespace(
        policy_settings=settings, capture_observation=None, diagnostics=None,
        num_dofs=14, init_pos=settings.default_pose.copy(),
        motor_targets=settings.default_pose.copy(), prev_motor_targets=settings.default_pose.copy(),
        last_action=np.zeros(14), last_last_action=np.zeros(14), last_last_last_action=np.zeros(14),
        last_commands=[0.15, 0, 0, 0, 0, 0, 0],
        imitation_i=1, imitation_phase=settings.phase(1),
        PRM=SimpleNamespace(nb_steps_in_period=27), phase_frequency_factor=1.3,
        phase_frequency_factor_offset=0.2, action_scale=0.5,
        max_motor_velocity=5.24, control_freq=50,
        paused=False, commands=False, action_filter=None, save_obs=False, replay_obs=None,
        cloud_command_receiver=None, cloud_publisher=Mock() if cloud else None,
        duck_config=SimpleNamespace(antennas=False, eyes=False, projector=False),
        last_dof_pos=None, telemetry_joint_names=list(JOINT_NAMES),
    )
    def row():
        return golden[cursor[0]]
    robot.imu = SimpleNamespace(get_data=lambda: {
        'gyro': row()[:3], 'accelero': row()[3:6] - [1.3, 0, 0]})
    robot.feet_contacts = SimpleNamespace(get=lambda: row()[97:99], stop=Mock())
    def send(actual):
        np.testing.assert_allclose(
            list(actual.values()), settings.actuator_targets(targets[cursor[0]]), atol=1e-5, rtol=0
        )
        cursor[0] += 1
    robot.hwi = SimpleNamespace(
        joints=dict.fromkeys(JOINT_NAMES), turn_off=Mock(), set_position_all=send,
        get_present_positions=Mock(side_effect=lambda **_: row()[13:27] + settings.default_pose),
        get_present_velocities=Mock(side_effect=lambda **_: row()[27:41] / 0.05),
    )
    methods = runner_methods()
    def observe():
        if cursor[0] == len(golden):
            raise SystemExit()
        actual = methods['get_obs'](robot)
        np.testing.assert_allclose(actual, row(), atol=1e-5, rtol=0)
        return actual
    robot.get_obs = observe
    robot.policy = SimpleNamespace(infer=lambda _: actions[cursor[0]].copy())
    methods['run'](robot)
    assert cursor[0] == 500
    assert robot.hwi.get_present_positions.call_count == 500
    assert robot.hwi.get_present_velocities.call_count == 500
    if cloud:
        assert robot.cloud_publisher.publish.call_count == 500
    robot.hwi.turn_off.assert_called_once()


def test_capture_does_not_enable_or_position_motors(tmp_path):
    methods = runner_methods()
    robot = SimpleNamespace(
        pid=[30, 0, 0], capture_observation=str(tmp_path / 'obs.npy'), diagnostics=None,
        hwi=Mock(), get_obs=lambda: np.zeros(101, dtype=np.float32),
        cloud_command_receiver=None, cloud_publisher=None,
        duck_config=SimpleNamespace(antennas=False, eyes=False, projector=False),
        feet_contacts=SimpleNamespace(stop=Mock()), save_obs=False,
    )
    methods['start'](robot)
    methods['run'](robot)
    robot.hwi.set_kps.assert_not_called()
    robot.hwi.set_kds.assert_not_called()
    robot.hwi.turn_on.assert_not_called()
    robot.hwi.set_position_all.assert_not_called()
    assert np.load(robot.capture_observation).shape == (101,)


def test_invalid_action_cannot_become_motor_target():
    settings = PolicySettings(BUNDLE / 'runtime_settings.json')
    with pytest.raises(ValueError):
        settings.targets(np.full(14, np.nan), settings.default_pose)


def test_actuator_clamp_preserves_preclamp_policy_state():
    settings = PolicySettings(BUNDLE / 'runtime_settings.json')
    raw = settings.default_pose.copy()
    raw[2], raw[11] = -0.95, 0.85
    sent = settings.actuator_targets(raw)
    assert sent[2] == pytest.approx(-0.690291)
    assert sent[11] == pytest.approx(0.624821)
    assert raw[2] == -0.95 and raw[11] == 0.85
    assert settings.default_pose[11] == .635
    assert not np.shares_memory(sent, raw)


def test_missing_ctrlrange_preserves_legacy_bundle(tmp_path):
    import json
    data = json.loads((BUNDLE / 'runtime_settings.json').read_text())
    data.pop('actuator_ctrlrange_rad')
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(data))
    settings = PolicySettings(path)
    raw = np.full(14, .9)
    np.testing.assert_array_equal(settings.actuator_targets(raw), raw)


@pytest.mark.parametrize('limits', [{}, {n: [1, -1] for n in JOINT_NAMES}])
def test_invalid_ctrlranges_are_rejected(tmp_path, limits):
    import json
    data = json.loads((BUNDLE / 'runtime_settings.json').read_text())
    data['actuator_ctrlrange_rad'] = limits
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        PolicySettings(path)
