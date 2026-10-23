"""Execute the production runner loop with fake hardware; never start real motors."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import tnkr_server
from conftest import write_walk_script
from mini_bdx_runtime import walk_pause
from mini_bdx_runtime.walk_timing import LoopTimer


@pytest.fixture(autouse=True)
def pause_file(tmp_path, monkeypatch):
    monkeypatch.setattr(walk_pause, 'PAUSE_FILE', str(tmp_path / 'pause.json'))


def run_until_observation(paused, during_wait, commands=False):
    # Load only run() so importing hardware drivers cannot open a real device.
    source = Path(tnkr_server.__file__).with_name('v2_rl_walk_mujoco.py')
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RLWalk')
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    sleeps = []
    observations = []
    buttons = SimpleNamespace(**{n: SimpleNamespace(triggered=False, is_pressed=False)
                                for n in ('dpad_up', 'dpad_down', 'LB', 'X', 'B', 'A')})
    def sleep(seconds):
        sleeps.append(seconds)
        assert len(sleeps) <= 3, 'runner never resumed'
        during_wait(robot)
    def observe():
        observations.append(robot.paused)
        raise SystemExit()
    robot = SimpleNamespace(
        paused=paused, commands=commands, get_obs=observe, diagnostics=None, control_freq=50,
        buttons=buttons, xbox_controller=SimpleNamespace(
            get_last_command=lambda: ([0]*7, buttons, 0, 0)),
        duck_config=SimpleNamespace(antennas=False, eyes=False, projector=False),
        cloud_command_receiver=None, cloud_publisher=None,
        feet_contacts=SimpleNamespace(stop=Mock()),
        hwi=SimpleNamespace(turn_off=Mock()), save_obs=False,
    )
    if commands:
        buttons.A.triggered = True
    namespace = {'time': SimpleNamespace(time=lambda: 1, perf_counter=lambda: 1, sleep=sleep),
                 'LoopTimer': lambda hz: LoopTimer(hz, clock=lambda: 1, sleep=sleep),
                 'signal': SimpleNamespace(SIGTERM=15, signal=lambda *args: None),
                 'walk_pause': walk_pause, 'local_telemetry': SimpleNamespace(clear=Mock())}
    exec(compile(ast.Module(body=[run], type_ignores=[]), str(source), 'exec'), namespace)
    namespace['run'](robot)
    assert observations == [False]
    assert sleeps == [0.1]
    robot.hwi.turn_off.assert_called_once()


@pytest.mark.parametrize('initial', [True, False])
def test_http_pause_is_observed_and_resume_releases_gate(initial):
    walk_pause.write(True)
    def resume(robot):
        assert robot.paused is True
        walk_pause.write(False)
    run_until_observation(initial, resume)


def test_controller_toggle_is_published_and_http_can_resume_it():
    walk_pause.write(False)
    def resume(robot):
        assert walk_pause.read() is True
        robot.buttons.A.triggered = False
        walk_pause.write(False)
    run_until_observation(False, resume, commands=True)


def test_missing_file_preserves_runner_state():
    assert walk_pause.read(default=True) is True
    assert walk_pause.read() is False


@pytest.mark.parametrize('initial', [True, False])
def test_server_publishes_startup_pause_before_runner_boots(client, fake_walk_dir, tmp_path, monkeypatch, initial):
    config = tmp_path / 'config.json'
    config.write_text(__import__('json').dumps({'start_paused': initial}))
    monkeypatch.setattr(tnkr_server, 'CONFIG_PATH', str(config))
    write_walk_script(fake_walk_dir, 'import time; time.sleep(30)\n')
    try:
        assert client.post('/api/walk/start', json={}).status_code == 200
        assert client.get('/api/health').json()['paused'] is initial
        assert client.post('/api/walk/resume').status_code == 200
        assert client.get('/api/health').json()['paused'] is False
    finally:
        client.post('/api/walk/stop')


def test_api_and_controller_can_publish_concurrently():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(walk_pause.write, [True, False] * 20))
    assert isinstance(walk_pause.read(), bool)
    assert not list(Path(walk_pause.PAUSE_FILE).parent.glob('.tnkr-pause-*'))
