import json
import shutil
import time
from pathlib import Path

import pytest
import tnkr_server
from conftest import write_walk_script

SOURCE = Path(__file__).resolve().parents[1] / 'policies/robo_duck_walk_orient48M'


@pytest.fixture
def studio_scripts(fake_walk_dir, monkeypatch):
    scripts = fake_walk_dir / 'scripts'
    scripts.mkdir()
    (scripts / 'model.onnx').write_bytes(b'')
    monkeypatch.setattr(tnkr_server, 'SCRIPTS_DIR', scripts)
    return scripts


def install_bundle(fake_walk_dir):
    policies = fake_walk_dir.parent / 'policies'
    bundle = policies / SOURCE.name
    bundle.mkdir(parents=True)
    shutil.copyfile(SOURCE / 'runtime_settings.json', bundle / 'runtime_settings.json')
    (bundle / 'policy.onnx').write_bytes(b'fake-model-for-subprocess-argv-test')
    (policies / 'active.json').write_text(json.dumps({'bundle': SOURCE.name}))
    return bundle


def test_studio_passes_active_model_and_settings_to_walker(client, studio_scripts):
    fake_walk_dir = studio_scripts
    bundle = install_bundle(fake_walk_dir)
    output = fake_walk_dir / 'argv.json'
    write_walk_script(fake_walk_dir,
        f'import sys, json, time\nopen({str(output)!r}, "w").write(json.dumps(sys.argv))\ntime.sleep(30)\n')
    try:
        result = client.post('/api/walk/start', json={})
        assert result.status_code == 200
        deadline = time.monotonic() + 3
        while not output.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        args = json.loads(output.read_text())
        assert args[args.index('--onnx_model_path') + 1] == str(bundle / 'policy.onnx')
        assert args[args.index('--runtime_settings_path') + 1] == str(bundle / 'runtime_settings.json')
        assert '--remote' in args
        assert args[args.index('--diagnostics_path') + 1] == str(
            fake_walk_dir.parent / 'diagnostics' / 'walk' / 'latest.npz')
        policy = client.get('/api/walk/policy').json()
        assert policy['model'] == str(bundle / 'policy.onnx')
        assert policy['action_scale'] == 0.5
        assert policy['phase_period'] == 27
        assert policy['actuator_limits_enabled'] is True
        assert len(policy['sha256']) == 64
    finally:
        client.post('/api/walk/stop')


@pytest.mark.parametrize('damage', ['missing_model', 'bad_settings', 'bad_manifest'])
def test_invalid_bundle_cannot_fall_back_to_old_policy(client, studio_scripts, damage):
    fake_walk_dir = studio_scripts
    bundle = install_bundle(fake_walk_dir)
    if damage == 'missing_model':
        (bundle / 'policy.onnx').unlink()
    elif damage == 'bad_settings':
        (bundle / 'runtime_settings.json').write_text('{}')
    else:
        (bundle.parent / 'active.json').write_text('{bad json')
    assert client.post('/api/walk/start', json={}).status_code == 503
    assert tnkr_server.walk_session is None


def test_legacy_install_without_active_bundle(client, studio_scripts):
    fake_walk_dir = studio_scripts
    model, settings_path, settings = tnkr_server.selected_walk_policy()
    assert model == fake_walk_dir / 'model.onnx'
    assert settings_path is None and settings is None
