"""Status feed: built only from server-held state, never from the bus."""
import json
import time

import pytest

import tnkr_server
from mini_bdx_runtime import status_feed

SNAP = {"joints": {"left_hip_yaw": 0.12345, "right_knee": -0.5}, "imu": {"quaternion": [1, 0, 0, 0], "gyro": [0, 0, 1], "accelero": [0, 0, 9.8]}}


def build(**kw):
    base = dict(walking=False, paused=False, diagnostic=False, walk_snapshot=None, cached_joints={}, last_voltage=None, now_monotonic=100.0)
    return status_feed.build_status(**{**base, **kw})


def test_idle_without_data_reports_nothing_invented():
    s = build()
    assert s == {"mode": "idle", "joints": None, "imu": None, "battery": None,
                 "sensors": {"walk_paused": None, "joints_source": "none"}}


def test_walking_uses_walk_loop_snapshot():
    s = build(walking=True, paused=True, walk_snapshot=SNAP)
    assert s["mode"] == "autonomous" and s["sensors"] == {"walk_paused": True, "joints_source": "walk_loop"}
    assert s["joints"]["left_hip_yaw"] == {"position_rad": 0.1235}
    assert s["imu"]["accel"] == [0.0, 0.0, 9.8]


def test_cached_pose_is_labelled_cached_and_diagnostic_mode():
    s = build(diagnostic=True, cached_joints={"neck_pitch": 0.2})
    assert s["mode"] == "diagnostic" and s["sensors"]["joints_source"] == "cached_last_read"
    assert s["imu"] is None  # the IMU is only read by the walk loop or /api/state, never by the feed


def test_battery_from_last_voltage_with_age():
    v = {"volts": 7.41, "health": "ok", "max_temp_c": 38, "temp_health": "ok", "at": 90.0}
    assert build(last_voltage=v)["battery"] == {"voltage_v": 7.41, "health": "ok", "max_temp_c": 38, "temp_health": "ok", "age_s": 10.0}


def test_write_is_atomic_and_world_readable(tmp_path):
    p = tmp_path / "status.json"
    status_feed.write_status(str(p), build())
    assert json.loads(p.read_text())["mode"] == "idle" and oct(p.stat().st_mode & 0o777) == "0o644"
    assert not (tmp_path / "status.json.tmp").exists()


def test_server_publishes_without_touching_the_bus(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    def no_bus(*a, **k):
        raise AssertionError("status feed must not open the servo bus")

    monkeypatch.setattr(tnkr_server, "get_hwi", no_bus)
    monkeypatch.setattr(status_feed, "INTERVAL_S", 0.05)
    path = tmp_path / "status.json"
    monkeypatch.setenv(status_feed.ENV_VAR, str(path))
    with TestClient(tnkr_server.app):
        deadline = time.time() + 5
        while not path.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert json.loads(path.read_text())["mode"] in ("idle", "diagnostic")


def test_feed_is_off_without_env(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.delenv(status_feed.ENV_VAR, raising=False)
    with TestClient(tnkr_server.app):
        time.sleep(0.2)
    assert not any(t.name == "status-feed" and t.is_alive() for t in __import__("threading").enumerate())
