"""Read-only status file for external monitors (e.g. the ByteWolf agent).

Built only from state the server already holds — the walk loop's /dev/shm snapshot, the last pose
/api/state cached, the last /api/voltage reading. It never opens the servo bus or the IMU, never
takes BUS_LOCK, and never sends a command, so a monitor polling the file cannot disturb the robot.
Off unless TNKR_STATUS_FILE is set; written atomically (temp + rename), same pattern as walk_telemetry.
"""
from __future__ import annotations

import json
import os
import time

ENV_VAR = "TNKR_STATUS_FILE"
INTERVAL_S = 0.5


def build_status(
    *,
    walking: bool,
    paused: bool,
    diagnostic: bool,
    walk_snapshot: dict | None,
    cached_joints: dict[str, float],
    last_voltage: dict | None,
    now_monotonic: float,
) -> dict:
    """Pure function: server state in, status document out (shape documented in README)."""
    if walking:
        mode = "autonomous"
    elif diagnostic:
        mode = "diagnostic"
    else:
        mode = "idle"
    sensors: dict = {"walk_paused": paused if walking else None}

    joints = None
    imu = None
    if walking and walk_snapshot and walk_snapshot.get("joints"):
        joints = {n: {"position_rad": round(float(p), 4)} for n, p in walk_snapshot["joints"].items()}
        sensors["joints_source"] = "walk_loop"
        snap_imu = walk_snapshot.get("imu") or {}
        if snap_imu.get("quaternion"):
            imu = {"quaternion": [float(q) for q in snap_imu["quaternion"]],
                   "gyro": [float(g) for g in snap_imu.get("gyro", [])],
                   "accel": [float(a) for a in snap_imu.get("accelero", [])]}
    elif cached_joints:
        # Last pose read by /api/state: may be old. Reported as cached, never as live.
        joints = {n: {"position_rad": round(float(p), 4)} for n, p in cached_joints.items()}
        sensors["joints_source"] = "cached_last_read"
    else:
        sensors["joints_source"] = "none"

    battery = None
    if last_voltage:
        battery = {"voltage_v": last_voltage["volts"], "health": last_voltage["health"],
                   "max_temp_c": last_voltage["max_temp_c"], "temp_health": last_voltage["temp_health"],
                   "age_s": round(now_monotonic - float(last_voltage["at"]), 1)}
    return {"mode": mode, "joints": joints, "imu": imu, "battery": battery, "sensors": sensors}


def write_status(path: str, status: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({**status, "timestamp": time.time()}, f)
    os.chmod(tmp, 0o644)  # world-readable: monitors run as other (unprivileged) users
    os.replace(tmp, path)
