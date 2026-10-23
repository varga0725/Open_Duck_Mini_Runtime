"""The pause flag — freeze the gait without dropping torque.

/api/walk/stop kills the walk process, and its cleanup disables torque, so the
duck goes limp and falls. Pausing has to leave the walk alive: it keeps running
but stops issuing new targets, and Feetech servos hold their last commanded
target in firmware, so the duck stands exactly where it was.

The walk is a separate process that owns the servo bus, so the server can't
reach into its loop — the flag goes through a one-slot /dev/shm file the walk
checks each control tick (~20ms at 50 Hz). Unlike the steering command file,
this is latched state rather than a stream: it stays set until someone clears
it, so there's no freshness window here. Both sides write it — the server on
/api/walk/pause, the walk itself when the controller's A button toggles — which
is what keeps the dashboard and the gamepad agreeing on one pause state.

The walk's start and stop paths both call clear(): a pause belongs to the walk
that was paused, and a walk killed hard (SIGKILL, power cut) never runs its own
cleanup, so a leftover file would otherwise read as the *next* walk's pause.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

# tmpfs on the Pi (RAM — no SD-card wear); tempdir off-Pi so the code stays
# importable/testable on dev machines without /dev/shm. Mirrors walk_telemetry.
PAUSE_FILE = (
    "/dev/shm/tnkr_walk_pause.json"
    if os.path.isdir("/dev/shm")
    else os.path.join(tempfile.gettempdir(), "tnkr_walk_pause.json")
)


def write(paused: bool) -> None:
    """Publish the pause state. Temp + atomic rename, so a reader mid-tick sees
    either the old flag or the new one — never a half-written file."""
    # Both the API process and the controller can publish a toggle. Each
    # writer needs its own staging file so concurrent replaces cannot collide.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=os.path.dirname(PAUSE_FILE) or ".",
            prefix=".tnkr-pause-", delete=False,
        ) as output:
            tmp = output.name
            json.dump({"timestamp": time.time(), "paused": bool(paused)}, output)
        os.replace(tmp, PAUSE_FILE)
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)


def read(default: bool = False) -> bool:
    """Whether the walk should be frozen right now.

    The runner passes its current state as default for a standalone launch or
    a transient read failure; the server defaults to false outside a walk.

    Missing or unreadable state uses the supplied default. Without an explicit
    default, "no file" means "not paused", as expected by server status reads.
    """
    try:
        with open(PAUSE_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return bool(default)
    return bool(data.get("paused", default))


def clear() -> None:
    try:
        os.remove(PAUSE_FILE)
    except FileNotFoundError:
        pass
