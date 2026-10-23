# Testing robo_duck_walk_orient48M

The original export is in `robo_duck_walk_orient48M/`. Tnkr Studio's server selects it through `policies/active.json` and passes both
the ONNX and runtime settings paths to the walker. `GET /api/walk/policy` reports
the model path, SHA-256, and settings configured for the next start. Each start
also logs the selected model and settings. An invalid active bundle blocks startup
instead of silently falling back. Removing `active.json` restores selection from
`scripts/*.onnx`. Run these commands
from the repository root using the runtime virtual environment.

## Offline validation

```bash
.venv/bin/python scripts/check_policy_bundle.py policies/robo_duck_walk_orient48M
```

Checks all 500 observations against ONNX actions, reconstructs all rate-limited
motor targets, and checks the 27-tick phase sequence. This checks the exported
contract; it does not simulate or establish balance on physical hardware.

## Sensor observation before walking

Stop any existing walk and release the server's servo/IMU connections first.
Hold the unpowered robot in the exported default crouch, with both feet touching.
Run on the robot:

```bash
.venv/bin/python scripts/v2_rl_walk_mujoco.py \
  --onnx_model_path policies/robo_duck_walk_orient48M/policy.onnx \
  --runtime_settings_path policies/robo_duck_walk_orient48M/runtime_settings.json \
  --capture_observation /tmp/orient48M-obs.npy

.venv/bin/python scripts/check_policy_bundle.py policies/robo_duck_walk_orient48M \
  --observation /tmp/orient48M-obs.npy
```

Capture does not enable motors or write position targets. It reads the joints,
IMU, and feet once, saves float32[101], then exits and disables torque. The
forward command in this observation is 0.15 to match the golden sample; no
movement command is sent. The comparison prints every block, including the
training-time +1.3 acceleration X bias. Physical sensor readings will differ from
the simulated tick; inspect gravity direction, units, joint order, position
residuals, contacts, and zero action history rather than expecting exact equality.

## Walking with a controller

After checking the sensor observation, support the robot with a catch harness
or have the operator ready to catch it. Run on the robot with an Xbox controller:

```bash
.venv/bin/python scripts/v2_rl_walk_mujoco.py \
  --onnx_model_path policies/robo_duck_walk_orient48M/policy.onnx \
  --runtime_settings_path policies/robo_duck_walk_orient48M/runtime_settings.json \
  --commands
```

The settings select the exported crouch before HWI startup, scale 0.5, 50 Hz,
5.24 rad/s target limit, and fixed 27-tick phase (including the golden first-tick
phase). Movement commands are clamped to X ±0.15, Y ±0.2 m/s, yaw ±1 rad/s.
The phase speed controls are ignored for this bundle. Do not use an action
filter during the first parity test. `--remote` selects the existing remote
controller instead. Startup still follows `start_paused` in `duck_config.json`;
A toggles pause. Ctrl+C exits and disables torque.

The export's README reports slow forward motion, turning drift, and about 8°
forward lean at standstill. Those are exporter-reported simulation results;
physical walking has not been validated by the offline check.

## Runtime timing and robot-side diagnostics

The runtime reads all 14 positions and velocities with two sync-read packets.
Malformed or failed replies fall back to individual reads with the existing
retry/sign/offset handling. Position writes and gain writes remain individual.
The policy sample is reused for cloud telemetry, avoiding extra servo reads.
The loop uses monotonic deadlines and prints measured frequency and 95th
percentile work time every 250 ticks; pauses reset the timing window.

Studio starts automatically pass `--diagnostics_path` pointing to
`diagnostics/walk/latest.npz` in this repository. On exit the walker atomically
saves the most recent 3000 successful policy ticks (about 60 seconds at 50 Hz).
Recording stays in RAM during walking. Each tick contains the actual 101D
observation, raw 14D policy action, rate-limited motor targets, robot quaternion
(wxyz), a monotonic timestamp, and four stage durations: observation, policy,
motor write, telemetry. The metadata includes the model path, control frequency,
action scale, default pose and wall/monotonic start times. An abrupt SIGKILL or
power loss cannot save the in-memory buffer. `latest.npz` is replaced by the
next completed session; copy it before running another test if you need it.

The exported Studio LeRobot recording has a different purpose: in the supplied
export, leg `action` columns repeat measured joint positions and do not contain
independent policy outputs. Use the robot-side NPZ for policy/runtime debugging.


## Training actuator limits (2026-10-20 correction)

The supplied `robo_duck.xml` position actuators use `inheritrange="1"`.
Their control limits are exported by joint name into `actuator_ctrlrange_rad`
in the policy settings, with a SHA-256 of the source model. These limits are
applied to the physical actuator commands and startup pose.

This is essential: the golden simulation requests left/right hip pitch targets
beyond its limits on every tick. MuJoCo clamps those commands; the previous
hardware runtime did not. Do not interpret the golden `motor_targets` as the
post-clamp actuator input.

The policy still observes its original default pose and previous PRE-clamp
rate-limited targets, exactly as in the golden data. Only the final actuator
command is clipped. Diagnostics format v2 saves `motor_targets` as the actual
sent command and `policy_motor_targets` as the separate pre-clamp state.
`obs[:, 83:97]` contains the previous pre-clamp state. The controller limits
are included in the diagnostic metadata. This distinction preserves the
training-time recurrent inputs while reproducing the actuator bounds.
