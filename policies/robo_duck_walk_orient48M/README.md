# robo_duck walk policy (orient 48.7M), robot package

Walking policy trained on `scene_flat_terrain_robo_duck.xml` (2.94 kg model).
Source: `checkpoints_orient/2026_10_20_081209_48660480`.

| file | purpose |
|---|---|
| `policy.onnx` | `obs float32[1,101] -> action float32[1,14]` |
| `runtime_settings.json` | default pose, action scale, phase period, obs layout, command ranges |
| `polynomial_coefficients.pkl` | reference motion (only its period, 27 ticks, is used at runtime) |
| `golden.npz` | 500 simulated ticks (`obs`, `action`, `motor_targets`) with command `lin_vel_x=0.15` |

## Three settings the runtime must change

This policy is the standard Open Duck 101D layout (`v2_rl_walk_mujoco.py`),
not the HW10 v2 124D / stand_v1 103D layout.

1. **Default pose = the crouch pose** in `runtime_settings.json`
   (`left_hip_pitch -0.63, left_knee 1.368, left_ankle -0.784`, ...).
   Not the hw9 home (`-0.319, 0.368, 0.097`). Observations are `joint - default_pose`
   and targets are `default_pose + action_scale * action`.
2. **`action_scale = 0.5`** (was 0.25).
3. **Phase period 27 ticks** at 50 Hz.

## Before the robot moves

Feed the `golden.npz` observations to `policy.onnx` and compare to `action`
(must match to ~1e-5). Then log the runtime's own 101D observation while the
robot stands at the default pose and compare block by block to `golden.npz` tick 0.

## Simulated behaviour (10 s per command, no fall in any)

| command | reached | foot lift | trunk pitch |
|---|---|---|---|
| stand | 0 | 0 mm | +8.2° |
| forward 0.15 m/s | 0.042 m/s, **turns 0.23 rad/s while walking** | 23 mm | +5.7° |
| backward -0.15 m/s | -0.014 m/s | 24 mm | +0.5° |
| left 0.2 m/s | 0.110 m/s | 25 mm | +3.1° |
| turn 1.0 rad/s | 0.94 rad/s | 29 mm | +4.3° |

Known limits: slow forward, drifts into a curve, still leans ~8° forward
standing. Test with the operator's hands ready or a catch harness.
