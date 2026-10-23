import signal
import time
import pickle

import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer

from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.poly_reference_motion import PolyReferenceMotion
from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter
from mini_bdx_runtime.duck_config import DuckConfig
# StateServer removed — cloud telemetry via CloudPublisher; local telemetry is a
# one-slot /dev/shm snapshot the server's /api/state reads while we own the bus
from mini_bdx_runtime import walk_telemetry as local_telemetry
from mini_bdx_runtime import walk_pause
from mini_bdx_runtime.policy_settings import JOINT_NAMES, PolicySettings
from mini_bdx_runtime.walk_timing import LoopTimer
from mini_bdx_runtime.walk_diagnostics import WalkDiagnostics

import os

HOME_DIR = os.path.expanduser("~")


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        duck_config_path: str = f"{HOME_DIR}/duck_config.json",
        serial_port: str = None,  # None -> HWI auto-detects the servo adapter by USB vendor id
        control_freq: float = 50,
        pid=[30, 0, 0],
        action_scale=0.25,
        commands=False,
        remote=False,
        pitch_bias=0,
        save_obs=False,
        replay_obs=None,
        cutoff_frequency=None,
        cloud_channel=None,
        cloud_commands_channel=None,
        supabase_url=None,
        supabase_key=None,
        runtime_settings_path=None,
        capture_observation=None,
        diagnostics_path=None,
    ):

        self.policy_settings = (
            PolicySettings(runtime_settings_path) if runtime_settings_path else None
        )
        self.capture_observation = capture_observation
        self.diagnostics = None
        if self.policy_settings:
            control_freq = self.policy_settings.control_hz
            action_scale = self.policy_settings.action_scale

        self.duck_config = DuckConfig(config_json_path=duck_config_path)
        if capture_observation:
            for feature in ("eyes", "projector", "speaker", "antennas"):
                setattr(self.duck_config, feature, False)

        self.cloud_publisher = None
        if cloud_channel and supabase_url and supabase_key and not capture_observation:
            from mini_bdx_runtime.cloud_publisher import CloudPublisher
            self.cloud_publisher = CloudPublisher(cloud_channel, supabase_url, supabase_key)
            self.cloud_publisher.start()

        self.cloud_command_receiver = None
        if cloud_commands_channel and remote and supabase_url and supabase_key and not capture_observation:
            from mini_bdx_runtime.cloud_command_receiver import CloudCommandReceiver
            self.cloud_command_receiver = CloudCommandReceiver(cloud_commands_channel, supabase_url, supabase_key)
            self.cloud_command_receiver.start()

        self.commands = commands and not capture_observation
        self.pitch_bias = pitch_bias

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        self.num_dofs = 14
        self.max_motor_velocity = (
            self.policy_settings.max_motor_velocity_rad_s if self.policy_settings else 5.24
        )  # rad/s

        # Control
        self.control_freq = control_freq
        if diagnostics_path and not capture_observation:
            self.diagnostics = WalkDiagnostics(
                diagnostics_path, control_freq, onnx_model_path
            )
        self.pid = pid

        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        self.hwi = HWI(self.duck_config, serial_port)

        if self.policy_settings:
            names = list(self.hwi.joints)
            if names != list(JOINT_NAMES):
                raise ValueError("Hardware joint order does not match policy")
            self.hwi.init_pos = dict(zip(
                names, self.policy_settings.actuator_targets(self.policy_settings.default_pose)
            ))

        # get_obs stashes each tick's raw positions here so the local telemetry
        # drop costs zero extra bus reads
        self.telemetry_joint_names = [
            k for k in self.hwi.joints.keys() if "antenna" not in k
        ]
        self.last_dof_pos = None
        self.last_dof_vel = None
        self.last_feet_contacts = None
        self.last_imu_data = None

        self.start()

        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )

        self.feet_contacts = FeetContacts()

        # Scales
        self.action_scale = action_scale

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        # Normalization and recurrent targets retain the training default, even
        # when the actuator clamps its startup command (right hip: .635 -> .624821).
        self.init_pos = (
            self.policy_settings.default_pose.tolist() if self.policy_settings
            else list(self.hwi.init_pos.values())
        )
        if self.diagnostics is not None:
            self.diagnostics.metadata.update({
                "default_pose": self.init_pos,
                "action_scale": self.action_scale,
                "runtime_settings_path": runtime_settings_path,
                "actuator_ctrlrange_rad": (
                    self.policy_settings.actuator_ctrlrange.tolist()
                    if self.policy_settings and self.policy_settings.actuator_ctrlrange is not None
                    else None
                ),
            })

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = self.duck_config.start_paused

        self.command_freq = 20  # hz
        self.remote = remote
        if self.commands:
            if self.remote:
                from mini_bdx_runtime.remote_controller import RemoteController
                self.xbox_controller = RemoteController(self.command_freq)
            else:
                self.xbox_controller = XBoxController(self.command_freq)

        # Reference motion, but we only really need the length of one phase
        # TODO
        if self.policy_settings:
            from types import SimpleNamespace
            self.PRM = SimpleNamespace(nb_steps_in_period=self.policy_settings.period)
        else:
            self.PRM = PolyReferenceMotion("./polynomial_coefficients.pkl")
        self.imitation_i = 1 if self.policy_settings else 0
        self.imitation_phase = (
            self.policy_settings.phase(self.imitation_i)
            if self.policy_settings else np.array([0, 0])
        )
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = (
            self.duck_config.phase_frequency_factor_offset
        )

        # Optional expression features — failures must never prevent walking
        if self.duck_config.eyes:
            try:
                self.eyes = Eyes()
            except Exception as e:
                print(f"[Expression] Eyes init failed, disabling: {e}")
                self.duck_config.eyes = False
        if self.duck_config.projector:
            try:
                self.projector = Projector()
            except Exception as e:
                print(f"[Expression] Projector init failed, disabling: {e}")
                self.duck_config.projector = False
        if self.duck_config.speaker:
            try:
                self.sounds = Sounds(
                    volume=1.0, sound_directory="../mini_bdx_runtime/assets/"
                )
            except Exception as e:
                print(f"[Expression] Sounds init failed, disabling: {e}")
                self.duck_config.speaker = False
        if self.duck_config.antennas:
            try:
                self.antennas = Antennas()
            except Exception as e:
                print(f"[Expression] Antennas init failed, disabling: {e}")
                self.duck_config.antennas = False

    def get_obs(self):

        imu_data = self.imu.get_data()

        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs:
            print(f"ERROR len(dof_pos) != {self.num_dofs}")
            return None

        if len(dof_vel) != self.num_dofs:
            print(f"ERROR len(dof_vel) != {self.num_dofs}")
            return None

        # stash for the local telemetry drop in the run loop — no extra bus reads
        self.last_dof_pos = dof_pos
        self.last_dof_vel = dof_vel
        self.last_imu_data = imu_data

        cmds = self.last_commands

        feet_contacts = self.feet_contacts.get()
        self.last_feet_contacts = feet_contacts

        acceleration = np.array(imu_data["accelero"], dtype=float, copy=True)
        if self.policy_settings:
            acceleration[0] += self.policy_settings.accel_x_bias

        obs = np.concatenate(
            [
                imu_data["gyro"],
                acceleration,
                cmds,
                dof_pos - self.init_pos,
                dof_vel * 0.05,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                feet_contacts,
                self.imitation_phase,
            ]
        )

        return obs.astype(np.float32)

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

        # lower head kps
        kps[5:9] = [8, 8, 8, 8]

        if self.capture_observation:
            return
        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

    def get_phase_frequency_factor(self, x_velocity):

        max_phase_frequency = 1.2
        min_phase_frequency = 1.0

        # Perform linear interpolation
        freq = min_phase_frequency + (abs(x_velocity) / 0.15) * (
            max_phase_frequency - min_phase_frequency
        )

        return freq

    def run(self):
        # Convert SIGTERM to SystemExit so the finally block runs cleanup
        def handle_sigterm(signum, frame):
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, handle_sigterm)

        i = 0
        try:
            if getattr(self, "capture_observation", None):
                self.last_commands = [0.15, 0, 0, 0, 0, 0, 0]
                obs = self.get_obs()
                if obs is None:
                    raise RuntimeError("Cannot capture observation: joint read failed")
                np.save(self.capture_observation, obs)
                print(f"Observation saved: {self.capture_observation}")
                return
            print("Starting")
            start_t = time.perf_counter()
            timer = LoopTimer(self.control_freq)
            while True:
                left_trigger = 0
                right_trigger = 0
                t = time.perf_counter()

                # HTTP pause/resume and the controller share one latched state.
                self.paused = walk_pause.read(default=self.paused)

                if self.commands:
                    self.last_commands, self.buttons, left_trigger, right_trigger = (
                        self.xbox_controller.get_last_command()
                    )
                    if self.buttons.dpad_up.triggered:
                        self.phase_frequency_factor_offset += 0.05
                        print(
                            f"Phase frequency factor offset {round(self.phase_frequency_factor_offset, 3)}"
                        )

                    if self.buttons.dpad_down.triggered:
                        self.phase_frequency_factor_offset -= 0.05
                        print(
                            f"Phase frequency factor offset {round(self.phase_frequency_factor_offset, 3)}"
                        )

                    if self.buttons.LB.is_pressed:
                        self.phase_frequency_factor = 1.3
                    else:
                        self.phase_frequency_factor = 1.0

                    if self.buttons.X.triggered:
                        if self.duck_config.projector:
                            self.projector.switch()

                    if self.buttons.B.triggered:
                        if self.duck_config.speaker:
                            self.sounds.play_random_sound()

                    if self.duck_config.antennas:
                        self.antennas.set_position_left(right_trigger)
                        self.antennas.set_position_right(left_trigger)

                    if self.buttons.A.triggered:
                        self.paused = not self.paused
                        walk_pause.write(self.paused)
                        if self.paused:
                            print("PAUSE")
                        else:
                            print("UNPAUSE")

                if getattr(self, "policy_settings", None):
                    self.last_commands = list(self.last_commands)
                    self.last_commands[:3] = np.clip(
                        self.last_commands[:3], [-0.15, -0.2, -1.0], [0.15, 0.2, 1.0]
                    ).tolist()

                if self.paused:
                    timer.reset()
                    time.sleep(0.1)
                    continue

                obs = self.get_obs()
                if obs is None:
                    timer.finish(t)
                    continue
                observation_done = time.perf_counter()

                self.imitation_i += (
                    1 if self.policy_settings else
                    self.phase_frequency_factor + self.phase_frequency_factor_offset
                )
                self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                self.imitation_phase = np.array(
                    [
                        np.cos(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                        np.sin(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                    ]
                )

                if self.save_obs:
                    self.saved_obs.append(obs)

                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        print("BREAKING ")
                        break

                action = self.policy.infer(obs)
                if self.policy_settings and (
                    np.asarray(action).shape != (14,) or not np.isfinite(action).all()
                ):
                    raise ValueError("Policy returned an invalid action")

                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                policy_done = time.perf_counter()

                self.motor_targets = self.init_pos + action * self.action_scale

                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if (
                        time.perf_counter() - start_t > 1
                    ):  # give time to the filter to stabilize
                        self.motor_targets = filtered_motor_targets

                head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets

                if self.policy_settings:
                    limit = self.max_motor_velocity / self.control_freq
                    self.motor_targets = np.clip(
                        self.motor_targets, self.prev_motor_targets - limit,
                        self.prev_motor_targets + limit,
                    )
                self.prev_motor_targets = self.motor_targets.copy()

                # MuJoCo position actuators clamp ctrl to the inherited joint
                # range. Keep self.motor_targets unmodified: training feeds those
                # PRE-clamp targets into the next 101D observation.
                actuator_targets = (
                    self.policy_settings.actuator_targets(self.motor_targets)
                    if self.policy_settings else self.motor_targets.copy()
                )
                action_dict = make_action_dict(
                    actuator_targets, list(self.hwi.joints.keys())
                )

                self.hwi.set_position_all(action_dict)
                motor_write_done = time.perf_counter()

                if self.last_dof_pos is not None:
                    local_telemetry.write_snapshot(
                        dict(
                            zip(
                                self.telemetry_joint_names,
                                (float(p) for p in self.last_dof_pos),
                            )
                        ),
                        imu={
                            "quaternion": [
                                float(q)
                                for q in self.last_imu_data.get(
                                    "quaternion", [1.0, 0.0, 0.0, 0.0]
                                )
                            ],
                            "gyro": [float(g) for g in self.last_imu_data["gyro"]],
                            "accelero": [float(a) for a in self.last_imu_data["accelero"]],
                        },
                    )

                if self.cloud_publisher is not None:
                    # Reuse the policy sample: telemetry must not reread the bus.
                    dof_pos = self.last_dof_pos
                    dof_vel = self.last_dof_vel
                    imu_data = self.last_imu_data
                    feet = self.last_feet_contacts
                    joint_names = self.telemetry_joint_names
                    state_snapshot = {
                        "timestamp": time.time(),
                        "joint_positions": dict(zip(joint_names, [float(p) for p in dof_pos])) if dof_pos is not None else {},
                        "joint_velocities": dict(zip(joint_names, [float(v) for v in dof_vel])) if dof_vel is not None else {},
                        "motor_targets": dict(zip(joint_names, [float(m) for m in actuator_targets])),
                        "imu": {
                            "quaternion": imu_data.get("quaternion", [0, 0, 0, 1]),
                            "gyro": [float(g) for g in imu_data["gyro"]],
                            "accelero": [float(a) for a in imu_data["accelero"]],
                        },
                        "feet_contacts": {
                            "left": bool(feet[0]),
                            "right": bool(feet[1]),
                        },
                        "paused": self.paused,
                        "fps": timer.actual_hz,
                    }
                    self.cloud_publisher.publish(state_snapshot)

                telemetry_done = time.perf_counter()
                if self.diagnostics is not None:
                    self.diagnostics.append(
                        t, obs, action, actuator_targets,
                        self.last_imu_data.get("quaternion", [1, 0, 0, 0]),
                        (observation_done - t, policy_done - observation_done,
                         motor_write_done - policy_done, telemetry_done - motor_write_done),
                        policy_motor_targets=self.motor_targets,
                    )
                i += 1

                timer.finish(t)

        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            # a leftover snapshot from a dead walk must never read as a live pose
            local_telemetry.clear()
            walk_pause.clear()
            # Clean up cloud connections
            if self.cloud_command_receiver is not None:
                self.cloud_command_receiver.stop()
            if self.cloud_publisher is not None:
                self.cloud_publisher.stop()
            if self.duck_config.antennas:
                self.antennas.stop()
            if self.duck_config.eyes:
                self.eyes.stop()
            if self.duck_config.projector:
                self.projector.stop()
            self.feet_contacts.stop()

            # Disable torque so the robot goes limp
            try:
                self.hwi.turn_off()
            except Exception:
                pass

            if self.diagnostics is not None:
                try:
                    self.diagnostics.save()
                    print(f"[Walk diagnostics] saved: {self.diagnostics.path}", flush=True)
                except Exception as exc:
                    print(f"[Walk diagnostics] save failed: {exc}", flush=True)
            if self.save_obs:
                pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
            print("TURNING OFF")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument("--diagnostics_path", type=str, default=None,
                        help="Save the last 3000 real policy ticks as .npz on exit")
    parser.add_argument("--runtime_settings_path", type=str, default=None,
                        help="Use an exported 101D policy runtime_settings.json")
    parser.add_argument("--capture_observation", type=str, default=None,
                        help="Save one .npy observation without enabling motors, then exit")
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="save the run's observations",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        default=False,
        help="Use remote commands from TNKR dashboard instead of Xbox controller",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument(
        "--cloud_channel",
        type=str,
        default=None,
        help="Supabase Broadcast channel name for cloud telemetry relay",
    )
    parser.add_argument(
        "--cloud_commands_channel",
        type=str,
        default=None,
        help="Supabase Broadcast channel name for cloud command relay",
    )
    parser.add_argument(
        "--supabase_url",
        type=str,
        default=None,
        help="Supabase project URL for cloud telemetry/commands",
    )
    parser.add_argument(
        "--supabase_key",
        type=str,
        default=None,
        help="Supabase anon key for cloud telemetry/commands",
    )

    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("Done parsing args")
    rl_walk = RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        remote=args.remote,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
        cloud_channel=args.cloud_channel,
        cloud_commands_channel=args.cloud_commands_channel,
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        runtime_settings_path=args.runtime_settings_path,
        capture_observation=args.capture_observation,
        diagnostics_path=args.diagnostics_path,
    )
    print("Done instantiating RLWalk")
    rl_walk.run()
