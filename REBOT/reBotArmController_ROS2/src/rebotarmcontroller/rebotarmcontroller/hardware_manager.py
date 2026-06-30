from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .conversions import fk_to_pose

_GC_VEL_THRESHOLD = 0.04
_GC_W_VEL_THRESHOLD = 0.08
_GC_KP = 7.0
_GC_KD = 0.8
_GRIPPER_GOAL_TOLERANCE_RAD = 0.12
_GRIPPER_CLOSED_POSITION = 0.0


class HardwareManager:
    """Thin ROS-facing holder for SDK RobotArm, ArmEndPos, and Gripper."""

    def __init__(
        self,
        arm_cfg: Optional[str] = None,
        gripper_cfg: Optional[str] = None,
        channel: str = "",
    ) -> None:
        self._sdk_root = self._ensure_rebot_sdk_in_syspath()

        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.controllers import ArmEndPos
        from reBotArm_control_py.dynamics import compute_generalized_gravity
        from reBotArm_control_py.kinematics import (
            get_end_effector_frame_id,
            load_robot_model,
        )
        import pinocchio as pin

        arm_cfg_path = (
            Path(arm_cfg).expanduser()
            if arm_cfg
            else self._sdk_root / "config" / "arm.yaml"
        )
        arm_cfg_path = self._cfg_with_channel(
            arm_cfg_path,
            channel,
            "arm_channel_override.yaml",
        )
        self._gripper_cfg_path = (
            Path(gripper_cfg).expanduser()
            if gripper_cfg
            else self._sdk_root / "config" / "gripper.yaml"
        )
        self._gripper_cfg_path = self._cfg_with_channel(
            self._gripper_cfg_path,
            channel,
            "gripper_channel_override.yaml",
        )

        self._arm = RobotArm(cfg_path=str(arm_cfg_path))
        self._endpos_ctrl = ArmEndPos(self._arm)
        self._gripper_cfg = None
        self._gripper_mot = None
        self._gripper_ctrl = None
        self._gripper_mode = "mit"
        self._gripper_target_position: float | None = None

        self._gc_model = load_robot_model()
        self._gc_data = self._gc_model.createData()
        self._gc_ee_frame_id = get_end_effector_frame_id(self._gc_model)
        self._gc_compute_generalized_gravity = compute_generalized_gravity
        self._gc_pin = pin

        self._connected = False
        self._enabled = False
        self._state_machine = "IDLE"
        self._error_codes: list[str] = []
        self._gravity_comp_active = False
        self._gravity_comp_q_target: np.ndarray | None = None
        self._gravity_comp_integral: np.ndarray | None = None
        self._gravity_comp_lock_counter = 0
        self._gravity_comp_q_last: np.ndarray | None = None

    @staticmethod
    def _workspace_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _sdk_candidates(cls) -> list[Path]:
        workspace = cls._workspace_root()
        return [
            workspace / "third_party" / "reBotArm_control_py",
        ]

    @classmethod
    def _ensure_rebot_sdk_in_syspath(cls) -> Path:
        for root in cls._sdk_candidates():
            if (root / "reBotArm_control_py").is_dir():
                root_str = str(root)
                if root_str not in sys.path:
                    sys.path.insert(0, root_str)
                return root
        candidates = "\n".join(f"  - {path}" for path in cls._sdk_candidates())
        raise FileNotFoundError(
            "Cannot find reBotArm_control_py. Clone it into one of:\n"
            f"{candidates}"
        )

    @staticmethod
    def _cfg_with_channel(cfg_path: Path, channel: str, filename: str) -> Path:
        if not channel:
            return cfg_path
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["channel"] = channel
        tmp_dir = Path("/tmp") / "rebotarm_ros2"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / filename
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return tmp_path

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    @property
    def endpos_ctrl(self):
        return self._endpos_ctrl

    @property
    def joint_names(self) -> list[str]:
        return list(self._arm.joint_names)

    @property
    def mode(self) -> str:
        return str(self._arm.mode)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def control_loop_active(self) -> bool:
        return bool(self._arm.control_loop_active)

    @property
    def has_gripper(self) -> bool:
        return self._gripper_mot is not None

    @property
    def state_machine(self) -> str:
        return self._state_machine

    @property
    def error_codes(self) -> list[str]:
        return list(self._error_codes)

    def set_state_machine(self, state: str) -> None:
        if state not in ("IDLE", "TRAJ_RUNNING", "LOWLEVEL_STREAMING", "GRAVITY_COMP"):
            raise ValueError(f"unsupported state machine value: {state}")
        self._state_machine = state

    # ------------------------------------------------------------------
    # arm
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._connected:
            return
        self._endpos_ctrl.start()
        self._connected = True
        self._enabled = True
        self.init_gripper(str(self._gripper_cfg_path))

    def shutdown(
        self,
        safe_home: bool = True,
        disable_after_safe_home: bool = True,
    ) -> None:
        if not self._connected:
            return
        try:
            self.stop_gravity_compensation()

            if safe_home:
                self.safe_home()

            if disable_after_safe_home:
                self.disable()

            self.disconnect_gripper()
            if self._endpos_ctrl._running:
                self._endpos_ctrl.end()
            else:
                self._arm.disconnect()
        finally:
            self._connected = False
            self._enabled = False
            self.set_state_machine("IDLE")

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._arm.get_state()

    def get_joint_positions(self, request: bool = False) -> np.ndarray:
        return self._arm.get_positions(request=request)

    def get_joint_velocities(self, request: bool = False) -> np.ndarray:
        return self._arm.get_velocities(request=request)

    def hold_current_position(self) -> np.ndarray:
        q, _, _ = self.get_joint_state()
        current = np.array(q, dtype=np.float64, copy=True)
        self._endpos_ctrl._q_target[:] = current
        return current

    def set_joint_position_target(self, positions) -> None:
        target = np.asarray(positions, dtype=np.float64).reshape(-1)
        if len(target) != self._arm.num_joints:
            raise ValueError(
                f"expected {self._arm.num_joints} joint targets, got {len(target)}"
            )
        self._endpos_ctrl._stop_send.set()
        if self._endpos_ctrl._send_thread is not None:
            self._endpos_ctrl._send_thread.join()
        self._endpos_ctrl._moving = False
        self._endpos_ctrl._q_target[:] = target

    def start_endpos_control(self) -> None:
        if self._gravity_comp_active:
            raise RuntimeError("stop gravity compensation before starting endpos control")

        if self.mode != "pos_vel" or not self.control_loop_active:
            self._arm.stop_control_loop()
            self._endpos_ctrl._running = False
            self._endpos_ctrl.start()
        else:
            self._arm.enable()
            self._endpos_ctrl._running = True
            self.hold_current_position()
        self._enabled = True
        self.set_state_machine("IDLE")

    def enable(self) -> None:
        self.start_endpos_control()
        if self._gripper_ctrl is not None:
            self._gripper_ctrl.enable_all()

    def disable(self) -> None:
        if self._gravity_comp_active:
            raise RuntimeError("stop gravity compensation before disable")
        self._arm.disable()
        self._endpos_ctrl._stop_send.set()
        self._endpos_ctrl._moving = False
        self._endpos_ctrl._running = False
        self._enabled = False
        self.set_state_machine("IDLE")

    def safe_home(self) -> None:
        self.stop_gravity_compensation()
        if self.has_gripper:
            self.set_gripper_position(_GRIPPER_CLOSED_POSITION)
        self.start_endpos_control()
        self._endpos_ctrl.safe_home()

    def set_mode(self, mode: str) -> bool:
        mode = mode.strip().lower()
        if mode not in ("mit", "pos_vel", "vel"):
            raise ValueError(f"unsupported mode: {mode}")
        self.stop_gravity_compensation()
        self._arm.stop_control_loop()
        self._endpos_ctrl._running = False

        if mode == "mit":
            ok = self._arm.mode_mit()
        elif mode == "pos_vel":
            ok = self._arm.mode_pos_vel()
            if self._enabled:
                self._start_pos_vel_hold()
        else:
            ok = self._arm.mode_vel()
        self.set_state_machine("IDLE")
        return bool(ok)

    def set_zero(self, joint_name: str = "") -> bool:
        self._arm.stop_control_loop()
        self._endpos_ctrl._running = False
        if joint_name:
            ok = self._arm.set_zero_single(joint_name)
        else:
            self._arm.set_zero()
            ok = True
        self._enabled = False
        self.set_state_machine("IDLE")
        return bool(ok)

    def send_joint_mit_cmd(
        self,
        joint_name: str,
        pos: float,
        vel: float,
        kp: float,
        kd: float,
        tau: float,
    ) -> None:
        index = self._joint_index(joint_name)
        self._begin_lowlevel_streaming("mit")
        q, _, _ = self._arm.get_state()
        target_pos = np.array(q, dtype=np.float64, copy=True)
        target_vel = np.zeros(self._arm.num_joints, dtype=np.float64)
        target_tau = np.zeros(self._arm.num_joints, dtype=np.float64)
        target_kp = np.array([joint.kp for joint in self._arm._joints], dtype=np.float64)
        target_kd = np.array([joint.kd for joint in self._arm._joints], dtype=np.float64)
        target_pos[index] = float(pos)
        target_vel[index] = float(vel)
        target_kp[index] = float(kp)
        target_kd[index] = float(kd)
        target_tau[index] = float(tau)
        self._arm.mit(target_pos, target_vel, target_kp, target_kd, target_tau)
        self.set_state_machine("LOWLEVEL_STREAMING")

    def send_joint_pos_vel_cmd(
        self,
        joint_name: str,
        pos: float,
        vlim: float,
    ) -> None:
        index = self._joint_index(joint_name)
        self._begin_lowlevel_streaming("pos_vel")
        q, _, _ = self._arm.get_state()
        target_pos = np.array(q, dtype=np.float64, copy=True)
        target_vlim = np.array(
            [joint.vlim for joint in self._arm._joints],
            dtype=np.float64,
        )
        target_pos[index] = float(pos)
        target_vlim[index] = float(vlim)
        self._arm.pos_vel(target_pos, target_vlim)
        self.set_state_machine("LOWLEVEL_STREAMING")

    def send_joint_vel_cmd(self, joint_name: str, vel: float) -> None:
        index = self._joint_index(joint_name)
        self._begin_lowlevel_streaming("vel")
        target_vel = np.zeros(self._arm.num_joints, dtype=np.float64)
        target_vel[index] = float(vel)
        self._arm.set_vel(target_vel)
        self.set_state_machine("LOWLEVEL_STREAMING")

    def current_pose(self):
        from reBotArm_control_py.kinematics import compute_fk

        q, _, _ = self.get_joint_state()
        position, rotation, _ = compute_fk(self._gc_model, q)
        return fk_to_pose(position, rotation)

    def get_joint_status_codes(self) -> list[int]:
        codes: list[int] = []
        for name in self.joint_names:
            try:
                st = self._arm._motor_map[name].get_state()
                codes.append(int(st.status_code if st is not None else 0))
            except Exception:
                codes.append(0)
        return codes

    # ------------------------------------------------------------------
    # gravity compensation
    # ------------------------------------------------------------------

    def start_gravity_compensation(self) -> None:
        self.stop_gravity_compensation()
        if not self._enabled:
            self._arm.enable()
            self._enabled = True
        self._arm.stop_control_loop()
        self._endpos_ctrl._stop_send.set()
        self._endpos_ctrl._moving = False
        self._endpos_ctrl._running = False

        self._gravity_comp_q_target = self._arm.get_positions(request=True).copy()
        self._gravity_comp_q_last = self._gravity_comp_q_target.copy()
        self._arm.mode_mit(
            kp=np.full(self._arm.num_joints, _GC_KP, dtype=np.float64),
            kd=np.full(self._arm.num_joints, _GC_KD, dtype=np.float64),
        )
        self._gravity_comp_integral = np.zeros_like(self._gravity_comp_q_target)
        self._gravity_comp_lock_counter = 0
        self._gravity_comp_active = True
        arm_rate = float(getattr(self._arm, "_rate", 500.0))
        self._gravity_comp_tick(self._arm, 1.0 / arm_rate)
        self._arm.start_control_loop(self._gravity_comp_tick, rate=arm_rate)
        self.set_state_machine("GRAVITY_COMP")

    def stop_gravity_compensation(self) -> None:
        if not self._gravity_comp_active:
            return
        hold_target = (
            self._gravity_comp_q_last.copy()
            if self._gravity_comp_q_last is not None
            else None
        )
        self._arm.stop_control_loop()
        self._gravity_comp_active = False
        self._gravity_comp_q_target = None
        self._gravity_comp_integral = None
        self._gravity_comp_lock_counter = 0
        self._gravity_comp_q_last = None
        if self._enabled:
            self._arm.mode_pos_vel()
            self._start_pos_vel_hold(target=hold_target)
        self.set_state_machine("IDLE")

    def gravity_compensation_active(self) -> bool:
        return self._gravity_comp_active

    def gravity_compensation_target(self) -> np.ndarray | None:
        if self._gravity_comp_q_target is None:
            return None
        return self._gravity_comp_q_target.copy()

    @staticmethod
    def _angles_near_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        delta = values - reference
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        return reference + delta

    def _read_gravity_comp_positions(
        self,
        *,
        request: bool = False,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        q = self._arm.get_positions(request=request)
        ref = reference if reference is not None else self._gravity_comp_q_last
        if ref is not None:
            q = self._angles_near_reference(q, ref)
        self._gravity_comp_q_last = np.array(q, dtype=np.float64, copy=True)
        return self._gravity_comp_q_last.copy()

    def _gravity_comp_tick(self, arm, dt: float) -> None:
        del dt
        if not self._gravity_comp_active or self._gravity_comp_q_target is None:
            return

        q = self._read_gravity_comp_positions()
        qd = arm.get_velocities()
        tau_g = self._gc_compute_generalized_gravity(q=q)

        q_error = self._gravity_comp_q_target - q
        if self._gravity_comp_integral is None:
            self._gravity_comp_integral = np.zeros_like(q)
        self._gravity_comp_integral += q_error * 1.0
        np.clip(self._gravity_comp_integral, -0.5, 0.5, out=self._gravity_comp_integral)

        self._gc_pin.computeJointJacobians(self._gc_model, self._gc_data, q)
        self._gc_pin.updateFramePlacements(self._gc_model, self._gc_data)
        jacobian = self._gc_pin.getFrameJacobian(
            self._gc_model,
            self._gc_data,
            self._gc_ee_frame_id,
            self._gc_pin.ReferenceFrame.WORLD,
        )
        spatial_velocity = jacobian @ qd
        linear_speed = float(np.linalg.norm(spatial_velocity[:3]))
        angular_speed = float(np.linalg.norm(spatial_velocity[3:]))

        if linear_speed > _GC_VEL_THRESHOLD or angular_speed > _GC_W_VEL_THRESHOLD:
            self._gravity_comp_q_target = q.copy()
            self._gravity_comp_lock_counter = 0
            self._gravity_comp_integral *= 0.9
        else:
            self._gravity_comp_lock_counter += 1

        arm.mit(
            pos=self._gravity_comp_q_target,
            vel=np.zeros(arm.num_joints),
            kp=np.full(arm.num_joints, _GC_KP),
            kd=np.full(arm.num_joints, _GC_KD),
            tau=tau_g + self._gravity_comp_integral,
        )

    # ------------------------------------------------------------------
    # gripper
    # ------------------------------------------------------------------

    def init_gripper(self, cfg_path: str) -> None:
        from reBotArm_control_py.actuator.gripper import load_cfg as load_gripper_cfg

        gcfg = load_gripper_cfg(cfg_path)
        gc = gcfg["gripper"]
        vendor = gc.vendor
        if vendor not in self._arm._ctrl_map:
            raise RuntimeError(
                f"gripper vendor={vendor!r} cannot share arm Controller"
            )

        ctrl = self._arm._ctrl_map[vendor]
        if vendor == "damiao":
            mot = ctrl.add_damiao_motor(gc.motor_id, gc.feedback_id, gc.model)
        elif vendor == "myactuator":
            mot = ctrl.add_myactuator_motor(gc.motor_id, gc.feedback_id, gc.model)
        elif vendor == "robstride":
            mot = ctrl.add_robstride_motor(gc.motor_id, gc.feedback_id, gc.model)
        else:
            raise ValueError(f"unsupported gripper vendor: {vendor!r}")

        self._gripper_cfg = gc
        self._gripper_ctrl = ctrl
        self._gripper_mot = mot
        self._patch_shared_bus(ctrl, mot)

        ctrl.enable_all()
        self._set_gripper_mode("pos_vel")
        self._gripper_target_position = self.get_gripper_state()[0]

    def disconnect_gripper(self) -> None:
        if self._gripper_mot is None:
            return
        self._gripper_mot = None
        self._gripper_ctrl = None
        self._gripper_cfg = None
        self._gripper_target_position = None

    def set_gripper_target(self, position: float) -> None:
        self._begin_gripper_command()
        self._set_gripper_mode("pos_vel")
        self._gripper_mot.send_pos_vel(float(position), float(self._gripper_cfg.vlim))
        self._gripper_request_and_poll()
        self._gripper_target_position = float(position)

    def wait_gripper_target(self, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.gripper_reached_target():
                return True
            time.sleep(0.02)
        return False

    def set_gripper_position(
        self,
        position: float,
        timeout: float = 3.0,
    ) -> tuple[bool, float]:
        self.set_gripper_target(position)
        reached = self.wait_gripper_target(timeout)
        return reached, self.get_gripper_state()[0]

    def get_gripper_state(self) -> tuple[float, float, float, int]:
        if self._gripper_mot is None:
            return 0.0, 0.0, 0.0, 0
        self._gripper_request_and_poll()
        status = 0
        pos = vel = torque = 0.0
        try:
            st = self._gripper_mot.get_state()
            if st is not None:
                pos = float(st.pos)
                vel = float(st.vel)
                torque = float(st.torq)
                status = int(st.status_code)
        except Exception:
            status = 0
        return float(pos), float(vel), float(torque), status

    def gripper_reached_target(self) -> bool:
        if self._gripper_target_position is None:
            return True
        pos = self.get_gripper_state()[0]
        return abs(pos - self._gripper_target_position) < _GRIPPER_GOAL_TOLERANCE_RAD

    def send_gripper_mit_cmd(
        self,
        pos: float,
        vel: float,
        kp: float,
        kd: float,
        tau: float,
    ) -> None:
        self._begin_gripper_command()
        self._set_gripper_mode("mit", kp=float(kp), kd=float(kd))
        self._gripper_mot.send_mit(
            float(pos),
            float(vel),
            float(kp),
            float(kd),
            float(tau),
        )
        self._gripper_request_and_poll()
        self._gripper_target_position = None

    def send_gripper_pos_vel_cmd(self, pos: float, vlim: float) -> None:
        self._begin_gripper_command()
        self._set_gripper_mode("pos_vel")
        self._gripper_mot.send_pos_vel(float(pos), float(vlim))
        self._gripper_request_and_poll()
        self._gripper_target_position = None

    def send_gripper_vel_cmd(self, vel: float) -> None:
        self._begin_gripper_command()
        self._set_gripper_mode("vel")
        self._gripper_mot.send_vel(float(vel))
        self._gripper_request_and_poll()
        self._gripper_target_position = None

    def _begin_gripper_command(self) -> None:
        if not self._enabled:
            raise RuntimeError("rejecting gripper command while arm is disabled")
        if self._gravity_comp_active or self.state_machine == "GRAVITY_COMP":
            raise RuntimeError("rejecting gripper command during gravity compensation")
        if self.state_machine == "TRAJ_RUNNING":
            raise RuntimeError("rejecting gripper command while trajectory is running")
        if self._gripper_mot is None:
            raise RuntimeError("gripper is not initialized")

    def _set_gripper_mode(
        self,
        mode: str,
        *,
        kp: float | None = None,
        kd: float | None = None,
    ) -> None:
        from motorbridge import Mode

        if mode == self._gripper_mode:
            return
        if mode == "mit":
            if kp is not None:
                self._gripper_cfg.kp = float(kp)
            if kd is not None:
                self._gripper_cfg.kd = float(kd)
            self._gripper_mot.ensure_mode(Mode.MIT, 1000)
        elif mode == "pos_vel":
            self._gripper_mot.write_register_f32(25, self._gripper_cfg.vel_kp)
            self._gripper_mot.write_register_f32(26, self._gripper_cfg.vel_ki)
            self._gripper_mot.write_register_f32(27, self._gripper_cfg.pos_kp)
            self._gripper_mot.write_register_f32(28, self._gripper_cfg.pos_ki)
            time.sleep(0.02)
            self._gripper_mot.ensure_mode(Mode.POS_VEL, 1000)
        elif mode == "vel":
            self._gripper_mot.ensure_mode(Mode.VEL, 1000)
        else:
            raise ValueError(f"unsupported gripper mode: {mode}")
        self._gripper_mode = mode
        time.sleep(0.2)

    def _gripper_request_and_poll(self) -> None:
        try:
            self._gripper_mot.request_feedback()
            self._gripper_ctrl.poll_feedback_once()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _begin_lowlevel_streaming(self, required_mode: str) -> None:
        if not self._enabled:
            raise RuntimeError("rejecting low-level command while arm is disabled")
        if self._gravity_comp_active or self.state_machine == "GRAVITY_COMP":
            raise RuntimeError("rejecting low-level command during gravity compensation")
        if self.state_machine == "TRAJ_RUNNING":
            self._endpos_ctrl._stop_send.set()
            self._endpos_ctrl._moving = False
        self._arm.stop_control_loop()
        self._endpos_ctrl._running = False

        if required_mode == self.mode:
            self.set_state_machine("LOWLEVEL_STREAMING")
            return
        if required_mode == "mit":
            ok = self._arm.mode_mit()
        elif required_mode == "pos_vel":
            ok = self._arm.mode_pos_vel()
        elif required_mode == "vel":
            ok = self._arm.mode_vel()
        else:
            raise ValueError(f"unsupported low-level mode: {required_mode}")
        if not ok:
            raise RuntimeError(f"not all arm joints entered {required_mode} mode")
        self.set_state_machine("LOWLEVEL_STREAMING")

    def _start_pos_vel_hold(self, target: np.ndarray | None = None) -> None:
        if self.control_loop_active:
            return
        if target is None:
            self.hold_current_position()
        else:
            self._endpos_ctrl._q_target[:] = np.array(target, dtype=np.float64)
        self._endpos_ctrl._running = True
        self._arm.start_control_loop(self._endpos_ctrl._loop_cb)

    def _joint_index(self, joint_name: str) -> int:
        try:
            return self.joint_names.index(joint_name)
        except ValueError as exc:
            raise KeyError(f"unknown joint: {joint_name}") from exc

    def _patch_shared_bus(self, ctrl, gripper_mot) -> None:
        if not hasattr(ctrl, "_bus_lock"):
            ctrl._bus_lock = threading.RLock()
        lock = ctrl._bus_lock

        def locked(fn):
            def _wrapped(*args, **kwargs):
                with lock:
                    return fn(*args, **kwargs)

            return _wrapped

        if not getattr(ctrl, "_rebotarm_ros_bus_lock_patched", False):
            ctrl.poll_feedback_once = locked(ctrl.poll_feedback_once)
            ctrl.enable_all = locked(ctrl.enable_all)
            ctrl.disable_all = locked(ctrl.disable_all)
            ctrl._rebotarm_ros_bus_lock_patched = True

        if not getattr(self._arm, "_rebotarm_ros_bus_lock_patched", False):
            for joint in self._arm._joints:
                mot = self._arm._motor_map[joint.name]
                self._patch_motor_bus(mot, locked)
            self._arm._rebotarm_ros_bus_lock_patched = True

        self._patch_motor_bus(gripper_mot, locked)

    @staticmethod
    def _patch_motor_bus(mot, locked) -> None:
        if getattr(mot, "_rebotarm_ros_bus_lock_patched", False):
            return
        for name in (
            "send_pos_vel",
            "send_mit",
            "send_vel",
            "request_feedback",
            "write_register_f32",
            "ensure_mode",
            "set_zero_position",
        ):
            if hasattr(mot, name):
                setattr(mot, name, locked(getattr(mot, name)))
        mot._rebotarm_ros_bus_lock_patched = True
