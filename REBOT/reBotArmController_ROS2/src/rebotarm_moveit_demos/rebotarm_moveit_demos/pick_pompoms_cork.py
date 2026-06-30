from __future__ import annotations

import math
import sys

from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    MoveItErrorCodes,
    PlanningScene,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Joy
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf_transformations import quaternion_about_axis, quaternion_multiply
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from vision_msgs.msg import Detection3DArray
import tf2_ros

from rebotarm_moveit_demos.demo_common import MoveItDemoBase

# Button indices of the RvizVisualToolsGui panel (sensor_msgs/Joy on /rviz_visual_tools_gui).
_BTN_NEXT = 1  # run the next step of the pick
_BTN_CONTINUE = 2  # lift back up to the hover pose and reset
_BTN_STOP = 4  # quit the node


class PickPompomsCork(MoveItDemoBase):
    """Top-down pick&place of corks and pompoms from RGB-D detections.

    Phases: PREHOVER (far look) -> HOVER -> DESCEND (+close+attach) -> LIFT -> PLACE -> release.
    PREHOVER aims gripper_tcp (camera un-occluded); HOVER/DESCEND aim grasp_tcp (the fingertips).
    Two run modes (run_mode): 'manual' steps with the RvizVisualToolsGui buttons (Next/Continue/
    Stop); 'auto' clears the whole scene object by object without buttons.

    Detections are timestamped: a stale or empty one is treated as "object not in view", so the
    node never plans toward a last-known pose after the object is lifted/dropped/out of frame.
    """

    def __init__(self) -> None:
        super().__init__("pick_pompoms_cork")
        # Planning, planning scene, ghost display and the driver's safe_home service.
        self._planner = self.node.create_client(GetMotionPlan, "/plan_kinematic_path")
        self._planning_scene = self.node.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self._display_pub = self.node.create_publisher(
            DisplayTrajectory, "/display_planned_path", 1
        )
        self._safe_home = self.node.create_client(
            Trigger, str(self._param("safe_home_service"))
        )
        # Buttons and perception.
        self._gui_button: int | None = None
        self.node.create_subscription(Joy, "/rviz_visual_tools_gui", self._gui_cb, 10)
        self._latest_det = None
        self._latest_dets = []  # full Detection3DArray contents (for multi-object auto)
        self._latest_det_stamp = None
        self._latest_det_empty = True
        self.node.create_subscription(
            Detection3DArray,
            str(self._param("perception_topic")),
            self._det_cb,
            10,
        )
        # TF (read gripper_tcp height for the descent), gripper action clients and pick state.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)
        self._gripper_trajectory = ActionClient(
            self.node, FollowJointTrajectory, str(self._param("gripper_action_name"))
        )
        self._gripper_command = ActionClient(
            self.node, GripperCommand, str(self._param("hardware_gripper_action_name"))
        )
        self._gripper = None
        self._gripper_kind = None
        self._hover_joints = None  # arm config at hover, captured before the descent

    def run(self) -> bool:
        if not self._wait_for_dependencies():
            return False
        if str(self._param("run_mode")).lower() == "auto":
            return self._run_auto()
        return self._run_manual()

    def _wait_for_dependencies(self) -> bool:
        if not self._planner.wait_for_service(timeout_sec=10.0):
            self.node.get_logger().error("/plan_kinematic_path unavailable")
            return False
        if not self._planning_scene.wait_for_service(timeout_sec=10.0):
            self.node.get_logger().error("/apply_planning_scene unavailable")
            return False
        if not self.wait_for_ik_service() or not self.wait_for_execute_server():
            return False
        if bool(self._param("use_gripper")) and not self._wait_for_gripper_server():
            self.node.get_logger().error("gripper action unavailable")
            return False
        return True

    def _run_manual(self) -> bool:
        self.node.get_logger().info(
            "pick active (manual). Next=run step (prehover -> hover -> grasp -> lift), "
            "Continue=back to hover, Stop=quit."
        )

        phase = "PREHOVER"
        cached_traj = None
        planned_target = None  # full [x, y, z] of the cached plan / committed measurement
        cached_size = None
        cached_class = None
        waited_logged = False

        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

            button = self._take_button()
            if button == _BTN_STOP:
                self.node.get_logger().info("Stop: quitting")
                return True
            if button == _BTN_CONTINUE:
                # Back up to hover and reset; no safe_home and no gripper command (safe_home
                # force-closes the gripper and would crush a held object).
                lift = self._plan_lift()
                if lift is not None:
                    self.node.get_logger().info("Continue: back to hover (gripper untouched)")
                    self._execute_aligned(lift)
                phase, cached_traj, planned_target = "PREHOVER", None, None
                cached_size, cached_class = None, None
                waited_logged = False
                self._invalidate_detection()
                continue
            if button == _BTN_NEXT:
                if phase == "DONE":
                    # Release (open + detach + clean scene), then safe_home. Forget the stale
                    # detection so we wait for a fresh, in-view one before re-picking.
                    self.node.get_logger().info("Next: releasing object and going home")
                    self._command_gripper("open")
                    self._reset_scene()
                    self._return_home()
                    phase, cached_traj, planned_target = "PREHOVER", None, None
                    cached_size, cached_class = None, None
                    waited_logged = False
                    self._invalidate_detection()
                    continue
                if cached_traj is None:
                    self.node.get_logger().warn("Next with no trajectory ready yet")
                    continue
                if not self._run_next(phase, cached_traj):
                    self.node.get_logger().error(f"{phase} step failed")
                    return False
                phase, cached_traj, planned_target = self._advance(
                    phase, planned_target, cached_size, cached_class
                )
                continue

            # PREHOVER/HOVER plan from the live detection; later phases run the plan cached at
            # the transition (the object leaves the view at hover, so we do not re-measure).
            if phase not in ("PREHOVER", "HOVER"):
                continue

            target, size, object_class = self._current_target()
            if target is None:
                if not waited_logged:
                    self.node.get_logger().info("waiting for a fresh detection ...")
                    waited_logged = True
                continue
            waited_logged = False

            if not self._should_replan(target, planned_target):
                continue
            planner = self._plan_prehover if phase == "PREHOVER" else self._plan_hover
            trajectory = planner(target, size, object_class)
            if trajectory is not None:
                cached_traj, planned_target = trajectory, target
                cached_size, cached_class = size, object_class

        return True

    # ---- auto mode ----------------------------------------------------------------------

    def _run_auto(self) -> bool:
        """Pick&place every detected object, nearest first, until the scene is clear.

        Objects are tracked by perception track_id; placed (or given-up) ids are remembered so a
        still-visible object is never re-picked. Stop aborts between phases. Returns True on a
        clean finish/Stop.
        """
        loop = bool(self._param("auto_loop"))
        max_attempts = max(1, int(self._param("max_pick_attempts")))
        self.node.get_logger().info(
            f"pick active (auto, {'clear-scene' if loop else 'single object'}); Stop aborts."
        )
        done_ids: set[str] = set()  # placed or given-up: never re-picked
        attempts: dict[str, int] = {}
        while rclpy.ok():
            self._invalidate_detection()
            selection = self._auto_wait_selection(done_ids)
            if selection == "STOP":
                self.node.get_logger().info("Stop: quitting")
                return True
            if selection is None:
                self.node.get_logger().info("auto: no more objects to pick")
                break
            track_id, target, size, object_class = selection
            attempts[track_id] = attempts.get(track_id, 0) + 1
            self.node.get_logger().info(
                f"auto: next object {object_class}#{track_id} "
                f"at [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] "
                f"(attempt {attempts[track_id]}/{max_attempts})"
            )
            result = self._auto_pick_place(track_id, target, size, object_class)
            if result == "STOP":
                return True
            if not result:
                # One object failing must not kill the run: recover, then retry or skip it.
                self._auto_recover()
                if attempts[track_id] >= max_attempts:
                    self.node.get_logger().warn(
                        f"auto: giving up on {object_class}#{track_id} after "
                        f"{attempts[track_id]} attempt(s); skipping it"
                    )
                    done_ids.add(track_id)
                else:
                    self.node.get_logger().warn(
                        f"auto: {object_class}#{track_id} failed; will retry"
                    )
                continue
            done_ids.add(track_id)
            if not loop:
                break

        self._return_home()
        self._invalidate_detection()
        self.node.get_logger().info("auto run complete")
        return True

    def _auto_recover(self) -> None:
        """Cleanup after a failed object: release, clear the scene and go home."""
        self.node.get_logger().info("auto: recovering to a clean home state")
        self._command_gripper("open")
        self._reset_scene()
        self._return_home()
        self._invalidate_detection()

    def _auto_pick_place(self, track_id, target, size, object_class):
        """One object end to end: approach -> grasp -> lift -> place -> release -> home.
        Returns True on success, False on failure, or 'STOP' if aborted."""
        committed = self._auto_approach(track_id, target, size, object_class)
        if committed == "STOP":
            return "STOP"
        if committed is None:
            return False
        target, size, object_class = committed

        if self._stop_requested():
            return "STOP"
        if not self._auto_grasp(target, size, object_class):
            return False

        if self._stop_requested():
            return "STOP"
        if not self._auto_execute(self._plan_lift(), "lift"):
            return False

        if bool(self._param("use_place")):
            place_traj = self._plan_place(object_class)
            if place_traj is not None:
                if self._stop_requested():
                    return "STOP"
                if not self._auto_execute(place_traj, "place"):
                    return False
            else:
                self.node.get_logger().warn(
                    "place unavailable/unset; releasing at the lift pose instead"
                )

        self.node.get_logger().info(f"auto: releasing {object_class}#{track_id} and going home")
        self._command_gripper("open")
        self._reset_scene()
        self._return_home()
        return True

    def _auto_approach(self, track_id, target, size, object_class):
        """PREHOVER then HOVER on the initial-scan measurement, committed for the descent.

        The object is not re-observable from the prehover/hover poses (gripper occlusion + it
        leaves the eye-in-hand FOV), so we keep the fresh home-scan measurement for the whole
        approach. Returns the committed (target, size, class), None on failure, or 'STOP'.
        """
        if not self._auto_execute(self._plan_prehover(target, size, object_class), "prehover"):
            return None
        if self._stop_requested():
            return "STOP"
        if not self._auto_execute(self._plan_hover(target, size, object_class), "hover"):
            return None
        return target, size, object_class

    def _auto_grasp(self, target, size, object_class) -> bool:
        """Open, descend, close on the object and attach it."""
        self._command_gripper("open")
        descent = self._plan_descent(target, size, object_class)
        if not self._auto_execute(descent, "descent"):
            return False
        width = size[1] if size else None
        self._command_gripper("grasp", width, object_class)
        self._attach_object(target, size, object_class)
        return True

    def _auto_execute(self, trajectory, label: str) -> bool:
        if trajectory is None:
            self.node.get_logger().error(f"auto: no {label} trajectory; aborting")
            return False
        self.node.get_logger().info(f"auto: executing {label}")
        return self._execute_aligned(trajectory)

    def _auto_wait_selection(self, exclude_ids, reference=None):
        """Spin until a fresh detection yields an unplaced candidate (nearest first). Returns
        (track_id, target, size, class), None when none appears within auto_done_timeout (scene
        clear), or 'STOP' if Stop is pressed."""
        timeout = float(self._param("auto_done_timeout"))
        deadline = self.node.get_clock().now() + Duration(seconds=timeout)
        logged = False
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self._take_button() == _BTN_STOP:
                return "STOP"
            selection = self._select_nearest(exclude_ids, reference)
            if selection is not None:
                return selection
            if not logged:
                self.node.get_logger().info("auto: scanning for objects ...")
                logged = True
            if self.node.get_clock().now() >= deadline:
                return None
        return None

    def _stop_requested(self) -> bool:
        """Non-blocking check for the Stop button between auto phases."""
        rclpy.spin_once(self.node, timeout_sec=0.0)
        if self._take_button() == _BTN_STOP:
            self.node.get_logger().info("Stop: aborting auto run")
            return True
        return False

    def _run_next(self, phase: str, trajectory) -> bool:
        label = {
            "PREHOVER": "prehover",
            "HOVER": "hover",
            "DESCEND": "descent",
            "LIFT": "lift",
            "PLACE": "place",
        }[phase]
        self.node.get_logger().info(f"Next: executing {label}")
        return self._execute_aligned(trajectory)

    def _execute_aligned(self, trajectory) -> bool:
        """Execute a (possibly cached) trajectory, snapping its start to the current joints first."""
        self._align_trajectory_start(trajectory)
        return self.execute_trajectory(trajectory, float(self._param("result_timeout")))

    def _align_trajectory_start(self, trajectory) -> None:
        """Overwrite the trajectory's first point with the current joints.

        Plans are computed on entering a phase but executed later (on Next); the arm may drift
        meanwhile and move_group rejects a start that deviates from the current state. Snapping
        absorbs that drift into the first segment.
        """
        jt = trajectory.joint_trajectory
        if not jt.points:
            return
        positions = list(jt.points[0].positions)
        for index, name in enumerate(jt.joint_names):
            if name in self._latest_joint_positions:
                positions[index] = self._latest_joint_positions[name]
        jt.points[0].positions = positions

    def _advance(self, phase, committed_target, size, object_class):
        """Return (next_phase, cached_traj, planned_target) after a Next executed `phase`."""
        if phase == "PREHOVER":
            self.node.get_logger().info("prehover done; planning hover (Next to run)")
            return "HOVER", None, None
        if phase == "HOVER":
            # Last visual measurement is committed: plan the descent now (object out of view).
            self._command_gripper("open")
            descent_traj = self._plan_descent(committed_target, size, object_class)
            self.node.get_logger().info("hover done; planning descent (Next to run)")
            return "DESCEND", descent_traj, committed_target
        if phase == "DESCEND":
            width = size[1] if size else None
            self._command_gripper("grasp", width, object_class)
            self._attach_object(committed_target, size, object_class)
            lift_traj = self._plan_lift()
            self.node.get_logger().info("grasped + attached; planning lift (Next to run)")
            return "LIFT", lift_traj, committed_target
        if phase == "LIFT":
            # Optionally move (object held) to a fixed, measured drop config before releasing.
            if bool(self._param("use_place")):
                place_traj = self._plan_place(object_class)
                if place_traj is not None:
                    self.node.get_logger().info(
                        "lifted; planning place at the fixed drop config (Next to run)"
                    )
                    return "PLACE", place_traj, committed_target
                self.node.get_logger().warn(
                    "place unavailable/unset; will release at hover instead"
                )
            self.node.get_logger().info(
                "object lifted at hover. Next=open/release+safe_home, "
                "Continue=back to hover, Stop=quit."
            )
            return "DONE", None, None
        if phase == "PLACE":
            self.node.get_logger().info(
                "at the fixed drop config (object held). Next=open/release+safe_home, "
                "Continue=back to hover, Stop=quit."
            )
            return "DONE", None, None
        return phase, None, None

    # ---- perception + buttons -----------------------------------------------------------

    def _gui_cb(self, msg: Joy) -> None:
        for index, value in enumerate(msg.buttons):
            if value:
                self._gui_button = index

    def _det_cb(self, msg: Detection3DArray) -> None:
        # Stamp every message (even empty) so stale/empty data can be rejected as "not in view".
        self._latest_det_stamp = self.node.get_clock().now()
        self._latest_dets = list(msg.detections)
        if msg.detections:
            self._latest_det = msg.detections[0]
            self._latest_det_empty = False
        else:
            self._latest_det_empty = True

    def _take_button(self) -> int | None:
        button = self._gui_button
        self._gui_button = None
        return button

    def _invalidate_detection(self) -> None:
        """Forget the last detection so the loop waits for a fresh, in-view one."""
        self._latest_det = None
        self._latest_dets = []
        self._latest_det_stamp = None
        self._latest_det_empty = True

    def _fresh_detections(self):
        """The latest detection list if fresh (within detection_timeout) and non-empty, else None."""
        if self._latest_det_stamp is None or not self._latest_dets:
            return None
        age = (self.node.get_clock().now() - self._latest_det_stamp).nanoseconds * 1e-9
        if age > float(self._param("detection_timeout")):
            return None
        return self._latest_dets

    @staticmethod
    def _det_fields(det):
        """(track_id, [x,y,z], [sx,sy,sz], class) for one Detection3D."""
        c, s = det.bbox.center.position, det.bbox.size
        cls = det.results[0].hypothesis.class_id if det.results else "unknown"
        return str(det.id), [c.x, c.y, c.z], [s.x, s.y, s.z], cls

    def _select_nearest(self, exclude_ids, reference=None):
        """Nearest unplaced detection to `reference` (defaults to the current grasp_tcp position).
        Returns (track_id, target, size, class) or None if no fresh/eligible detection."""
        dets = self._fresh_detections()
        if not dets:
            return None
        if reference is None:
            reference = self._link_translation(str(self._param("ik_link_name"))) or (0.0, 0.0, 0.0)
        best = None
        best_dist = float("inf")
        for det in dets:
            track_id, target, size, cls = self._det_fields(det)
            if track_id in exclude_ids:
                continue
            dist = math.dist(target, list(reference))
            if dist < best_dist:
                best_dist, best = dist, (track_id, target, size, cls)
        return best

    def _current_target(self):
        if not bool(self._param("use_perception")):
            target = [float(v) for v in self._param("target")]
            size = [float(v) for v in self._param("object_size")]
            return target, size, str(self._param("object_class"))
        # Need a fresh, non-empty detection; a stale/empty one means "not in view".
        if (
            self._latest_det is None
            or self._latest_det_empty
            or self._latest_det_stamp is None
        ):
            return None, None, None
        age = (self.node.get_clock().now() - self._latest_det_stamp).nanoseconds * 1e-9
        if age > float(self._param("detection_timeout")):
            return None, None, None
        det = self._latest_det
        c, s = det.bbox.center.position, det.bbox.size
        cls = det.results[0].hypothesis.class_id if det.results else "unknown"
        return [c.x, c.y, c.z], [s.x, s.y, s.z], cls

    def _should_replan(self, target, planned_target) -> bool:
        if planned_target is None:
            return True
        return math.dist(target, planned_target) > float(self._param("replan_min_move"))

    def _return_home(self) -> None:
        if not self._safe_home.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().error(
                f"service {self._param('safe_home_service')} unavailable (driver up?)"
            )
            return
        self.node.get_logger().info("returning home (safe_home) ...")
        future = self._safe_home.call_async(Trigger.Request())
        if not self.wait(future, float(self._param("safe_home_timeout"))):
            self.node.get_logger().error("safe_home did not respond in time")
            return
        response = future.result()
        if response is not None and response.success:
            self.node.get_logger().info("home reached")
        else:
            message = response.message if response is not None else "empty response"
            self.node.get_logger().warn(f"safe_home failed: {message}")

    # ---- planning -------------------------------------------------------------------------

    def _plan_prehover(self, target, size, object_class):
        """Far, high look pose: aims gripper_tcp so the camera sees the object un-occluded."""
        if not self._add_collision_object(target, size, object_class):
            return None
        seed = [float(v) for v in self._param("prehover_seed")]
        current = self.current_joint_values(seed, "prehover_seed")
        joint_goal = self._prehover_ik(target, seed)
        if joint_goal is None:
            return None
        if bool(self._param("lock_wrist_roll")):
            joint_goal[-1] = current[-1]
        trajectory = self._plan_to_joints(current, joint_goal)
        if trajectory is None:
            return None
        self._publish_display(trajectory)
        return trajectory

    def _prehover_ik(self, target, seed):
        theta = self._azimuth(target[0], target[1])
        x, y, z, w = self._yaw_quat(
            [float(v) for v in self._param("prehover_quat_xyzw")], theta
        )
        dx, dy = self._rotate_xy(*[float(v) for v in self._param("prehover_xy_offset")], theta)
        pz = target[2] + float(self._param("prehover_height"))
        px, py = target[0] + dx, target[1] + dy
        pose_stamped = PoseStamped(
            header=Header(frame_id=str(self._param("object_frame_id"))),
            pose=Pose(
                position=Point(x=px, y=py, z=pz),
                orientation=Quaternion(x=x, y=y, z=z, w=w),
            ),
        )
        link = str(self._param("prehover_ik_link_name"))
        self.node.get_logger().info(
            f"prehover IK ({link}) [{px:.3f}, {py:.3f}, {pz:.3f}]"
        )
        return self.compute_ik_joint_target(
            pose_stamped,
            seed,
            link,
            float(self._param("ik_timeout")),
            True,
            "prehover IK",
        )

    def _plan_hover(self, target, size, object_class):
        if not self._add_collision_object(target, size, object_class):
            return None
        # Seed IK from seed_point (reachable) so KDL converges; plan from current for minimal motion.
        seed = [float(v) for v in self._param("seed_point")]
        current = self.current_joint_values(seed, "seed_point")
        joint_goal = self._hover_ik(target, seed)
        if joint_goal is None:
            return None
        if bool(self._param("lock_wrist_roll")):
            joint_goal[-1] = current[-1]
        trajectory = self._plan_to_joints(current, joint_goal)
        if trajectory is None:
            return None
        self._publish_display(trajectory)
        return trajectory

    def _hover_ik(self, target, seed):
        """Hover IK aiming grasp_tcp at the measured tilt, azimuth-rotated and shifted by
        grasp_xy_offset. The collision object stays at the true pose; only the IK target shifts."""
        theta = self._azimuth(target[0], target[1])
        dx, dy = self._rotate_xy(*[float(v) for v in self._param("grasp_xy_offset")], theta)
        x, y, z, w = self._yaw_quat(
            [float(v) for v in self._param("hover_quat_xyzw")], theta
        )
        hover_z = target[2] + float(self._param("hover_height"))
        pose_stamped = PoseStamped(
            header=Header(frame_id=str(self._param("object_frame_id"))),
            pose=Pose(
                position=Point(x=target[0] + dx, y=target[1] + dy, z=hover_z),
                orientation=Quaternion(x=x, y=y, z=z, w=w),
            ),
        )
        self.node.get_logger().info(
            f"hover IK at [{target[0] + dx:.3f}, {target[1] + dy:.3f}, {hover_z:.3f}] "
            f"quat [{x:.3f}, {y:.3f}, {z:.3f}, {w:.3f}]"
        )
        return self.compute_ik_joint_target(
            pose_stamped,
            seed,
            str(self._param("ik_link_name")),
            float(self._param("ik_timeout")),
            True,
            "hover IK",
        )

    def _link_translation(self, link):
        """(x, y, z) of `link` in the object frame (base), or None if TF not ready."""
        frame = str(self._param("object_frame_id"))
        try:
            tf = self._tf_buffer.lookup_transform(frame, link, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.node.get_logger().warn(f"TF {frame}->{link} failed: {exc}")
            return None
        t = tf.transform.translation
        return (t.x, t.y, t.z)

    def _plan_descent(self, target, size, object_class):
        """Straight drop of grasp_tcp by descend_delta from its current (hover) height read from
        TF, same xy. Clamped above table_z + table_margin. Direct joint move (no OMPL) so the
        finger-object contact is not rejected."""
        if not self._add_collision_object(target, size, object_class):
            return None
        link = str(self._param("descend_ik_link_name"))
        trans = self._link_translation(link)
        if trans is None:
            self.node.get_logger().warn(f"no TF for {link} yet; skipping descent plan")
            return None
        z_table = target[2] - float(self._param("object_height")) / 2.0
        floor = z_table + float(self._param("table_margin"))
        z_target = max(trans[2] - float(self._param("descend_delta")), floor)
        current = self.current_joint_values(
            [float(v) for v in self._param("seed_point")], "seed_point"
        )
        # `current` is the hover config: save it so LIFT can return there with a direct joint move.
        self._hover_joints = list(current)
        joint_goal = self._descent_ik((trans[0], trans[1]), z_target, current)
        if joint_goal is None:
            return None
        if bool(self._param("lock_wrist_roll")):
            joint_goal[-1] = current[-1]
        trajectory = self._smooth_trajectory(
            current, joint_goal, float(self._param("grasp_descend_duration"))
        )
        self._publish_display(trajectory)
        self.node.get_logger().info(
            f"descent ghost: {link} z {trans[2]:.3f} -> {z_target:.3f} "
            f"(table {z_table:.3f}, floor {floor:.3f})"
        )
        return trajectory

    def _azimuth(self, x, y) -> float:
        """Azimuth (rad) of a base-frame XY point, relative to azimuth_ref_deg (where the grasp
        quaternions were measured). The grasp frame and its XY offset are rotated about base Z by
        this angle so the tilt stays radial off-axis. Returns 0 when azimuth_align is off."""
        if not bool(self._param("azimuth_align")):
            return 0.0
        ref = math.radians(float(self._param("azimuth_ref_deg")))
        return math.atan2(y, x) - ref

    @staticmethod
    def _yaw_quat(quat, theta):
        """Rotate a base-frame quaternion (xyzw) about base Z by theta (pre-multiply)."""
        if theta == 0.0:
            return list(quat)
        delta = quaternion_about_axis(theta, (0.0, 0.0, 1.0))
        return list(quaternion_multiply(delta, list(quat)))

    @staticmethod
    def _rotate_xy(dx, dy, theta):
        """Rotate a base-frame XY offset by theta, so a radial offset stays radial off-axis."""
        if theta == 0.0:
            return dx, dy
        c, s = math.cos(theta), math.sin(theta)
        return dx * c - dy * s, dx * s + dy * c

    def _descent_quat(self, theta: float = 0.0):
        """Hover orientation tilted by descend_pitch_deg about base Y, then azimuth-rotated about
        base Z by theta (xyzw)."""
        hover_q = [float(v) for v in self._param("hover_quat_xyzw")]
        pitch = math.radians(float(self._param("descend_pitch_deg")))
        if pitch != 0.0:
            delta = quaternion_about_axis(pitch, (0.0, 1.0, 0.0))
            hover_q = list(quaternion_multiply(delta, hover_q))
        return self._yaw_quat(hover_q, theta)

    def _descent_ik(self, xy, z, seed):
        x, y, zq, w = self._descent_quat(self._azimuth(xy[0], xy[1]))
        pose_stamped = PoseStamped(
            header=Header(frame_id=str(self._param("object_frame_id"))),
            pose=Pose(
                position=Point(x=xy[0], y=xy[1], z=z),
                orientation=Quaternion(x=x, y=y, z=zq, w=w),
            ),
        )
        return self.compute_ik_joint_target(
            pose_stamped,
            seed,
            str(self._param("descend_ik_link_name")),
            float(self._param("ik_timeout")),
            False,  # tip sits at the object: do not reject the contact in IK
            "descent IK",
        )

    def _plan_lift(self):
        """Lift straight back to the saved hover config with a direct joint move (no IK that
        could fail); the path is just the reverse of the descent."""
        if self._hover_joints is None:
            self.node.get_logger().warn("no saved hover config; cannot plan lift")
            return None
        current = self.current_joint_values(self._hover_joints, "hover_joints")
        trajectory = self._smooth_trajectory(
            current, list(self._hover_joints), float(self._param("grasp_lift_duration"))
        )
        self._publish_display(trajectory)
        return trajectory

    def _place_joints_for(self, object_class: str | None):
        """Per-class drop config (joints): place_joints_<class> if declared, else place_joints."""
        if object_class:
            name = f"place_joints_{object_class.lower()}"
            if self.node.has_parameter(name):
                self.node.get_logger().info(f"place config '{name}' for class '{object_class}'")
                return [float(v) for v in self.node.get_parameter(name).value], name
        return [float(v) for v in self._param("place_joints")], "place_joints"

    def _plan_place(self, object_class: str | None = None):
        """Move (object held) to the fixed, hand-measured per-class drop config via OMPL
        (collision-aware). Returns None (caller releases at hover) if the config is unset (6
        zeros) or the wrong length. See _place_joints_for and common.yaml for measuring it."""
        place_joints, source = self._place_joints_for(object_class)
        if len(place_joints) != len(self.joint_names):
            self.node.get_logger().warn(
                f"{source} must have {len(self.joint_names)} values; skipping place"
            )
            return None
        if all(value == 0.0 for value in place_joints):
            self.node.get_logger().warn(f"{source} unset (all zeros); skipping place")
            return None
        current = self.current_joint_values(place_joints, source)
        trajectory = self._plan_to_joints(current, place_joints)
        if trajectory is None:
            return None
        self._publish_display(trajectory)
        return trajectory

    def _smooth_trajectory(self, current, goal, duration_sec, steps: int = 30):
        """Linearly interpolated multi-point trajectory so RViz shows a slow, smooth move
        (a 2-point trajectory animates as an instant jump)."""
        points = [
            [c + (g - c) * (i / steps) for c, g in zip(current, goal)]
            for i in range(steps + 1)
        ]
        return self.joint_trajectory_points(points, duration_sec)

    def _plan_to_joints(self, start_values, goal_values):
        request = GetMotionPlan.Request()
        mreq = request.motion_plan_request
        mreq.group_name = self.group_name
        mreq.pipeline_id = str(self._param("pipeline_id"))
        mreq.planner_id = str(self._param("planner_id"))
        mreq.allowed_planning_time = float(self._param("planning_time"))
        mreq.num_planning_attempts = 5
        mreq.max_velocity_scaling_factor = float(self._param("velocity_scaling"))
        mreq.max_acceleration_scaling_factor = float(self._param("acceleration_scaling"))
        mreq.start_state = self._joint_state(start_values, is_diff=False)
        mreq.goal_constraints = [self._joint_constraints(goal_values)]

        timeout = float(self._param("result_timeout"))
        future = self._planner.call_async(request)
        if not self.wait(future, timeout):
            self.node.get_logger().error(f"planner did not respond within {timeout:.1f}s")
            return None
        response = future.result()
        if response is None:
            self.node.get_logger().error("planner returned an empty response")
            return None
        plan = response.motion_plan_response
        if plan.error_code.val != MoveItErrorCodes.SUCCESS:
            self.node.get_logger().error(
                f"planning failed with code {plan.error_code.val}: {plan.error_code.message}"
            )
            return None
        self._trajectory_start = plan.trajectory_start
        self.node.get_logger().info("plan computed")
        return plan.trajectory

    def _joint_constraints(self, joint_values):
        tolerance = float(self._param("joint_tolerance"))
        return Constraints(
            joint_constraints=[
                JointConstraint(
                    joint_name=name,
                    position=value,
                    tolerance_above=tolerance,
                    tolerance_below=tolerance,
                    weight=1.0,
                )
                for name, value in zip(self.joint_names, joint_values)
            ]
        )

    def _publish_display(self, trajectory) -> None:
        display = DisplayTrajectory()
        display.trajectory.append(trajectory)
        if getattr(self, "_trajectory_start", None) is not None:
            display.trajectory_start = self._trajectory_start
        self._display_pub.publish(display)

    # ---- planning scene: add / attach / detach / allow collisions -----------------------

    def _make_primitive(self, object_class: str, size: list[float]) -> SolidPrimitive:
        sx, sy, sz = size
        cls = object_class.lower()
        if cls == "cork":
            return SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[sz, max(sx, sy) / 2.0])
        if cls == "pompom":
            return SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[(sx + sy + sz) / 6.0])
        return SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[sx, sy, sz])

    def _add_collision_object(self, target, size, object_class) -> bool:
        object_id = str(self._param("object_id"))
        frame_id = str(self._param("object_frame_id"))
        pose = Pose(
            position=Point(x=target[0], y=target[1], z=target[2]),
            orientation=Quaternion(w=1.0),
        )
        self.node.get_logger().info(
            f"object '{object_id}' class={object_class} at "
            f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] {frame_id}"
        )
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(
            CollisionObject(
                id=object_id,
                header=Header(frame_id=frame_id),
                primitives=[self._make_primitive(object_class, size)],
                primitive_poses=[pose],
                operation=CollisionObject.ADD,
            )
        )
        return self._apply_scene(scene)

    def _attach_object(self, target, size, object_class) -> bool:
        object_id = str(self._param("object_id"))
        frame_id = str(self._param("object_frame_id"))
        link = str(self._param("attached_link_name"))
        pose = Pose(
            position=Point(x=target[0], y=target[1], z=target[2]),
            orientation=Quaternion(w=1.0),
        )
        attached = AttachedCollisionObject(
            link_name=link,
            touch_links=[str(name) for name in self._param("touch_links")],
            object=CollisionObject(
                id=object_id,
                header=Header(frame_id=frame_id),
                primitives=[self._make_primitive(object_class, size)],
                primitive_poses=[pose],
                operation=CollisionObject.ADD,
            ),
        )
        scene = PlanningScene(is_diff=True, robot_state=RobotState(is_diff=True))
        scene.robot_state.attached_collision_objects.append(attached)
        self.node.get_logger().info(f"attached '{object_id}' to {link}")
        return self._apply_scene(scene)

    def _detach_object(self) -> bool:
        """Detach the held object from the gripper and remove it from the world."""
        object_id = str(self._param("object_id"))
        link = str(self._param("attached_link_name"))
        detach = AttachedCollisionObject(
            link_name=link,
            object=CollisionObject(id=object_id, operation=CollisionObject.REMOVE),
        )
        scene = PlanningScene(is_diff=True, robot_state=RobotState(is_diff=True))
        scene.robot_state.attached_collision_objects.append(detach)
        scene.world.collision_objects.append(
            CollisionObject(id=object_id, operation=CollisionObject.REMOVE)
        )
        self.node.get_logger().info(f"detached '{object_id}' from {link}")
        return self._apply_scene(scene)

    def _reset_scene(self) -> None:
        """Detach the object so the next cycle plans against a clean scene.

        Do NOT touch the AllowedCollisionMatrix: a non-empty ACM in a diff REPLACES the whole ACM,
        wiping the SRDF defaults and making every collision-aware IK time out.
        """
        self._detach_object()
        self._hover_joints = None

    def _apply_scene(self, scene) -> bool:
        future = self._planning_scene.call_async(ApplyPlanningScene.Request(scene=scene))
        if not self.wait(future, float(self._param("result_timeout"))):
            self.node.get_logger().error("timeout applying planning scene")
            return False
        response = future.result()
        if response is None or not response.success:
            self.node.get_logger().error("failed applying planning scene")
            return False
        return True

    # ---- gripper ------------------------------------------------------------------------

    def _wait_for_gripper_server(self) -> bool:
        if self._gripper_trajectory.wait_for_server(timeout_sec=1.0):
            self._gripper, self._gripper_kind = self._gripper_trajectory, "trajectory"
            return True
        if self._gripper_command.wait_for_server(timeout_sec=10.0):
            self._gripper, self._gripper_kind = self._gripper_command, "command"
            return True
        return False

    def _command_gripper(
        self, mode: str, object_width: float | None = None, object_class: str | None = None
    ) -> bool:
        if not bool(self._param("use_gripper")):
            self.node.get_logger().info(f"gripper disabled, skip {mode}")
            return True
        # Hardware grasp uses a calibrated, reachable per-class close angle (the motor stops there
        # instead of crushing), not a width->angle guess. See _hardware_grasp_position_for.
        if mode == "grasp" and self._gripper_kind == "command":
            return self._command_gripper_action(
                mode, 0.0, raw_position=self._hardware_grasp_position_for(object_class)
            )
        if mode == "open":
            position = self._open_gripper_position()
        elif mode == "grasp":
            position = self._grasp_gripper_position(object_width)
        else:
            position = self._closed_gripper_position()
        if self._gripper_kind == "command":
            return self._command_gripper_action(mode, position)
        return self._command_gripper_trajectory(mode, position)

    def _command_gripper_trajectory(self, mode: str, position: float) -> bool:
        joint_names = [str(name) for name in self._param("gripper_joint_names")]
        goal = FollowJointTrajectory.Goal(
            trajectory=JointTrajectory(
                joint_names=joint_names,
                points=[
                    JointTrajectoryPoint(
                        positions=[position] * len(joint_names),
                        time_from_start=self.duration(
                            float(self._param("gripper_motion_duration"))
                        ),
                    )
                ],
            )
        )
        self.node.get_logger().info(f"{mode} gripper to {position:.4f} on {joint_names}")
        return self._send_gripper_goal(goal)

    def _command_gripper_action(self, mode: str, position: float, raw_position=None) -> bool:
        goal = GripperCommand.Goal()
        goal.command.position = (
            float(raw_position) if raw_position is not None
            else self._hardware_gripper_position(position)
        )
        goal.command.max_effort = float(self._param("gripper_max_effort"))
        self.node.get_logger().info(f"{mode} gripper to {goal.command.position:.4f} rad")
        return self._send_gripper_goal(goal)

    def _send_gripper_goal(self, goal) -> bool:
        send_future = self._gripper.send_goal_async(goal)
        if not self.wait(send_future, 5.0):
            self.node.get_logger().error("timed out sending gripper goal")
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.node.get_logger().error("gripper goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        if not self.wait(result_future, float(self._param("result_timeout"))):
            self.node.get_logger().error("gripper did not return a result in time")
            return False
        return result_future.result() is not None

    def _hardware_grasp_position_for(self, object_class: str | None) -> float:
        """Per-class hardware close angle (rad): hardware_grasp_position_<class> if declared,
        else hardware_grasp_position."""
        default = float(self._param("hardware_grasp_position"))
        if not object_class:
            return default
        name = f"hardware_grasp_position_{object_class.lower()}"
        if self.node.has_parameter(name):
            value = float(self.node.get_parameter(name).value)
            self.node.get_logger().info(
                f"grasp close {value:.3f} rad for class '{object_class}'"
            )
            return value
        self.node.get_logger().info(
            f"no per-class close for '{object_class}'; using default {default:.3f} rad"
        )
        return default

    def _open_gripper_position(self) -> float:
        return self._clamp_gripper_width(float(self._param("open_gripper_position")))

    def _closed_gripper_position(self) -> float:
        return self._clamp_gripper_width(float(self._param("closed_gripper_position")))

    def _grasp_gripper_position(self, object_width: float | None) -> float:
        if object_width is None or not bool(self._param("grasp_gripper_to_object_width")):
            return self._closed_gripper_position()
        padding = float(self._param("gripper_grasp_padding"))
        target = self._clamp_gripper_width((object_width + padding) * 0.5)
        self.node.get_logger().info(
            f"grasp width {target:.4f} for object width {object_width:.4f}"
        )
        return target

    def _clamp_gripper_width(self, position: float) -> float:
        low = float(self._param("closed_gripper_position"))
        high = float(self._param("open_gripper_position"))
        return max(low, min(high, position))

    def _hardware_gripper_position(self, sim_position: float) -> float:
        max_width = float(self._param("max_gripper_width"))
        ratio = 0.0 if max_width <= 0.0 else (2.0 * sim_position / max_width)
        ratio = max(0.0, min(1.0, ratio))
        open_position = float(self._param("hardware_open_gripper_position"))
        closed_position = float(self._param("hardware_closed_gripper_position"))
        return closed_position + (open_position - closed_position) * ratio


def main() -> None:
    rclpy.init()
    demo = PickPompomsCork()
    try:
        ok = demo.run()
    except Exception as exc:  # noqa: BLE001
        demo.node.get_logger().error(str(exc))
        ok = False
    finally:
        demo.node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
