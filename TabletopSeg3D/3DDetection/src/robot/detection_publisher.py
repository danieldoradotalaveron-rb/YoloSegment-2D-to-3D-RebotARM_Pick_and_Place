"""Publish STABLE 3D detections in the robot base frame on a single ROS2 topic.

Single source of truth for the perception->ROS bridge, shared by:
  - perception_node.py        (headless publisher)
  - realtime_open3d_scene.py  (viewer + publisher, with --real)

A process that already owns the camera builds tracked detections and calls
``publish(tracked)``. For every stable, in-workspace track this transforms the
filtered camera-frame center to base_link (base_T_ee from TF x ee_T_cam from
hand-eye) and emits a vision_msgs/Detection3DArray. The TF reader runs in its own
background thread, so the calling loop never has to spin.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from robot.extrinsics import cam_point_to_base
from robot.robot_pose import RobotPoseReader


def build_detection3d(det, base_xyz: np.ndarray, frame_id: str, stamp) -> Detection3D:
    """One stable detection -> vision_msgs/Detection3D in base frame.

    Position is the validated base-frame center; orientation is identity for now
    (grasp orientation is top-down and decided by the pick node). size is the
    object extent so the pick node can size the gripper; class/score/id ride along.
    """
    msg = Detection3D()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.id = str(getattr(det, "track_id", -1))

    bbox = BoundingBox3D()
    bbox.center.position.x = float(base_xyz[0])
    bbox.center.position.y = float(base_xyz[1])
    bbox.center.position.z = float(base_xyz[2])
    bbox.center.orientation.w = 1.0
    ext = getattr(det, "extent_filtered_xyz", None) or [0.0, 0.0, 0.0]
    bbox.size.x, bbox.size.y, bbox.size.z = (float(ext[0]), float(ext[1]), float(ext[2]))
    msg.bbox = bbox

    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = str(det.class_name)
    hyp.hypothesis.score = float(getattr(det, "confidence_mean", 0.0))
    msg.results.append(hyp)
    return msg


class DetectionPublisher:
    """Owns the ROS side: node, publisher, TF pose reader and hand-eye extrinsics."""

    def __init__(
        self,
        ee_t_cam_path: str,
        *,
        topic: str = "/perception/detections",
        frame_id: str = "base_link",
        base_frame: str = "base_link",
        ee_frame: str = "end_link",
        tf_timeout: float = 0.5,
        node_name: str = "perception_publisher",
    ) -> None:
        path = Path(ee_t_cam_path)
        if not ee_t_cam_path or not path.exists():
            raise FileNotFoundError(
                f"ee_T_cam.json not found: {ee_t_cam_path!r}. Pass --ee-t-cam."
            )
        self.ee_T_cam = np.array(json.loads(path.read_text())["ee_T_cam"], dtype=np.float64)
        self.topic = topic
        self.frame_id = frame_id
        self.tf_timeout = float(tf_timeout)

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        self.publisher = self.node.create_publisher(Detection3DArray, topic, 10)
        self.pose_reader = RobotPoseReader(
            base_frame, ee_frame, node_name=f"{node_name}_pose"
        )
        self._last_base_T_cam: np.ndarray | None = None

    def base_T_cam(self, timeout: float | None = None) -> np.ndarray | None:
        """Latest base_link <- camera transform (4x4): base_T_ee (TF) @ ee_T_cam.

        Returns the last good transform if TF is momentarily unavailable, or None
        until the first successful lookup. Lets a viewer render the cloud in the
        static base frame for an eye-in-hand camera.
        """
        base_T_ee = self.pose_reader.lookup(
            self.tf_timeout if timeout is None else timeout
        )
        if base_T_ee is not None:
            self._last_base_T_cam = base_T_ee @ self.ee_T_cam
        return self._last_base_T_cam

    def publish(self, tracked) -> tuple[int, int, bool, list[str]]:
        """Publish stable in-workspace detections.

        Returns (n_stable, n_published, tf_ok, names). ``names`` are the stable
        tracks as ``"class#id"``. When there are stable tracks but TF is
        unavailable, an empty array is still published and tf_ok is False.
        """
        stable = [
            d
            for d in tracked
            if getattr(d, "stable", False) and getattr(d, "in_workspace", True)
        ]
        names = [f"{d.class_name}#{getattr(d, 'track_id', -1)}" for d in stable]
        base_T_ee = self.pose_reader.lookup(self.tf_timeout) if stable else None

        arr = Detection3DArray()
        arr.header.frame_id = self.frame_id
        arr.header.stamp = self.node.get_clock().now().to_msg()
        if base_T_ee is not None:
            for d in stable:
                base_xyz = cam_point_to_base(
                    np.asarray(d.center_filtered_xyz, dtype=np.float64),
                    base_T_ee,
                    self.ee_T_cam,
                )
                arr.detections.append(
                    build_detection3d(d, base_xyz, self.frame_id, arr.header.stamp)
                )
        self.publisher.publish(arr)
        return len(stable), len(arr.detections), base_T_ee is not None, names

    def shutdown(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.pose_reader.shutdown()
        except Exception:  # noqa: BLE001
            pass
