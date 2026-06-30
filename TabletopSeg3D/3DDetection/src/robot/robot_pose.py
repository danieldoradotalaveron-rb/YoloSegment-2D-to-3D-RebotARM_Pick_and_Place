"""Read the live robot end-effector pose (base_T_ee) from ROS2 TF.

Generic robot plumbing shared by the calibration capture and the Fase A
validation: spins an rclpy node in a background thread with a TF listener and
hands back base_T_ee as a 4x4 matrix on demand.

Self-contained: uses scipy for the quaternion -> rotation conversion (already in
the venv), so it does not depend on the calibration package.
"""

from __future__ import annotations

import threading

import numpy as np
from scipy.spatial.transform import Rotation


class RobotPoseReader:
    """Background rclpy node + TF listener; exposes base_T_ee on demand."""

    def __init__(
        self,
        base_frame: str = "base_link",
        ee_frame: str = "end_link",
        node_name: str = "robot_pose_reader",
    ) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from tf2_ros import Buffer, TransformListener

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self._node)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        self.base_frame = base_frame
        self.ee_frame = ee_frame

    def lookup(self, timeout_s: float = 0.5) -> np.ndarray | None:
        """Return base_T_ee (4x4) at the latest available time, or None on failure."""
        from rclpy.duration import Duration
        from rclpy.time import Time

        try:
            tf = self._buffer.lookup_transform(
                self.base_frame, self.ee_frame, Time(), timeout=Duration(seconds=timeout_s)
            )
        except Exception as exc:  # noqa: BLE001 - tf2 raises several unrelated types
            print(f"[tf] lookup {self.base_frame} -> {self.ee_frame} failed: {exc}")
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def shutdown(self) -> None:
        try:
            self._executor.shutdown()
            self._node.destroy_node()
            self._rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
