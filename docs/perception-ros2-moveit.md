# Perception, ROS 2 and MoveIt

This document describes how the 3D detector is bridged to ROS 2, how the detections
are expressed in the robot base frame via a hand-eye calibration, and what was added
or modified on the MoveIt/RViz side to plan and execute tabletop pick-and-place.

The ROS 2 workspace under `REBOT/reBotArmController_ROS2/` is based on Seeed
Studio's controller:

    https://github.com/EclipseaHime017/reBotArmController_ROS2

It builds on that work with local modifications and may not be in sync with the
upstream `main`. The five ROS packages keep their upstream Apache-2.0 license; the
`pick_pompoms_cork` node and its configuration in `rebotarm_moveit_demos` are the
parts authored in this repository.

## Components

| Package / file | Role |
| --- | --- |
| `TabletopSeg3D/3DDetection/scripts/perception_node.py` | Perception ROS 2 publisher (runs in the repo venv) |
| `TabletopSeg3D/3DDetection/src/robot/detection_publisher.py` | Builds and publishes `Detection3DArray` |
| `TabletopSeg3D/3DDetection/src/robot/extrinsics.py` / `robot_pose.py` | Hand-eye transform and TF reader |
| `handeye/calibration_d405/` | ChArUco hand-eye calibration (generate / capture / solve / validate) |
| `rebotarm_moveit_config` | MoveIt config: URDF/xacro, SRDF, RViz, launch |
| `rebotarm_moveit_demos` | `pick_pompoms_cork` pick node + config |
| `rebotarm_msgs` | Custom messages and services |
| `rebotarm_bringup` / `rebotarmcontroller` | Robot driver and bring-up (upstream) |

## Perception node

`perception_node.py` is a plain `rclpy` participant, **not** a colcon package. It
runs in the repository virtual environment so it has access to
open3d/torch/ultralytics/pyrealsense2 and `rclpy` at the same time, and joins the
same DDS domain as the robot driver. There is no build step.

```bash
# Terminal A: robot driver / bring-up (provides TF base_link -> end_link)
ros2 launch rebotarm_bringup bringup.launch.py

# Terminal B: perception publisher (repo venv)
.venv/bin/python TabletopSeg3D/3DDetection/scripts/perception_node.py
```

### What it consumes

- RealSense color + aligned depth frames, read directly through `pyrealsense2`
  (the node owns the camera; it does not subscribe to an image topic).
- TF `base_link -> end_link` (configurable via `--base-frame` / `--ee-frame`),
  read in a background thread.
- The hand-eye extrinsic `ee_T_cam` from a calibration JSON. By default it uses the
  most recent `handeye/calibration_d405/captures/session_*/ee_T_cam.json`; override
  with `--ee-t-cam`.
- A trained YOLO-seg model (latest `runs/segment/*/weights/best.pt` by default).

### What it publishes

A `vision_msgs/Detection3DArray` on `/perception/detections` (override with
`--topic`), with `header.frame_id = base_link`. The array is published every frame;
when there are stable tracks but TF is unavailable, an empty array is published.

Only tracks that are both **stable** (tracker gate) and **in workspace** are
included. For each one, a `vision_msgs/Detection3D` is filled as:

- `id`: the tracker `track_id` as a string.
- `bbox.center.position`: object center in `base_link` (meters).
- `bbox.center.orientation`: identity. Grasp orientation is decided by the pick
  node (top-down approach), not by perception.
- `bbox.size`: object extent `(x, y, z)` in meters, so the pick node can size the
  gripper.
- `results[0].hypothesis.class_id`: class name (e.g. `cork`, `pompom`).
- `results[0].hypothesis.score`: mean detection confidence over the track.

### How a detection is produced

1. YOLO-seg inference on the color frame produces instance masks (Ultralytics).
2. Each mask is projected to camera-frame 3D points using the aligned depth and
   intrinsics, then depth-band filtered.
3. A tabletop-aligned OBB is fit using the per-frame table-plane normal
   (RANSAC plane fit on the scene cloud, EMA-smoothed across frames for an
   eye-in-hand camera). This yields center, extent and yaw.
4. A class policy decides yaw reliability: only classes listed as non-symmetric get
   a trusted yaw; symmetric classes (`cork`, `pompom`, `lighter`) are forced to a
   table-axis-aligned box (yaw = 0) and grasped top-down.
5. A temporal tracker links detections across frames, filters the pose, and raises
   `stable` once the object is well localized. The stability gate is configured in
   `config/sense.yaml` (`track_window`, `track_min_hits`, `track_max_pos_std`,
   `track_max_yaw_std`, `track_min_conf`, `track_assoc_dist`, and the hysteresis
   parameters).
6. The filtered camera-frame center is transformed to the base frame and published.

### Camera-to-base transform (eye-in-hand)

The camera is mounted on the end-effector. A camera-frame point is mapped to the
base frame as:

    base_point = base_T_ee @ ee_T_cam @ [x, y, z, 1]

where `base_T_ee` is read live from TF (`base_link -> end_link`) and `ee_T_cam` is
the hand-eye calibration result. This is implemented in
`robot/extrinsics.py::cam_point_to_base`.

## Hand-eye calibration (ChArUco)

The calibration lives in `handeye/calibration_d405/` and follows the standard
eye-in-hand OpenCV procedure. Captured sessions and validation output are
git-ignored (`handeye/**/captures/`, `handeye/validation_base_frame.json`); the
scripts are tracked.

1. Generate a ChArUco board.

   ```bash
   uv run python handeye/calibration_d405/generate_charuco.py
   ```

   Defaults: 8x6 squares, 25 mm nominal square, `5x5_100` dictionary. The script
   writes a PNG plus a JSON spec. Always measure the displayed/printed square edge
   with a ruler and set the real `square_length_m` in the board JSON before
   calibrating; the square size sets the translation scale.

2. Capture synchronized samples while moving the arm by hand (gravity compensation).
   Each sample stores the raw color image and `base_T_ee` from TF; ChArUco detection
   and the solve happen offline.

   ```bash
   # Driver up, then enable gravity compensation, then:
   .venv/bin/python handeye/calibration_d405/capture_handeye.py
   ```

   Controls in the preview window: `c`/`SPACE` capture, `u` undo, `q`/`ESC` finish.
   Aim for 12-20 varied poses. Sessions are written incrementally to
   `handeye/calibration_d405/captures/session_*/`.

3. Solve for `ee_T_cam`.

   ```bash
   .venv/bin/python handeye/calibration_d405/calibrate_handeye.py \
       handeye/calibration_d405/captures/session_XXXX
   ```

   This detects the board in each saved image (`solvePnP` for `cam_T_target`),
   pairs it with the recorded `base_T_ee`, and runs `cv2.calibrateHandEye` across
   several methods (TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS). Because the board is
   physically fixed, `base_T_target = base_T_ee @ ee_T_cam @ cam_T_target` should be
   near-constant across poses; the script reports the translation and rotation
   spread of that estimate and picks the lowest-residual method. The result is
   written to `ee_T_cam.json` next to the session, with `ee_T_cam`, the translation,
   the `xyzw` quaternion, and the validation spread.

The solver already reports a validation residual (the spread of the fixed-board
estimate across poses); use it as the primary quality check. A small translation
and rotation spread indicates a good calibration.

The hand-eye reference frame is `end_link`, defined identically in the calibration
URDF and in the MoveIt URDF (see below), so a calibration solved against `end_link`
is reusable by the perception node and the pick node without re-solving.

## MoveIt and RViz

`rebotarm_moveit_config` holds the MoveIt configuration. The robot description adds
three frames in `config/rebotarm.urdf.xacro` on top of the upstream arm + gripper:

- `gripper_tcp`: `gripper_link` offset by `(-0.0443, 0, 0)` m. This is the SRDF arm
  chain tip.
- `grasp_tcp`: at `gripper_link` origin (the closed fingertips, on the approach
  axis). Used by the pick node to center the grasp on the object tips.
- `end_link`: hand-eye reference frame, fixed to `link6`, defined to match the
  calibration URDF so `ee_T_cam` is usable here without recalibrating.

The SRDF (`config/rebotarm.srdf`) defines:

- group `arm`: chain `base_link` -> `gripper_tcp`.
- group `gripper`: `gripper_joint1`, `gripper_joint2`, with `open` and `closed`
  group states.
- end effector `gripper` on `gripper_link`.

### Launch files

- `demo.launch.py`: MoveIt with mock controllers (simulation, no hardware). Brings
  up `move_group` and RViz with the motion planning plugin. RViz config defaults to
  `moveit.rviz`.
- `hardware.launch.py`: MoveIt against an already-running driver. Brings up
  `move_group`, `robot_state_publisher`, a static `world -> base_link` transform and
  RViz, with `/joint_states` remapped into the driver namespace (`rebotarm` by
  default, via the `arm_namespace` argument).

RViz configurations: `rebotarm_moveit_config/launch/moveit.rviz`,
`demo_view.rviz`, and `rebotarm_moveit_demos/rviz/grasp_tcp_check.rviz` (a minimal
config to visually check the `grasp_tcp` frame on the object).

## Pick node (`pick_pompoms_cork`)

`rebotarm_moveit_demos` provides the pick-and-place node that consumes perception
and drives MoveIt. It subscribes to `/perception/detections`
(`vision_msgs/Detection3DArray`) and runs a staged state machine
(prehover -> hover -> descend -> lift -> place -> done) per object. Parameters are
in `config/common.yaml` and `config/pick_pompoms_cork.yaml`; launch with:

```bash
ros2 launch rebotarm_moveit_demos pick_pompoms_cork.launch.py
```

Summary of how it uses MoveIt and the driver:

- Plans joint targets with the MoveIt IK and planning services and executes
  trajectories through `move_group`.
- Manages the planning scene (object as a collision object, attach/detach on grasp)
  via the `ApplyPlanningScene` service.
- Commands the gripper with a class-dependent close value (different for `cork` and
  `pompom`) and places each class at a fixed, pre-measured drop pose.
- Aligns the gripper approach radially to the object (azimuth compensation about the
  base Z axis) so the side objects are approached like centered ones.

For the design and staging details, see the node source in
`REBOT/reBotArmController_ROS2/src/rebotarm_moveit_demos/`.

## Custom interfaces (`rebotarm_msgs`)

Messages: `ArmStatus`, `JointMitCmd`, `JointMotorState`, `JointPosVelCmd`,
`JointVelCmd`. Services: `GripperCommand`, `MoveToPoseIK`, `SetGripper`, `SetMode`,
`SetZero`. These are part of the upstream controller and are used by the driver and
demos.

## Known gaps / TODO

- The perception node owns the camera directly (pyrealsense2); it does not consume a
  ROS image topic, so it cannot currently run from a recorded bag without changes.
- `perception_node.py` argparse defaults (camera serial, confidence, device) are not
  the same as the values in `config/sense.yaml` used by `just tabletop`; pass flags
  explicitly for a deployment.
- The hand-eye `end_link` correspondence between the calibration URDF and the MoveIt
  URDF is asserted in the xacro comments; re-verify it if either description changes.
- Workspace bounds are expressed in the camera frame in `config/sense.yaml`; confirm
  they match the physical table before trusting the in-workspace gate.
- No automated tests cover the ROS bridge or the pick state machine.
