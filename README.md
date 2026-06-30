# YoloSegment-2D-to-3D-RebotARM

<p align="center">
  <video src="docs/img/video2.mp4" alt="Eye-in-hand pick-and-place on the reBot arm" height="300" autoplay loop muted playsinline></video>
  &nbsp;
  <video src="docs/img/video.mp4" alt="Realtime RGB-D 3D detection alongside 2D segmentation" height="300" autoplay loop muted playsinline></video>
</p>

An RGB-D tabletop perception and grasping pipeline. It captures RGB-D with an Intel
RealSense camera, segments objects in 2D with a YOLO instance-segmentation model,
lifts the masks to 3D (point cloud + tabletop-aligned oriented bounding box),
estimates each object's pose in the robot base frame, and feeds those detections to
a ROS 2 / MoveIt pick-and-place node running on a reBot arm.

Status: work in progress. The example dataset uses the classes `cork`, `lighter`
and `pompom`, and the pick node is set up for `cork` and `pompom`.

## Pipeline overview

1. Capture, label and train a YOLO-seg model on your own object classes (2D).
2. Lift detections to 3D using aligned depth and camera intrinsics; fit a
   table-aligned box and track objects over time for a stable pose.
3. Transform stable detections into the robot base frame with a hand-eye
   calibration and publish them as `vision_msgs/Detection3DArray`.
4. Plan and execute the grasp with MoveIt (prehover, hover, descend, lift, place).

<p align="center">
  <img src="docs/img/train_pipe.png" alt="Synthetic-augmented RGB-D to 3D object localization pipeline" width="820">
</p>

## Repository structure

```text
DS/                          # 2D dataset pipeline (capture, label, convert, analyze)
TabletopSeg3D/3DDetection/   # realtime 3D detection + ROS 2 perception node
handeye/calibration_d405/    # ChArUco hand-eye calibration (generate/capture/solve)
REBOT/reBotArmController_ROS2/# ROS 2 workspace (driver, MoveIt config, pick demo, msgs)
config/                      # runtime defaults (sense.yaml, class_colors.yaml)
scripts/sense.sh             # command orchestrator wrapped by the justfile
justfile                     # entry point for all dataset/training/runtime recipes
runs/segment/                # training outputs (git-ignored)
docs/                        # documentation (this folder)
```

## Quick start

```bash
uv sync     # create the Python environment from pyproject.toml + uv.lock
just        # list all recipes
just help   # full CLI help
```

See the documentation below for the full workflows.

## Pretrained model and dataset

The dataset and the trained YOLO-seg weights are hosted on Hugging Face; they are
not shipped in this repository.

- Model (weights `best.pt`): https://huggingface.co/ddt1992/pompomcork_YoloSegment-2D-to-3D-RebotARM
- Dataset (raw + labeled images, YOLO-seg export): https://huggingface.co/datasets/ddt1992/pompomcork_YoloSegment-2D-to-3D-RebotARM

```python
from ultralytics import YOLO
model = YOLO("best.pt")  # download from the model repo above
```

## Documentation

- [Training and segmentation](docs/training-and-segmentation.md) — dataset capture,
  labeling, YOLO-seg training/validation, pre-labeling, synthetic data, and the
  realtime 3D detection runtime, all via `just`.
- [Perception, ROS 2 and MoveIt](docs/perception-ros2-moveit.md) — the perception
  node and its `Detection3DArray` interface, the eye-in-hand ChArUco calibration,
  and the MoveIt/RViz integration and pick node.

## Changes over upstream

This repository builds on two upstream projects. The work below is what was added or
modified here on top of them.

### Over `TabletopSeg3D` (`DS/`, `TabletopSeg3D/`)

- Symmetric vs non-symmetric class policy: a trusted yaw is only computed for
  non-symmetric classes; symmetric ones (`cork`, `pompom`, `lighter`) are forced to a
  table-axis-aligned box and grasped top-down.
- Tabletop-aligned oriented bounding box from a per-frame RANSAC table-plane normal,
  EMA-smoothed across frames for the eye-in-hand camera.
- Temporal tracker with a stability gate (window, min hits, position/yaw std,
  confidence, hysteresis) configured in `config/sense.yaml`.
- ROS 2 bridge: `DetectionPublisher` and `perception_node.py`, publishing stable
  in-workspace detections as `vision_msgs/Detection3DArray` in the robot base frame,
  plus the eye-in-hand transform consumed from the hand-eye calibration.
- Eye-in-hand ChArUco calibration under `handeye/calibration_d405/`
  (generate / capture / solve).
- `justfile` + `scripts/sense.sh` orchestration and the RGB-D synthetic augmentation
  pipeline (lift, render, composite) feeding the YOLO-seg dataset.

### Over `reBotArmController_ROS2` (`REBOT/reBotArmController_ROS2/`)

- New `pick_pompoms_cork` node in `rebotarm_moveit_demos`: a staged pick-and-place
  state machine (prehover, hover, descend, lift, place) with multi-object auto mode,
  azimuth-aligned approach, class-dependent gripper close values and fixed per-class
  place poses, consuming `/perception/detections`.
- MoveIt description additions in `rebotarm.urdf.xacro`: `grasp_tcp` (fingertips) and
  `gripper_tcp` frames, and an `end_link` hand-eye frame aligned to the calibration
  URDF so `ee_T_cam` is reusable without recalibrating; SRDF `arm` chain to
  `gripper_tcp`.
- `hardware.launch.py` bringing MoveIt up against the running driver (with
  `/joint_states` remapped into the driver namespace) and a `grasp_tcp_check.rviz`
  helper.

Full detail is in the two documents linked above.

## Attribution and license

- This repository's own code is released under the MIT License (see `LICENSE`).
- The 2D/3D detection tooling under `DS/` and `TabletopSeg3D/` is a modified and
  adapted version of [`TabletopSeg3D`](https://github.com/Miscanthus40076/TabletopSeg3D).
- The ROS 2 workspace under `REBOT/reBotArmController_ROS2/` is based on Seeed
  Studio's [`reBotArmController_ROS2`](https://github.com/EclipseaHime017/reBotArmController_ROS2)
  and keeps its upstream Apache-2.0 license. It contains local modifications and may
  not be in sync with upstream `main`.
