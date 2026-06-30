# Training and segmentation

This document describes the 2D perception side of the repository: how to capture
data, label it, train a YOLO instance-segmentation model, validate it, and run the
realtime 3D detection that feeds the robot pipeline. All steps are driven through
[`just`](https://github.com/casey/just) recipes, which wrap `scripts/sense.sh`
(defaults in `config/sense.yaml`).

## Origin

The 3D detection code under `TabletopSeg3D/` and the dataset/segmentation tooling
under `DS/` are derived from:

    git@github.com:Miscanthus40076/TabletopSeg3D.git

The code has been adapted for the tabletop pick-and-place task in this repository.
The main areas changed relative to the original are:

- Tabletop objects (`cork`, `lighter`, `pompom` in the example dataset) and a
  symmetric/non-symmetric class policy that only computes a trusted yaw for
  non-symmetric classes.
- YOLO instance-segmentation integration (Ultralytics) as the detector.
- RGB-D projection/deprojection: lifting 2D masks to camera-frame 3D points and
  estimating a tabletop-aligned oriented bounding box (OBB).
- Point-cloud handling: per-frame scene cloud, table-plane normal estimation, and
  a temporal tracker for pose smoothing and stability gating.
- `justfile` / `scripts/sense.sh` automation for the full dataset and training
  workflow.
- A ROS 2 bridge that publishes stable detections in the robot base frame to feed
  the picking pipeline (see [perception-ros2-moveit.md](perception-ros2-moveit.md)).

## Requirements

- Linux with an Intel RealSense camera (developed with a D405) for capture and the
  realtime runtime. Training and offline tooling do not need the camera.
- [`uv`](https://github.com/astral-sh/uv) for the Python environment.
- A CUDA-capable GPU is recommended for training; inference can run on CPU.
- The optional `gsplat` synthetic backend additionally needs the CUDA toolkit
  (`nvcc`), not just the driver.

## Setup

```bash
uv sync          # or: just sync   -> create the env from pyproject.toml + uv.lock
just             # list all recipes
just help        # full CLI help and options
```

The same environment (torch / ultralytics / open3d / pyrealsense2) is used for
training, the realtime runtime, and the offline pre-labeling tools.

## Classes

Classes are defined in `DS/yolo_classes.yaml`, which pins the class ids used by the
YOLO dataset. The example dataset uses `cork`, `lighter` and `pompom`. SAM3
text prompts (optional backend) are in `DS/prelabel/sam3_classes.yaml`. Viewer
colors are in `config/class_colors.yaml`.

## Core workflow: dataset and training

```bash
just capture              # capture RGB frames    -> DS/dataset_capture/rgb/
                          # (RGBD: just capture --rgbd -> DS/dataset_capture/rgbd/)
just labelme              # draw polygon masks in LabelMe on
                          # DS/dataset_labeled/manual_labelme/
just convert --data real  # LabelMe -> YOLO-seg dataset in DS/dataset_yolo/
just analyze              # (optional) class / instance counts
just train                # train YOLO-seg -> runs/segment/train/weights/best.pt
just val                  # (optional) validate the latest model
just predict              # (optional) run inference on the val images
```

Notes:

- `convert --data` is **required** and regenerates `DS/dataset_yolo/` from scratch.
  Accepted values: `real`, `point`, `3dgs`, `all` (`all` = real + both synthetic
  pools). The validation split stays 100% real regardless of `--data`.
- Curate the captures you want to keep into `DS/dataset_raw/` (your raw archive)
  and copy the ones to label into `DS/dataset_labeled/manual_labelme/`.
- Model selection: `val` / `predict` / `tabletop` use the most recent
  `runs/segment/*/weights/best.pt` unless overridden with `--run NAME`.

## Pre-labeling (faster annotation)

Use a model to pre-label new images, then only correct them by hand. The default
backend is the previously trained YOLO model; SAM3 (text prompts) is optional and
needs `sam3.pt` plus `uv sync --extra sam3`. See `DS/prelabel/README.md`.

```bash
just capture          # grab new frames -> DS/dataset_capture/rgb/
just stage            # copy new captures -> DS/dataset_prelabel/input_images/

# YOLO backend (default)
just prelabel-yolo    # predict masks  -> DS/dataset_prelabel/prelabels_yolo/
just export-yolo      # -> LabelMe pairs in DS/dataset_prelabel/to_review_yolo/
just review-yolo      # open LabelMe and fix masks/labels
just promote-yolo     # -> DS/dataset_labeled/reviewed_yolo_labelme/

# then rebuild + retrain
just convert --data real
just train
```

The SAM3 backend mirrors these steps (`prelabel-sam3`, `export-sam3`,
`review-sam3`, `promote-sam3`).

## Synthetic data from RGB-D (optional)

Turn a few RGB-D captures with reviewed masks into extra labeled 2D views by
lifting each labeled object to 3D and re-rendering it from nearby virtual cameras.
The instance mask is propagated geometrically. This is offline dataset tooling, not
the robot runtime; validate any gain against a 100% real validation split.

Two parallel backends exist: `points` (CPU) and `gsplat` (needs the CUDA toolkit).
`--backend` is required on every synthetic/composite/review/promote command.

```bash
just capture --rgbd                       # object frames + paired empty backgrounds
just prelabel-rgbd                        # YOLO-label rgbd/ captures to review
just review-rgbd                          # fix masks in LabelMe
just promote-rgbd                         # write reviewed labels back as rgb.json

just synth-render --backend points        # init + render object views
just composite-synth --backend points     # objects over a real background
just export-composite --backend points    # -> LabelMe pairs to review
just review-composite --backend points
just promote-composite --backend points   # -> DS/dataset_labeled_point/composited_labelme/

just convert --data point                 # real + point synth
just train
```

Swap `--backend points` for `--backend gsplat` (folders `*_3dgs`, pool
`dataset_labeled_3dgs/`) and use `just convert --data 3dgs` or `--data all` to
compare. See `DS/prelabel/README.md` for the full details.

## Realtime 3D detection

Uses the most recent `runs/segment/*/weights/best.pt` (override with `--run`).

```bash
just tabletop           # RealSense + YOLO + Open3D viewer with 3D boxes
just tabletop-headless  # no GUI; prints per-frame JSON (class, center, size, yaw)
just tabletop-cpu       # force CPU inference
```

This runs `TabletopSeg3D/3DDetection/scripts/realtime_open3d_scene.py`. Per-frame
output is a JSON record with the scene point count, the estimated table normal, and
one entry per stable in-workspace detection (class, camera-frame center, extent,
yaw, point count, and tracker fields). The same detections can be published to ROS
for the robot (see [perception-ros2-moveit.md](perception-ros2-moveit.md)).

## Command reference

Most-used recipes first. Run `just` for the full list and `just help` for options.

| Command | Purpose |
| --- | --- |
| `just sync` | Create/update the Python environment |
| `just capture [--rgbd]` | Capture RGB (default) or RGB-D frames |
| `just labelme` | Open LabelMe on `dataset_labeled/manual_labelme/` |
| `just convert --data real\|point\|3dgs\|all` | Build the YOLO-seg dataset (required) |
| `just train` | Train the YOLO segmentation model |
| `just val` / `predict` / `predict-val` | Validate / run inference |
| `just compare-models` | Re-validate every `runs/segment/*/best.pt` on the current val split |
| `just analyze` / `analyze-labelme` / `analyze-yolo` / `analyze-compare` | Dataset statistics |
| `just stage` | Copy new captures into `dataset_prelabel/input_images/` |
| `just prelabel-yolo` / `prelabel-sam3` | Pre-label new images (YOLO / SAM3) |
| `just export-yolo` / `review-yolo` / `promote-yolo` | Pre-label review loop (YOLO; SAM3 variants exist) |
| `just prelabel-rgbd` / `review-rgbd` / `promote-rgbd` | Label RGB-D captures for the synthetic pipeline |
| `just synth-render` / `composite-synth` `--backend ...` | Build synthetic views |
| `just export-composite` / `review-composite` / `promote-composite` `--backend ...` | Promote synthetic views |
| `just view-depth` / `view-lift` / `lift-rgbd` | Inspect depth maps and lifted point clouds |
| `just tabletop` / `tabletop-headless` / `tabletop-cpu` | Realtime 3D detection |
| `just sense +args` | Forward raw arguments to `scripts/sense.sh` |

## Data tracking

- Training runs and weights (`runs/`, `*.pt`) are git-ignored.
- `DS/dataset_raw/`, `DS/dataset_labeled*/` and the `to_review_*/` folders are the
  tracked source of truth / in-progress review.
- Capture staging, pre-label working dirs and the converted `DS/dataset_yolo/` are
  git-ignored build outputs (see `.gitignore`). Back up RGB-D captures separately
  if you need to keep them.

## Known gaps / TODO

- The example model's known weak spots are not all documented here; validate per
  class before relying on it.
- The `gsplat` synthetic backend requires a matching CUDA toolkit and is not
  exercised in CI.
- Exact training hyperparameters live in `scripts/sense.sh` / Ultralytics defaults;
  they are not yet surfaced as a single config file.
- No automated tests cover the dataset conversion or training recipes.
