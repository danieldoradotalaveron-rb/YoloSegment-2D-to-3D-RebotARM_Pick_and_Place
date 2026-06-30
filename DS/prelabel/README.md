# Pre-labeling (offline tooling)

Offline helper to **pre-label new images** and feed them into the existing
dataset pipeline. Kept **out of the runtime** (`TabletopSeg3D/`): it never runs
on the robot/RealSense loop.

Scripts (2D autolabel):

- `prelabel_images.py`: predicts polygons on new images with one of two backends
  (`yolo` default, or `sam3`) and writes one JSON per image.
- `export_to_labelme.py`: converts prelabels to LabelMe JSON (backend-agnostic).
- `sam3_classes.yaml`: class names + text prompts for the SAM3 backend (single
  source of truth for the SAM3 pipeline).

Scripts (Phase 1B synthetic data, see "Synthetic pipeline" below):

- `prelabel_rgbd.py`: YOLO-label every `dataset_capture/rgbd/capture_*/rgb.png`
  into `to_review_rgbd_yolo/<capture_id>.{png,json}` for review. Skips captures
  that already have a label; lower default conf (0.4) since the synth lifts every
  mask. The auto-label alternative to drawing each `rgb.json` by hand.
- `promote_rgbd.py`: scatter reviewed `to_review_rgbd_yolo/<id>.json` back as
  `rgbd/<id>/rgb.json` (rewrites `imagePath`, keeps a differing hand-edited label).
- `lift_rgbd.py`: lift labeled RGBD masks to 3D colored points (shared helpers).
- `labeled_gaussians.py`: data container; one gaussian per lifted point, with
  `class_id`/`instance_id`. Saved as `gaussians.npz` (+ `.meta.yaml`). The
  `points` render uses `xyz`/`rgb`/`instance_id`; `scales_log`/`quats_wxyz`/
  `opacity` are only used by the optional `gsplat` backend.
- `init_labeled_gaussians.py`: lift in RAM -> `synth_render_<backend>/<capture>/gaussians.npz`.
- `synth_camera.py`: virtual cameras (yaw ±15°, pitch ±10°) + projection.
- `render_synth_views.py`: render object RGB + instance-id views per virtual camera.
- `synth_3dgs.py`: orchestrator (init + render, 2 steps) behind `just synth-render`.
- `composite_backgrounds.py`: paste the synthetic object on a real RGB background
  (copy-paste augmentation). Only the **object** is rotated (yaw/pitch from
  `synth_render_<backend>`); the background photo is used **flat** — no reprojection, so no
  black borders and every view is kept. Honors an **occluder** mask (any LabelMe
  shape whose label starts with `_`, e.g. `_occluder_grp` for the gripper) read
  from the capture's `rgb.json`: those foreground pixels keep the real background
  on top and are removed from the exported instance mask. A view is dropped only
  if no object remains visible (rotated out of frame or fully occluded).
- `export_synth_labelme.py` / `export_composite_labelme.py`: instance-id masks ->
  LabelMe polygons for review (the composite exporter reuses the synth helpers).
- `view_lift_ply.py`: Open3D viewer for a lifted `.ply`, aligned to the camera.

## Backends

**`yolo` (default) — recommended now.** Uses the project's own trained model
(`runs/segment/*/weights/best.pt`, latest by default). Best for the classes it
already knows (whatever you trained it on; see `DS/yolo_classes.yaml`), needs no
extra download, and runs on the same `torch` used for training.

**`sam3` — for new/unknown classes.** Ultralytics SAM3 text-based concept
segmentation (`SAM3SemanticPredictor`). In Ultralytics, SAM3 image-exemplar
prompts are boxes drawn on the *same* image (in-image hints), not a cross-image
crop library, so the practical automatic prompt is **text**. Classes and their
prompts live in `DS/prelabel/sam3_classes.yaml`:

```yaml
classes:
  - name: cork
    prompt: wine cork
  - name: lighter
    prompt: lighter
  - name: pompom
    prompt: pom pom ball
```

SAM3 ships inside `ultralytics`, but the weights `sam3.pt` are **gated** and not
auto-downloaded. Request access and download from
<https://huggingface.co/facebook/sam3>, then place `sam3.pt` in the repo root
(or pass `--model /path/to/sam3.pt`). Optional helper deps: `uv sync --extra sam3`.
The CLIP text encoder is auto-installed by ultralytics on the first SAM3 run.

## Flow

The two backends are fully separate: each has its own commands and folders, so
they never overwrite each other. Suffix everything with the backend (`-yolo` or
`-sam3`).

```text
new images
DS/dataset_prelabel/input_images/
        │  prelabel-yolo                       │  prelabel-sam3
        ▼                                      ▼
prelabels_yolo/                          prelabels_sam3/      # predicted polygons (JSON per image)
        │  export-yolo                         │  export-sam3
        ▼                                      ▼
to_review_yolo/                          to_review_sam3/      # LabelMe .jpg + .json -> human review
        │  promote-yolo                        │  promote-sam3
        ▼                                      ▼
dataset_labeled/reviewed_yolo_labelme/   dataset_labeled/reviewed_sam3_labelme/   # human-owned training pool
        └──────────────┬───────────────────────┘
                       │  just convert --data real  (--data REQUIRED: real|point|3dgs|all)
                       ▼
DS/dataset_yolo/                                             # YOLO-Seg, ready to train
```

### 1. New images -> prelabels

Drop new, unlabeled images into `DS/dataset_prelabel/input_images/`, then run the
backend you want:

```bash
just prelabel-yolo     # YOLO backend (latest best.pt) -> prelabels_yolo/
just prelabel-sam3     # SAM3 text-prompt backend       -> prelabels_sam3/

# extra flags pass straight through, e.g.:
just prelabel-yolo --device 0 --conf 0.3
```

SAM3 classes and their text prompts are defined in **`DS/prelabel/sam3_classes.yaml`**
(the single source of truth for the SAM3 pipeline). Edit that file to add classes or
tune ambiguous concepts; `name` is the output label, `prompt` is what SAM3 searches for:

```yaml
classes:
  - name: cork
    prompt: wine cork      # tune the prompt without changing the label
  - name: pompom
    prompt: pom pom ball
```

Output: one `<stem>.json` per image (class, score, bbox, polygon) in the matching
`prelabels_yolo/` or `prelabels_sam3/` folder.

### 2. Human review in LabelMe / CVAT

Export the prelabels to LabelMe format, then review/fix them by hand:

```bash
just export-yolo       # prelabels_yolo/ -> to_review_yolo/  (keeps existing reviews)
just review-yolo       # open LabelMe on to_review_yolo/

just export-sam3       # prelabels_sam3/ -> to_review_sam3/  (keeps existing reviews)
just review-sam3       # open LabelMe on to_review_sam3/
```

Filter weak predictions with `--min-score` if needed (passes through):

```bash
just export-yolo --min-score 0.4
```

`export-*` is **non-destructive**: if a reviewed `.json` already exists it is
preserved (pass `--overwrite` to discard it and regenerate). `review-*` edits the
JSONs in place, so the `to_review_<backend>/` folder holds the finished labels.
You can also import the folder into CVAT instead.

### 3. Promote reviewed labels into the training pool

Once reviewed, promote the pairs into the human-owned pool `DS/dataset_labeled/`:

```bash
just promote-yolo      # to_review_yolo/ -> dataset_labeled/reviewed_yolo_labelme/
just promote-sam3      # to_review_sam3/ -> dataset_labeled/reviewed_sam3_labelme/
```

They now belong to `dataset_labeled/`, which `just convert` reads recursively, so
rebuild and retrain (`--data` is REQUIRED; `real` = real-only baseline):

```bash
just convert --data real   # dataset_labeled/ (real) -> DS/dataset_yolo/
just train
```

## Folders

| Folder | Content | Tracked in git |
| --- | --- | --- |
| `input_images/` | new images to pre-label | structure only (`.gitkeep`) |
| `prelabels_yolo/` / `prelabels_sam3/` | predicted polygons (JSON per image) | structure only |
| `to_review_yolo/` / `to_review_sam3/` | LabelMe pairs to review (edited in place) | structure only |
| `exports_yolo/` | optional ad-hoc YOLO-Seg export | structure only |

Reviewed labels are not consumed from here directly; promote them into
`DS/dataset_labeled/reviewed_*_labelme/` (tracked) and let `just convert --data ...` pick them up.
Outputs here are ignored by `.gitignore` (only `.gitkeep` markers are committed).

## Synthetic pipeline (Phase 1B)

Multiply each reviewed RGBD capture into extra labeled 2D views by lifting its
objects to 3D and re-rendering them from nearby virtual cameras. Masks are
propagated geometrically, so labels stay reliable.

Labeling each capture's `rgb.json` can be manual (draw in LabelMe) or
auto-labeled with YOLO then reviewed:

```text
dataset_capture/rgbd/<capture>/rgb.png
        │  just prelabel-rgbd   (prelabel_rgbd.py, YOLO, skips already-labeled)
        ▼
to_review_rgbd_yolo/<capture>.png + .json
        │  just review-rgbd  (fix masks)  ->  just promote-rgbd  (promote_rgbd.py)
        ▼
dataset_capture/rgbd/<capture>/rgb.json   (LabelMe; the synth input below)
```

> **Occluder (gripper):** in `review-rgbd`, draw a polygon over the gripper and
> label it `_occluder_grp`. Any label starting with `_` is treated as an occluder,
> **not** a class: it is never lifted to 3D, never added to `yolo_classes.yaml`,
> and never seen by `convert`. `composite-synth` uses it to keep the gripper on top
> of pasted objects (and trims it from the exported masks). Re-promote afterwards
> with `just promote-rgbd --overwrite`.

`<backend>` is `point` (for `--backend points`) or `3dgs` (for `--backend gsplat`).
Both pipelines are parallel and symmetric; `<backend>` is REQUIRED on every step.

```text
dataset_capture/rgbd/<capture>/        rgb.png + depth.npy + intrinsics.yaml + rgb.json (LabelMe)
        │  just synth-render --backend points|gsplat   (init -> render, 2 steps)
        ▼
dataset_prelabel/synth_render_<backend>/<capture>/   gaussians.npz + <capture>_view_*.png/.instance.png/.meta.yaml
        │                                          │
        │ just export-synth --backend ...          │ just composite-synth --backend ...  (+ rgbd_backgrounds/<capture>/)
        ▼                                          ▼
to_review_synth_<backend>/                 dataset_prelabel/composited_views_<backend>/<capture>/<view>/
   <capture>_view_*.png + .json                   rgb.png + instance.png + metadata.json
        │ review-synth (QC ONLY, not promoted)     │ just export-composite --backend ...
        ▼                                          ▼
   (discard bad renders before compositing)    to_review_composite_<backend>/ -> review-composite / promote-composite
                                                   ▼
                                           dataset_labeled_<backend>/composited_labelme/
        └──────────────────────┬───────────────────────────────────┘
                               │  just convert --data real|point|3dgs|all
                               ▼
                       DS/dataset_yolo/    YOLO-Seg, ready to train
```

Notes:

- Default scope is **all labeled captures**; pass `--capture capture_XXXXXX` to limit.
- `--backend` is REQUIRED (no default): `points` -> `*_point` folders + `dataset_labeled_point/`;
  `gsplat` -> `*_3dgs` folders + `dataset_labeled_3dgs/`. `--backend gsplat` needs the
  CUDA toolkit (nvcc), not just the driver; the instance-id mask is always
  rasterized with `points`, so labels match either backend.
- `just convert --data real|point|3dgs|all` is REQUIRED and regenerates the dataset from scratch
  (not cumulative). `all` = real + point + 3dgs; synth stems are namespaced (`point__…`, `3dgs__…`)
  on the YOLO side so the two pipelines never collide. Val stays 100% real.
- `composite-synth` needs an empty-scene background per capture in
  `dataset_capture/rgbd_backgrounds/<capture>/` (paired via `just capture --rgbd`, mando `<-`).
  `--self-bg` reuses the original capture as background for a no-hardware test only
  (leaves real objects in the scene -> not for training).
- Idempotent and non-destructive: build folders are regenerated; `export-*` keeps
  existing reviews (`--overwrite` to force re-export). `promote-*` will NOT overwrite a
  label already promoted that differs from the reviewed one — it keeps the promoted copy
  and warns (so a regenerated/re-exported review never silently clobbers your promoted
  work). This also means a NEW fix in `to_review_*` is NOT pushed unless you ask: pass
  `promote-* --overwrite` to replace differing labels, optionally scoped with
  `--capture <id>` to act on a single capture/file.
- `prelabel-rgbd` skips captures that already have `rgb.json`, and `promote-rgbd`
  won't overwrite a differing `rgb.json` in the capture folder (both idempotent;
  pass `--overwrite` to force).

### Synthetic folders

| Folder | Content | Tracked in git |
| --- | --- | --- |
| `synth_render_{point,3dgs}/` | gaussians.npz + synthetic object views | structure only (`.gitkeep`) |
| `composited_views_{point,3dgs}/` | object-over-real-background views | structure only |
| `to_review_synth_{point,3dgs}/` / `to_review_composite_{point,3dgs}/` | LabelMe pairs to review (edited in place) | tracked (in-progress review) |
| `to_review_rgbd_yolo/` | YOLO prelabels of `rgbd/` captures to review | tracked (in-progress review) |
| `dataset_labeled_{point,3dgs}/` (under `DS/`) | promoted synth pools (train-only) | tracked (human-owned) |

## Dependencies

Both backends run on the already-pinned `ultralytics` + `torch` used for YOLO
training, so the runtime is untouched. The `yolo` backend needs nothing extra.
The `sam3` backend additionally needs the gated `sam3.pt` weights (see above);
`uv sync --extra sam3` adds `huggingface-hub` to help download them.

The synthetic `gsplat` backend is optional: `uv sync --extra 3dgs` installs it,
but it JIT-compiles CUDA kernels at render time and needs a matching CUDA toolkit.
The default `points` backend has no extra dependency.
