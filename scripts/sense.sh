#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

CONFIG_FILE="${SENSE_CONFIG:-$ROOT/config/sense.yaml}"
DS="$ROOT/DS"
PRELABEL="$ROOT/DS/prelabel"
DET="$ROOT/TabletopSeg3D/3DDetection"
RUNS_ROOT="$ROOT/runs"

UV_BIN="${UV_BIN:-uv}"
DATA_YAML="$ROOT/.generated/dataset_yolo.local.yaml"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Not found: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Cannot find '$1' in PATH. Install uv first."
}

uv_run() {
  "$UV_BIN" run --project "$ROOT" "$@"
}

uv_sync() {
  "$UV_BIN" sync --project "$ROOT"
}

# Parse the REQUIRED --backend flag of the synth/composite pipelines and derive
# the folder suffix. Sets globals BACKEND (points|gsplat) and SUFFIX (point|3dgs).
# The two pipelines are parallel: every pipeline-exclusive folder is suffixed so
# they never overwrite each other (file stems stay identical, see convert).
parse_backend() {
  BACKEND=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--backend" ]]; then
      BACKEND="${2:-}"
      break
    fi
    shift
  done
  [[ -n "$BACKEND" ]] || die "Missing required --backend (points | gsplat)."
  case "$BACKEND" in
    points) SUFFIX="point" ;;
    gsplat) SUFFIX="3dgs" ;;
    *) die "Invalid --backend '$BACKEND' (use: points | gsplat)." ;;
  esac
}

# Drop "--backend VALUE" from the args, leaving the rest in the STRIPPED array.
# Used for Python scripts that route by explicit path and don't accept --backend.
strip_backend() {
  STRIPPED=()
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--backend" ]]; then
      shift 2 || shift
      continue
    fi
    STRIPPED+=("$1")
    shift
  done
}

# Return 0 if a bare flag (e.g. --overwrite) appears anywhere in the args.
has_flag() {
  local needle="$1"; shift
  local a
  for a in "$@"; do [[ "$a" == "$needle" ]] && return 0; done
  return 1
}

# Echo the value following a flag (e.g. --capture ID); empty if the flag is absent.
flag_value() {
  local needle="$1"; shift
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "$needle" ]]; then printf '%s' "${2:-}"; return 0; fi
    shift
  done
  printf ''
}

# Promote reviewed (image,json) pairs from SRC into the human-owned pool DST.
# Reads globals: PROMOTE_OVERWRITE (true|false), PROMOTE_CAPTURE (stem-prefix filter or "").
# Sets globals: PROMOTED (copied), UPDATED (of those, replaced an existing differing
# label), KEPT (left untouched because dest differed and --overwrite was not given).
# Default is non-destructive: a differing label already in DST is kept (a re-export
# or regenerated review never silently clobbers your promoted work). Pass --overwrite
# to replace differing labels (optionally scoped with --capture to act per file/capture).
promote_pairs() {
  local src="$1" dst="$2"
  [[ -d "$src" ]] || die "Not found: $src (export it and review it first)."
  shopt -s nullglob
  local jsons=("$src"/*.json)
  shopt -u nullglob
  [[ ${#jsons[@]} -gt 0 ]] || die "No reviewed .json in $src. Nothing to promote."
  mkdir -p "$dst"
  PROMOTED=0; UPDATED=0; KEPT=0
  local jf stem img ext dst_json
  for jf in "${jsons[@]}"; do
    stem="$(basename "${jf%.json}")"
    if [[ -n "$PROMOTE_CAPTURE" && "$stem" != "$PROMOTE_CAPTURE"* ]]; then
      continue
    fi
    img=""
    for ext in jpg jpeg png bmp; do
      [[ -f "$src/$stem.$ext" ]] && img="$src/$stem.$ext" && break
    done
    [[ -n "$img" ]] || { echo "  skip (no image): $stem"; continue; }
    dst_json="$dst/$(basename "$jf")"
    if [[ -f "$dst_json" ]] && ! cmp -s "$jf" "$dst_json"; then
      if [[ "$PROMOTE_OVERWRITE" == true ]]; then
        echo "  overwrite (dest differs, replacing): $stem"
        UPDATED=$((UPDATED + 1))
      else
        echo "  keep (dest differs, NOT overwriting): $stem  (pass --overwrite to replace)"
        KEPT=$((KEPT + 1))
        continue
      fi
    fi
    cp -f "$img" "$dst/"
    cp -f "$jf" "$dst/"
    PROMOTED=$((PROMOTED + 1))
  done
}

cfg() {
  local key="$1"
  local default="${2-}"
  local found

  require_file "$CONFIG_FILE"
  found="$(
    awk -v key="$key" '
      /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
      {
        k = $0
        sub(/:.*/, "", k)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
        if (k == key) {
          v = $0
          sub(/^[^:]*:/, "", v)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
          if (v ~ /^".*"$/ || v ~ /^'\''.*'\''$/) {
            v = substr(v, 2, length(v) - 2)
          }
          print "__FOUND__" v
          exit
        }
      }
    ' "$CONFIG_FILE"
  )"

  if [[ "$found" == __FOUND__* ]]; then
    printf '%s' "${found#__FOUND__}"
  else
    printf '%s' "$default"
  fi
}

is_true() {
  case "${1,,}" in
    true|1|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

positive_int_or_default() {
  local value="$1"
  local default="$2"
  if [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 )); then
    printf '%s' "$value"
  else
    printf '%s' "$default"
  fi
}

nonzero_number() {
  case "${1:-}" in
    ''|0|0.0|0.00|0.000) return 1 ;;
    *) return 0 ;;
  esac
}

resolve_existing_path() {
  local value="$1"
  if [[ -z "$value" ]]; then
    printf ''
  elif [[ -e "$value" ]]; then
    realpath "$value"
  elif [[ -e "$ROOT/$value" ]]; then
    realpath "$ROOT/$value"
  else
    printf '%s' "$value"
  fi
}

resolve_best_pt() {
  local run_override="$1"
  local candidate
  local seg_root="$RUNS_ROOT/segment"

  if [[ -n "$run_override" ]]; then
    candidate="$(resolve_existing_path "$run_override")"
    if [[ -f "$candidate" ]]; then
      realpath "$candidate"
      return
    fi

    candidate="$seg_root/$run_override/weights/best.pt"
    if [[ -f "$candidate" ]]; then
      realpath "$candidate"
      return
    fi

    die "best.pt not found for run='$run_override' (tried: $candidate)"
  fi

  [[ -d "$seg_root" ]] || die "No runs/segment yet. Run 'train' first."

  candidate="$(
    find "$seg_root" -type f -path '*/weights/best.pt' -printf '%T@ %p\n' 2>/dev/null |
      sort -nr |
      head -n 1 |
      cut -d' ' -f2-
  )"
  [[ -n "$candidate" ]] || die "No best.pt found under $seg_root"
  realpath "$candidate"
}

ensure_dataset_yaml() {
  local source_yaml="$DS/dataset_yolo/data.yaml"

  require_file "$source_yaml"
  mkdir -p "$(dirname "$DATA_YAML")"
  awk -v dataset_root="$DS/dataset_yolo" '
    BEGIN { wrote_path = 0 }
    /^path:[[:space:]]*/ { print "path: " dataset_root; wrote_path = 1; next }
    { print }
    END { if (!wrote_path) print "path: " dataset_root }
  ' "$source_yaml" > "$DATA_YAML"
}

run_uv_in_dir() {
  local dir="$1"
  shift
  (cd "$dir" && uv_run "$@")
}

show_help() {
  cat <<'HELP'
YOLO-Seg 2D->3D pipeline. Run a command with `just <command>` or
`bash scripts/sense.sh <command> [options]`. Below, each command has a one-line
purpose; arrows (->) point to the typical next step.

SETUP
  sync                       Install/update the Python env from pyproject + uv.lock.
  capture [--rgbd]           Capture from RealSense: RGB frames (default) or paired
                             RGB-D object+background (--rgbd). -> stage / prelabel-rgbd
                             flags: [--overwrite] [--depth-filters] [--depth-preset NAME]
  stage                      Copy new RGB captures dataset_capture/rgb/ -> input_images/. -> prelabel-yolo

WORKFLOW A - real manual labeling (the val set comes from here)
  labelme                    Open LabelMe on dataset_labeled/manual_labelme/ to hand-draw masks.
                             -> convert --data real

WORKFLOW B1 - 2D auto-labeling of new RGB images (model in the loop)
  prelabel-yolo              Predict masks on input_images/ with the latest model -> prelabels_yolo/.
  prelabel-sam3              Same, using the open-vocab SAM3 text-prompt backend (needs sam3.pt).
  export-yolo / export-sam3  Turn prelabels into LabelMe pairs in to_review_*/ (keeps reviews; --overwrite).
  review-yolo / review-sam3  Open LabelMe to fix the prelabels (approve, don't redraw).
  promote-yolo / promote-sam3  Copy reviewed pairs -> dataset_labeled/reviewed_*_labelme/ (real pool).
                             flags: [--overwrite] [--capture ID]   -> convert --data real

WORKFLOW B2 - synthetic data from RGB-D (TWO parallel backends)
  --backend is REQUIRED on every step: points -> *_point folders (CPU),
  gsplat -> *_3dgs folders (3D Gaussian rasterization, needs CUDA toolkit).
  prelabel-rgbd              YOLO-label rgbd/ captures -> to_review_rgbd_yolo/. [--capture ID] [--conf C] [--overwrite]
  review-rgbd                Fix the RGB-D masks (also label the gripper as `_occluder_grp`).
  promote-rgbd               Write reviewed masks back into rgbd/<id>/rgb.json. [--capture ID] [--overwrite]
  synth-render --backend B   Init gaussians + render synthetic object views (2 steps). [--inpaint-depth] [--capture ID]
  export-synth --backend B   Render views -> LabelMe pairs in to_review_synth_<B>/. [--capture ID] [--overwrite]
  review-synth --backend B   Open LabelMe to QC synthetic views (not promoted; QC step only).
  composite-synth --backend B  Paste objects over the real empty background -> composited_views_<B>/.
                             flags: [--capture ID] [--self-bg] [--bg-capture ID]
  export-composite --backend B   Composited views -> LabelMe pairs in to_review_composite_<B>/. [--capture ID] [--overwrite]
  review-composite --backend B   Open LabelMe to fix/discard composited views.
  promote-composite --backend B  -> dataset_labeled_<B>/composited_labelme/. [--overwrite] [--capture ID]

BUILD DATASET + TRAIN + EVALUATE
  convert --data SEL         REQUIRED selector: real | point | 3dgs | all. Builds DS/dataset_yolo/
                             (real=baseline; point/3dgs=real+that synth pool; all=real+point+3dgs).
                             Synthetic goes to train only; val stays 100% real. Regenerated each run.
  train                      Train YOLO-Seg on DS/dataset_yolo/ -> runs/segment/*/weights/best.pt.
  val                        Validate the latest (or --run) model on the val split.
  predict / predict-val      Run inference on val images (predict-val = the val folder).
  compare-models [--imgsz N] Re-validate every runs/segment/*/best.pt on the current val split (ranked table).
  analyze | analyze-labelme | analyze-yolo | analyze-compare   Dataset stats / LabelMe-vs-YOLO drift.

WORKFLOW C - realtime 3D detection
  tabletop                   RealSense + YOLO + Open3D viewer with 3D boxes.
  tabletop-headless          Same without GUI; prints per-frame JSON.
  tabletop-cpu               Force CPU inference.

INSPECT / DEBUG
  view-depth --depth-file F.npy [--rgb] [--save OUT.png]   Visualize a depth map.
  lift-rgbd [--inpaint-depth] [--save-ply]                 Lift labeled RGB-D masks to 3D (report).
  view-lift --capture ID                                   Open3D view of a lifted point cloud.

KEY NOTES
  * --backend (points|gsplat): selects rasterizer AND folder suffix (_point / _3dgs). The two
    synthetic pipelines are parallel; convert namespaces them (point__/3dgs__) so they never collide.
  * promote-* are non-destructive: a label already promoted that DIFFERS is kept (warned), so neither
    a regenerated review nor a new fix is pushed silently. Use --overwrite (optionally --capture ID,
    which scopes to one capture/file) to force-replace differing labels.
  * Model selection: val/predict/tabletop use the most recent runs/segment/*/weights/best.pt,
    unless you pass --run NAME.

COMMON OPTIONS
  --run NAME                 Pick a specific training run (else: latest best.pt).
  --imgsz N / --conf C       Inference image size / confidence threshold.
  --capture ID               Scope a command to one capture_XXXXXX.
  --depth-filters / --depth-preset NAME   RealSense depth post-processing.
  --show-2d / --demo-visual / --view2d-* / --view3d-* / --source / --serial   Viewer + capture tuning.
  --freeze-normal / --live-normal / --table-normal-every N   Table normal: warm-up-only vs per-frame (eye-in-hand).

  Config defaults: config/sense.yaml      Full list: every option above maps to a sense.yaml key.

EXAMPLES
  just sync && just capture --rgbd
  just synth-render --backend points --inpaint-depth
  just composite-synth --backend points && just export-composite --backend points
  just review-composite --backend points && just promote-composite --backend points
  just convert --data point && just train
  just promote-composite --backend points --overwrite --capture capture_000004
HELP
}

# Focused per-command help, shown when any command is run with --help / -h.
command_help() {
  case "$1" in
    sync) cat <<'H'
just sync
  Install/update the Python environment from pyproject.toml + uv.lock (uv sync).
H
    ;;
    capture) cat <<'H'
just capture [--rgbd] [--overwrite] [--depth-filters] [--depth-preset NAME] [--serial S]
  Capture from the RealSense camera.
    (default)  RGB frames            -> DS/dataset_capture/rgb/
    --rgbd     paired RGB-D object + empty background -> dataset_capture/rgbd[ _backgrounds]/
  Presenter: -> saves the object, <- saves the paired empty background (same capture_id).
  IDs auto-increment (capture_000NNN); never overwrites (backgrounds need --overwrite).
  Next: just stage (RGB)  |  just prelabel-rgbd (RGB-D)
H
    ;;
    stage) cat <<'H'
just stage
  Copy NEW RGB captures DS/dataset_capture/rgb/ -> DS/dataset_prelabel/input_images/
  (skips files already staged; never overwrites). Next: just prelabel-yolo
H
    ;;
    labelme) cat <<'H'
just labelme
  Open LabelMe on DS/dataset_labeled/manual_labelme/ for first-pass hand labeling.
  This is the REAL pool that feeds both train and the 100%-real val split.
  Next: just convert --data real
H
    ;;
    prelabel-yolo|prelabel-sam3) cat <<'H'
just prelabel-yolo | prelabel-sam3   [extra flags pass through to the model]
  Run inference on DS/dataset_prelabel/input_images/ to produce prelabels:
    prelabel-yolo  -> prelabels_yolo/  (uses the latest trained model)
    prelabel-sam3  -> prelabels_sam3/  (open-vocab text-prompt backend; needs sam3.pt)
  Next: just export-yolo | export-sam3
H
    ;;
    export-yolo|export-sam3) cat <<'H'
just export-yolo | export-sam3   [--overwrite]
  Convert prelabels_*/ into LabelMe pairs (.png + .json) in to_review_*/ for review.
  Keeps any existing review JSON unless --overwrite. Next: just review-yolo | review-sam3
H
    ;;
    review-yolo|review-sam3) cat <<'H'
just review-yolo | review-sam3
  Open LabelMe on to_review_yolo/ | to_review_sam3/ to fix/approve the prelabels
  (you correct, not redraw). Next: just promote-yolo | promote-sam3
H
    ;;
    promote-yolo|promote-sam3) cat <<'H'
just promote-yolo | promote-sam3   [--overwrite] [--capture ID]
  Copy reviewed pairs to_review_*/ -> DS/dataset_labeled/reviewed_*_labelme/ (REAL pool).
  Non-destructive: a label already promoted that DIFFERS is kept (warned), so a re-export
  never clobbers your work AND a new fix is not pushed silently.
    --overwrite     replace differing labels with the reviewed ones
    --capture ID    scope the overwrite/promote to one capture_XXXXXX (per file/capture)
  Next: just convert --data real
H
    ;;
    prelabel-rgbd) cat <<'H'
just prelabel-rgbd   [--capture ID] [--conf C] [--overwrite]
  YOLO-label the RGB-D captures in dataset_capture/rgbd/ -> to_review_rgbd_yolo/
  (skips captures that already have rgb.json unless --overwrite). Next: just review-rgbd
H
    ;;
    review-rgbd) cat <<'H'
just review-rgbd
  Open LabelMe on to_review_rgbd_yolo/ to fix the object masks AND draw the gripper as
  an occluder labeled `_occluder_grp` (any label starting with `_` is an occluder: never
  lifted to 3D nor a class, only used to mask composites). Next: just promote-rgbd
H
    ;;
    promote-rgbd) cat <<'H'
just promote-rgbd   [--capture ID] [--overwrite]
  Write the reviewed masks back into each dataset_capture/rgbd/<id>/rgb.json (the synth
  input). Won't overwrite a differing rgb.json unless --overwrite.
  Next: just synth-render --backend points|gsplat
H
    ;;
    synth-render) cat <<'H'
just synth-render --backend points|gsplat   [--inpaint-depth] [--capture ID]
  Build synthetic object views in TWO steps (init gaussians + render). --backend is REQUIRED:
    points -> DS/dataset_prelabel/synth_render_point/   (CPU point-splat)
    gsplat -> DS/dataset_prelabel/synth_render_3dgs/    (3D Gaussian rasterization, needs CUDA toolkit)
  Processes only captures that have a reviewed rgb.json. Next: composite-synth (or QC via export-synth/review-synth)
H
    ;;
    export-synth) cat <<'H'
just export-synth --backend points|gsplat   [--capture ID] [--overwrite]
  Export synth_render_<backend> views -> LabelMe pairs in to_review_synth_<backend>/ for QC.
  Object-on-black views are NOT promoted (QC only). Next: just review-synth --backend ...
H
    ;;
    review-synth) cat <<'H'
just review-synth --backend points|gsplat
  Open LabelMe on to_review_synth_<backend>/ to QC the object-on-black views.
  This is a QC step only (no promote). Next: just composite-synth --backend ...
H
    ;;
    composite-synth) cat <<'H'
just composite-synth --backend points|gsplat   [--capture ID] [--self-bg] [--bg-capture ID]
  Paste each rendered object over its matched real empty background (lower domain gap)
  -> DS/dataset_prelabel/composited_views_<backend>/. Honors the `_occluder_grp` gripper mask.
    --self-bg   reuse the original capture as background (smoke test only, leaves real objects)
  Needs backgrounds in dataset_capture/rgbd_backgrounds/. Next: just export-composite --backend ...
H
    ;;
    export-composite) cat <<'H'
just export-composite --backend points|gsplat   [--capture ID] [--overwrite]
  Export composited_views_<backend> -> LabelMe pairs in to_review_composite_<backend>/.
  Keeps existing review JSON unless --overwrite. Next: just review-composite --backend ...
H
    ;;
    review-composite) cat <<'H'
just review-composite --backend points|gsplat
  Open LabelMe on to_review_composite_<backend>/ to fix/discard composited views.
  Next: just promote-composite --backend ...
H
    ;;
    promote-composite) cat <<'H'
just promote-composite --backend points|gsplat   [--overwrite] [--capture ID]
  Copy reviewed pairs -> DS/dataset_labeled_<backend>/composited_labelme/ (train-only pool).
  Non-destructive (keeps a differing promoted label, warned); --overwrite to replace,
  --capture ID to scope per file/capture. Next: just convert --data <backend>|all
H
    ;;
    convert) cat <<'H'
just convert --data real|point|3dgs|all     (--data is REQUIRED)
  Build DS/dataset_yolo/ from the labeled pools (regenerated from scratch each run):
    real   only dataset_labeled/ (baseline)
    point  real + dataset_labeled_point/   (synth, train-only)
    3dgs   real + dataset_labeled_3dgs/    (synth, train-only)
    all    real + point + 3dgs
  Synthetic goes to TRAIN only (val stays 100% real); synth stems are namespaced
  (point__/3dgs__) so the two pipelines never collide. Next: just train
H
    ;;
    train) cat <<'H'
just train
  Train YOLO-Seg on DS/dataset_yolo/ -> runs/segment/*/weights/best.pt
  (imgsz from config/sense.yaml). Run `just convert --data ...` first.
H
    ;;
    val) cat <<'H'
just val   [--run NAME]
  Validate a trained model on the current val split (100% real).
  Uses the latest runs/segment/*/weights/best.pt unless --run NAME is given.
H
    ;;
    predict|predict-val) cat <<'H'
just predict | predict-val   [--run NAME]
  Run inference and save annotated images. predict-val targets the val image folder.
  Uses the latest best.pt unless --run NAME.
H
    ;;
    compare-models) cat <<'H'
just compare-models   [--imgsz N]
  Re-validate every runs/segment/*/weights/best.pt on the CURRENT val split and print a
  ranked table (fair, apples-to-apples comparison across your trained runs).
H
    ;;
    analyze|analyze-labelme|analyze-yolo|analyze-compare) cat <<'H'
just analyze | analyze-labelme | analyze-yolo | analyze-compare
  Dataset statistics: class/instance counts. analyze-compare flags LabelMe-vs-YOLO drift
  (conversion mismatches). Read-only.
H
    ;;
    tabletop|tabletop-headless|tabletop-cpu) cat <<'H'
just tabletop | tabletop-headless | tabletop-cpu   [--run NAME] [--show-2d] [viewer flags]
  Realtime 3D detection (RealSense + YOLO + lift to 3D boxes):
    tabletop          Open3D viewer with 3D boxes
    tabletop-headless no GUI; prints per-frame JSON (class, center, size, yaw)
    tabletop-cpu      force CPU inference
  Uses the latest best.pt unless --run NAME.
H
    ;;
    view-depth) cat <<'H'
just view-depth --depth-file FILE.npy   [--rgb] [--save OUT.png]
  Visualize a depth.npy map (optionally alongside RGB, or save to PNG).
H
    ;;
    lift-rgbd) cat <<'H'
just lift-rgbd   [--inpaint-depth] [--save-ply]
  Lift labeled RGB-D masks to 3D points and report per-instance coverage (Phase 1B sanity
  check). --save-ply writes a point cloud you can inspect with just view-lift.
H
    ;;
    view-lift) cat <<'H'
just view-lift --capture CAPTURE_ID
  Open3D viewer of a lifted lift_*.ply aligned to the RealSense capture orientation.
H
    ;;
    help|"") show_help ;;
    *) cat <<H
No detailed help for '$1'. Run 'just help' for the full command list.
H
    ;;
  esac
}

SERIAL="$(cfg serial 353322271636)"
IMGSZ="$(cfg imgsz 448)"
CONF="$(cfg conf 0.15)"
MIN_POINTS="$(cfg min_points 50)"
MIN_DEPTH="$(cfg min_depth 0.08)"
MAX_DEPTH="$(cfg max_depth 0.50)"
WORKSPACE="$(cfg workspace '')"
DEPTH_FILTERS="$(cfg depth_filters false)"
DEPTH_PRESET="$(cfg depth_preset '')"
TRACK_WINDOW="$(cfg track_window 10)"
TRACK_MIN_HITS="$(cfg track_min_hits 6)"
TRACK_MAX_MISSES="$(cfg track_max_misses 5)"
TRACK_MAX_POS_STD="$(cfg track_max_pos_std 0.008)"
TRACK_MAX_YAW_STD="$(cfg track_max_yaw_std 10.0)"
TRACK_MIN_CONF="$(cfg track_min_conf 0.8)"
TRACK_ASSOC_DIST="$(cfg track_assoc_dist 0.04)"
TRACK_STABLE_ENTER_FRAMES="$(cfg track_stable_enter_frames 3)"
TRACK_STABLE_HYSTERESIS="$(cfg track_stable_hysteresis 1.5)"
TRACK_STABLE_CONF_MARGIN="$(cfg track_stable_conf_margin 0.05)"
NON_SYMMETRIC_CLASSES="$(cfg non_symmetric_classes '')"
SYMMETRIC_CLASSES="$(cfg symmetric_classes '')"
TABLE_NORMAL_EVERY="$(cfg table_normal_every 1)"
SHOW_2D="$(cfg show_2d false)"
DEMO_VISUAL="$(cfg demo_visual false)"
POINT_STRIDE="$(cfg point_stride 0)"
SCENE_MAX_POINTS="$(cfg scene_max_points 0)"
VIEW2D_SIZE="$(cfg view2d_size '')"
VIEW2D_POS="$(cfg view2d_pos '')"
VIEW2D_LAYOUT="$(cfg view2d_layout rgb-only)"
VIEW2D_ORDER="$(cfg view2d_order rgb-first)"
VIEW2D_FULLSCREEN="$(cfg view2d_fullscreen false)"
VIEW2D_DEPTH_MIN="$(cfg view2d_depth_min 0.0)"
VIEW2D_DEPTH_MAX="$(cfg view2d_depth_max 0.0)"
VIEW3D_SIZE="$(cfg view3d_size '')"
VIEW3D_POS="$(cfg view3d_pos '')"
RAW_MODE="$(cfg raw_mode false)"
SOURCE="$(cfg source '')"
RUN="$(cfg run '')"
REAL="$(cfg real false)"
EE_T_CAM="$(cfg ee_t_cam '')"
ROS_TOPIC="$(cfg ros_topic '/perception/detections')"
WORKSPACE_BASE="$(cfg workspace_base '')"

while [[ "${1:-}" == "--" ]]; do
  shift
done

COMMAND="${1:-help}"
if (($# > 0)); then
  shift
fi

# Per-command help: `just <command> --help` (or -h) explains that command and exits,
# instead of running it (which for synth/promote would error on the missing --backend).
if has_flag --help "$@" || has_flag -h "$@"; then
  command_help "$COMMAND"
  exit 0
fi

while [[ "${1:-}" == "--" ]]; do
  shift
done

# Pre-labeling commands forward their remaining flags straight to the Python
# scripts, so handle them here BEFORE the strict runtime option parser below
# (which would reject unknown flags like --backend / --classes-config). Each backend
# (yolo / sam3) has its own folders so the two pipelines never overwrite each other.
PL_DIR="$DS/dataset_prelabel"
case "$COMMAND" in
  # --- YOLO pipeline (default backend; classes from DS/yolo_classes.yaml) ---
  prelabel-yolo)
    require_command "$UV_BIN"
    uv_run python "$PRELABEL/prelabel_images.py" \
      --backend yolo --output "$PL_DIR/prelabels_yolo" "$@"
    exit 0
    ;;
  export-yolo)
    require_command "$UV_BIN"
    uv_run python "$PRELABEL/export_to_labelme.py" \
      --prelabels "$PL_DIR/prelabels_yolo" --output "$PL_DIR/to_review_yolo" "$@"
    exit 0
    ;;
  review-yolo)
    REVIEW_DIR="$PL_DIR/to_review_yolo"
    [[ -d "$REVIEW_DIR" ]] || die "Not found: $REVIEW_DIR (run 'just export-yolo' first)"
    # Use the system 'labelme' (GUI tool); the venv one clashes with opencv's Qt plugin.
    require_command labelme
    labelme "$REVIEW_DIR"
    exit 0
    ;;
  # --- YOLO pre-label of RGBD captures for the synth pipeline ---
  # Predicts straight to LabelMe in to_review_rgbd_yolo/, then promote scatters the
  # reviewed labels back as rgb.json inside each rgbd/<id>/ (input of synth-render).
  prelabel-rgbd)
    require_command "$UV_BIN"
    uv_run python "$PRELABEL/prelabel_rgbd.py" "$@"
    exit 0
    ;;
  review-rgbd)
    REVIEW_DIR="$PL_DIR/to_review_rgbd_yolo"
    [[ -d "$REVIEW_DIR" ]] || die "Not found: $REVIEW_DIR (run 'just prelabel-rgbd' first)"
    require_command labelme
    labelme "$REVIEW_DIR"
    exit 0
    ;;
  promote-rgbd)
    require_command "$UV_BIN"
    uv_run python "$PRELABEL/promote_rgbd.py" "$@"
    exit 0
    ;;
  # --- SAM3 pipeline (text-prompt backend; needs sam3.pt) ---
  prelabel-sam3)
    require_command "$UV_BIN"
    uv_run python "$PRELABEL/prelabel_images.py" \
      --backend sam3 --output "$PL_DIR/prelabels_sam3" "$@"
    exit 0
    ;;
  export-sam3)
    require_command "$UV_BIN"
    uv_run python "$PRELABEL/export_to_labelme.py" \
      --prelabels "$PL_DIR/prelabels_sam3" --output "$PL_DIR/to_review_sam3" "$@"
    exit 0
    ;;
  review-sam3)
    REVIEW_DIR="$PL_DIR/to_review_sam3"
    [[ -d "$REVIEW_DIR" ]] || die "Not found: $REVIEW_DIR (run 'just export-sam3' first)"
    require_command labelme
    labelme "$REVIEW_DIR"
    exit 0
    ;;
  # --- Promote reviewed autolabels into the human-owned training pool ---
  promote-yolo|promote-sam3)
    backend="${COMMAND#promote-}"
    SRC="$PL_DIR/to_review_${backend}"
    DST="$DS/dataset_labeled/reviewed_${backend}_labelme"
    PROMOTE_OVERWRITE=false; has_flag --overwrite "$@" && PROMOTE_OVERWRITE=true
    PROMOTE_CAPTURE="$(flag_value --capture "$@")"
    promote_pairs "$SRC" "$DST"
    echo "Promoted $PROMOTED reviewed pairs ($UPDATED overwritten): $SRC -> $DST"
    [[ $KEPT -gt 0 ]] && echo "Kept $KEPT existing label(s) that differ from $SRC (pass --overwrite to replace)."
    echo "Next: just convert --data real (or point|3dgs|all), then just train"
    exit 0
    ;;
  # --- Offline capture (forwards flags to DS/capture.py) ---
  capture)
    require_command "$UV_BIN"
    cap_args=(--serial "$SERIAL")
    if is_true "$DEPTH_FILTERS"; then
      cap_args+=(--depth-filters)
    fi
    if [[ -n "$DEPTH_PRESET" ]]; then
      cap_args+=(--depth-preset "$DEPTH_PRESET")
    fi
    cap_args+=("$@")
    run_uv_in_dir "$DS" python "$DS/capture.py" "${cap_args[@]}"
    exit 0
    ;;
  view-depth)
    require_command "$UV_BIN"
    uv_run python "$DS/visualize_depth.py" "$@"
    exit 0
    ;;
  lift-rgbd)
    require_command "$UV_BIN"
    uv_run python "$PRELABEL/lift_rgbd.py" "$@"
    exit 0
    ;;
  view-lift)
    require_command "$UV_BIN"
    uv_run python "$PRELABEL/view_lift_ply.py" "$@"
    exit 0
    ;;
  synth-render)
    require_command "$UV_BIN"
    parse_backend "$@"
    strip_backend "$@"
    uv_run python "$PRELABEL/synth_3dgs.py" \
      --backend "$BACKEND" \
      --synth-root "$PL_DIR/synth_render_$SUFFIX" \
      "${STRIPPED[@]}"
    echo "Next: just composite-synth --backend $BACKEND  ->  export-composite  ->  review-composite  ->  promote-composite"
    echo "      (optional QC of the raw renders first: just export-synth --backend $BACKEND -> review-synth --backend $BACKEND)"
    exit 0
    ;;
  export-synth)
    require_command "$UV_BIN"
    parse_backend "$@"
    strip_backend "$@"
    uv_run python "$PRELABEL/export_synth_labelme.py" \
      --synth-root "$PL_DIR/synth_render_$SUFFIX" \
      --output "$PL_DIR/to_review_synth_$SUFFIX" \
      "${STRIPPED[@]}"
    echo "Next: just review-synth --backend $BACKEND   (QC the object-on-black renders; not promoted)"
    echo "      then: just composite-synth --backend $BACKEND"
    exit 0
    ;;
  review-synth)
    parse_backend "$@"
    REVIEW_DIR="$PL_DIR/to_review_synth_$SUFFIX"
    [[ -d "$REVIEW_DIR" ]] || die "Not found: $REVIEW_DIR (run 'just export-synth --backend $BACKEND' first)"
    require_command labelme
    labelme "$REVIEW_DIR"
    exit 0
    ;;
  composite-synth)
    require_command "$UV_BIN"
    parse_backend "$@"
    strip_backend "$@"
    uv_run python "$PRELABEL/composite_backgrounds.py" \
      --synth-root "$PL_DIR/synth_render_$SUFFIX" \
      --output-root "$PL_DIR/composited_views_$SUFFIX" \
      "${STRIPPED[@]}"
    exit 0
    ;;
  export-composite)
    require_command "$UV_BIN"
    parse_backend "$@"
    strip_backend "$@"
    uv_run python "$PRELABEL/export_composite_labelme.py" \
      --composite-root "$PL_DIR/composited_views_$SUFFIX" \
      --output "$PL_DIR/to_review_composite_$SUFFIX" \
      "${STRIPPED[@]}"
    exit 0
    ;;
  review-composite)
    parse_backend "$@"
    REVIEW_DIR="$PL_DIR/to_review_composite_$SUFFIX"
    [[ -d "$REVIEW_DIR" ]] || die "Not found: $REVIEW_DIR (run 'just export-composite --backend $BACKEND' first)"
    require_command labelme
    labelme "$REVIEW_DIR"
    exit 0
    ;;
  promote-composite)
    parse_backend "$@"
    PROMOTE_OVERWRITE=false; has_flag --overwrite "$@" && PROMOTE_OVERWRITE=true
    PROMOTE_CAPTURE="$(flag_value --capture "$@")"
    SRC="$PL_DIR/to_review_composite_$SUFFIX"
    DST="$DS/dataset_labeled_$SUFFIX/composited_labelme"
    promote_pairs "$SRC" "$DST"
    echo "Promoted $PROMOTED composited pairs ($UPDATED overwritten): $SRC -> $DST"
    [[ $KEPT -gt 0 ]] && echo "Kept $KEPT existing label(s) that differ from $SRC (pass --overwrite to replace)."
    echo "Next: just convert --data $SUFFIX (or --data all), then just train"
    exit 0
    ;;
  convert)
    require_command "$UV_BIN"
    uv_run python "$DS/convert_labelme_to_yolo.py" "$@"
    exit 0
    ;;
esac

while (($# > 0)); do
  case "$1" in
    --run) RUN="${2:?}"; shift 2 ;;
    --serial) SERIAL="${2:?}"; shift 2 ;;
    --imgsz) IMGSZ="${2:?}"; shift 2 ;;
    --conf) CONF="${2:?}"; shift 2 ;;
    --min-points) MIN_POINTS="${2:?}"; shift 2 ;;
    --min-depth) MIN_DEPTH="${2:?}"; shift 2 ;;
    --max-depth) MAX_DEPTH="${2:?}"; shift 2 ;;
    --workspace) WORKSPACE="${2:?}"; shift 2 ;;
    --depth-filters) DEPTH_FILTERS=true; shift ;;
    --depth-preset) DEPTH_PRESET="${2:?}"; shift 2 ;;
    --show-2d) SHOW_2D=true; shift ;;
    --raw-mode) RAW_MODE=true; shift ;;
    --no-raw-mode) RAW_MODE=false; shift ;;
    --table-normal-every) TABLE_NORMAL_EVERY="${2:?}"; shift 2 ;;
    --freeze-normal) TABLE_NORMAL_EVERY=0; shift ;;
    --live-normal) TABLE_NORMAL_EVERY=1; shift ;;
    --demo-visual) DEMO_VISUAL=true; shift ;;
    --point-stride) POINT_STRIDE="${2:?}"; shift 2 ;;
    --scene-max-points) SCENE_MAX_POINTS="${2:?}"; shift 2 ;;
    --view2d-size) VIEW2D_SIZE="${2:?}"; shift 2 ;;
    --view2d-pos) VIEW2D_POS="${2:?}"; shift 2 ;;
    --view2d-layout) VIEW2D_LAYOUT="${2:?}"; shift 2 ;;
    --view2d-order) VIEW2D_ORDER="${2:?}"; shift 2 ;;
    --view2d-fullscreen) VIEW2D_FULLSCREEN=true; shift ;;
    --view2d-depth-min) VIEW2D_DEPTH_MIN="${2:?}"; shift 2 ;;
    --view2d-depth-max) VIEW2D_DEPTH_MAX="${2:?}"; shift 2 ;;
    --view3d-size) VIEW3D_SIZE="${2:?}"; shift 2 ;;
    --view3d-pos) VIEW3D_POS="${2:?}"; shift 2 ;;
    --source) SOURCE="${2:?}"; shift 2 ;;
    --real) REAL=true; shift ;;
    --ee-t-cam) EE_T_CAM="${2:?}"; shift 2 ;;
    --topic) ROS_TOPIC="${2:?}"; shift 2 ;;
    --workspace-base) WORKSPACE_BASE="${2:?}"; shift 2 ;;
    *) die "Unrecognized option: $1" ;;
  esac
done

run_analyze() {
  require_command "$UV_BIN"
  uv_run python "$DS/analyze_dataset.py" --mode "$1"
}

run_tabletop() {
  local device="$1"
  local no_display="$2"
  local script_path="$DET/scripts/realtime_open3d_scene.py"
  local stride
  local max_points
  local use_show_2d="$SHOW_2D"
  local use_filters="$DEPTH_FILTERS"
  local use_preset="$DEPTH_PRESET"
  local view3d_size="$VIEW3D_SIZE"
  local view3d_pos="$VIEW3D_POS"
  local view2d_size="$VIEW2D_SIZE"
  local view2d_pos="$VIEW2D_POS"
  local cmd_args

  require_command "$UV_BIN"
  require_file "$script_path"

  stride="$(positive_int_or_default "$POINT_STRIDE" 2)"
  max_points="$(positive_int_or_default "$SCENE_MAX_POINTS" 80000)"

  if is_true "$DEMO_VISUAL" && [[ "$no_display" != true ]]; then
    if [[ ! "$POINT_STRIDE" =~ ^[0-9]+$ ]] || (( POINT_STRIDE <= 0 )); then
      stride=1
    fi
    if [[ ! "$SCENE_MAX_POINTS" =~ ^[0-9]+$ ]] || (( SCENE_MAX_POINTS <= 0 )); then
      max_points=160000
    fi
    use_show_2d=true
    if ! is_true "$use_filters"; then
      use_filters=true
    fi
    if [[ -z "$use_preset" ]]; then
      use_preset=high_density
    fi
    [[ -n "$view3d_size" ]] || view3d_size=1080x960
    [[ -n "$view3d_pos" ]] || view3d_pos=0,0
    [[ -n "$view2d_size" ]] || view2d_size=1080x960
    [[ -n "$view2d_pos" ]] || view2d_pos=0,960
  fi

  cmd_args=(
    "$script_path"
    --serial "$SERIAL"
    --device "$device"
    --model "$BEST_PT"
    --imgsz "$IMGSZ"
    --conf "$CONF"
    --min-points "$MIN_POINTS"
    --min-depth "$MIN_DEPTH"
    --max-depth "$MAX_DEPTH"
    --point-stride "$stride"
    --scene-max-points "$max_points"
    --track-window "$TRACK_WINDOW"
    --track-min-hits "$TRACK_MIN_HITS"
    --track-max-misses "$TRACK_MAX_MISSES"
    --track-max-pos-std "$TRACK_MAX_POS_STD"
    --track-max-yaw-std "$TRACK_MAX_YAW_STD"
    --track-min-conf "$TRACK_MIN_CONF"
    --track-assoc-dist "$TRACK_ASSOC_DIST"
    --track-stable-enter-frames "$TRACK_STABLE_ENTER_FRAMES"
    --track-stable-hysteresis "$TRACK_STABLE_HYSTERESIS"
    --track-stable-conf-margin "$TRACK_STABLE_CONF_MARGIN"
    --non-symmetric-classes "$NON_SYMMETRIC_CLASSES"
    --symmetric-classes "$SYMMETRIC_CLASSES"
    --table-normal-every "$TABLE_NORMAL_EVERY"
  )

  if [[ -n "$WORKSPACE" ]]; then
    # Use --opt=value so argparse accepts values starting with '-' (negative coords).
    cmd_args+=("--workspace=$WORKSPACE")
  fi
  # --real: also publish stable detections to ROS so the viewer runs alongside RViz.
  # Needs a sourced ROS env (rclpy) and a hand-eye ee_T_cam.json (latest if not given).
  if is_true "$REAL"; then
    cmd_args+=(--real --topic "$ROS_TOPIC")
    if [[ -n "$EE_T_CAM" ]]; then
      cmd_args+=(--ee-t-cam "$EE_T_CAM")
    fi
    if [[ -n "$WORKSPACE_BASE" ]]; then
      # --opt=value so argparse accepts values starting with '-' (negative coords).
      cmd_args+=("--workspace-base=$WORKSPACE_BASE")
    fi
  fi
  if is_true "$use_filters"; then
    cmd_args+=(--depth-filters)
  fi
  if [[ -n "$use_preset" ]]; then
    cmd_args+=(--depth-preset "$use_preset")
  fi
  if [[ -n "$view3d_size" ]]; then
    cmd_args+=(--view3d-size "$view3d_size")
  fi
  if [[ -n "$view3d_pos" ]]; then
    cmd_args+=(--view3d-pos "$view3d_pos")
  fi

  if [[ "$no_display" == true ]]; then
    cmd_args+=(--no-display --frames 5)
  else
    cmd_args+=(--show-labels)
    if is_true "$RAW_MODE"; then
      cmd_args+=(--raw-mode)
    fi
    if is_true "$use_show_2d"; then
      [[ -n "$view2d_size" ]] || view2d_size=1080x960
      cmd_args+=(
        --show-2d
        --view2d-size "$view2d_size"
        --view2d-layout "$VIEW2D_LAYOUT"
        --view2d-order "$VIEW2D_ORDER"
      )
      if [[ -n "$view2d_pos" ]]; then
        cmd_args+=(--view2d-pos "$view2d_pos")
      fi
      if is_true "$VIEW2D_FULLSCREEN"; then
        cmd_args+=(--view2d-fullscreen)
      fi
      if nonzero_number "$VIEW2D_DEPTH_MIN"; then
        cmd_args+=(--view2d-depth-min "$VIEW2D_DEPTH_MIN")
      fi
      if nonzero_number "$VIEW2D_DEPTH_MAX"; then
        cmd_args+=(--view2d-depth-max "$VIEW2D_DEPTH_MAX")
      fi
    fi
  fi

  run_uv_in_dir "$DET" python "${cmd_args[@]}"
}

BEST_PT=''
case "$COMMAND" in
  val|predict|predict-val|tabletop|tabletop-headless|tabletop-cpu)
    BEST_PT="$(resolve_best_pt "$RUN")"
    printf 'Model: %s\n' "$BEST_PT"
    ;;
esac

case "$COMMAND" in
  help)
    show_help
    ;;
  sync)
    require_command "$UV_BIN"
    uv_sync
    ;;
  analyze)
    run_analyze all
    ;;
  analyze-labelme)
    run_analyze labelme
    ;;
  analyze-yolo)
    run_analyze yolo
    ;;
  analyze-compare)
    run_analyze compare
    ;;
  stage)
    SRC="$DS/dataset_capture/rgb"
    DST="$DS/dataset_prelabel/input_images"
    [[ -d "$SRC" ]] || die "Not found: $SRC (run 'just capture' first)"
    shopt -s nullglob
    imgs=("$SRC"/*.jpg "$SRC"/*.jpeg "$SRC"/*.png "$SRC"/*.bmp)
    shopt -u nullglob
    [[ ${#imgs[@]} -gt 0 ]] || die "No images in $SRC. Run 'just capture' first."
    mkdir -p "$DST"
    n=0
    skipped=0
    for img in "${imgs[@]}"; do
      base="$(basename "$img")"
      # Never overwrite: skip images already staged (idempotent).
      if [[ -e "$DST/$base" ]]; then
        skipped=$((skipped + 1))
        continue
      fi
      cp "$img" "$DST/"
      n=$((n + 1))
    done
    echo "Staged $n new image(s): $SRC -> $DST"
    [[ $skipped -gt 0 ]] && echo "Skipped $skipped already present in input_images/ (not overwritten)."
    echo "Next: just prelabel-yolo (or prelabel-sam3)"
    exit 0
    ;;
  train)
    require_command "$UV_BIN"
    ensure_dataset_yaml
    # Enable local MLflow logging if the optional 'tracking' extra is installed
    # (uv sync --extra tracking). No-op otherwise, so training never depends on it.
    if uv_run python -c "import mlflow" >/dev/null 2>&1; then
      export MLFLOW_TRACKING_URI="file://$RUNS_ROOT/mlflow"
      export MLFLOW_EXPERIMENT_NAME="tabletop-seg"
      uv_run yolo settings mlflow=True >/dev/null 2>&1 || true
      printf 'MLflow tracking ON -> %s (view: mlflow ui --backend-store-uri %s)\n' \
        "$RUNS_ROOT/mlflow" "$RUNS_ROOT/mlflow"
    else
      printf 'MLflow not installed (optional). Enable with: uv sync --extra tracking\n'
    fi
    run_uv_in_dir "$DET" yolo segment train \
      model=yolo26m-seg.pt \
      data="$DATA_YAML" \
      imgsz="$IMGSZ" \
      epochs=100 \
      device=0 \
      batch=16 \
      workers=4 \
      cache=True \
      patience=20 \
      project="$RUNS_ROOT/segment"
    ;;
  val)
    require_command "$UV_BIN"
    ensure_dataset_yaml
    run_uv_in_dir "$DET" yolo segment val \
      model="$BEST_PT" \
      data="$DATA_YAML" \
      imgsz="$IMGSZ" \
      device=0
    ;;
  compare-models)
    require_command "$UV_BIN"
    ensure_dataset_yaml
    uv_run python "$DS/compare_models.py" \
      --runs-root "$RUNS_ROOT/segment" \
      --data "$DATA_YAML" \
      --imgsz "$IMGSZ"
    exit 0
    ;;
  predict)
    require_command "$UV_BIN"
    src="$(resolve_existing_path "${SOURCE:-$DS/dataset_yolo/images/val}")"
    [[ -e "$src" ]] || die "Not found: $src (use --source with a folder or .jpg)"
    run_uv_in_dir "$DET" yolo segment predict \
      model="$BEST_PT" \
      source="$src" \
      imgsz="$IMGSZ" \
      device=0 \
      save=True
    printf '\nResults: check %s\n' "$DET/runs/segment/predict*"
    ;;
  predict-val)
    require_command "$UV_BIN"
    src="$DS/dataset_yolo/images/val"
    run_uv_in_dir "$DET" yolo segment predict \
      model="$BEST_PT" \
      source="$src" \
      imgsz="$IMGSZ" \
      device=0 \
      save=True
    printf '\nResults: check %s\n' "$DET/runs/segment/predict*"
    ;;
  labelme)
    require_command "$UV_BIN"
    MANUAL_DIR="$DS/dataset_labeled/manual_labelme"
    mkdir -p "$MANUAL_DIR"
    run_uv_in_dir "$DS" labelme "$MANUAL_DIR"
    ;;
  tabletop)
    run_tabletop 0 false
    ;;
  tabletop-headless)
    run_tabletop 0 true
    ;;
  tabletop-cpu)
    run_tabletop cpu false
    ;;
  *)
    die "Unrecognized command: $COMMAND"
    ;;
esac
