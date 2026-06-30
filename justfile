# List all available recipes (default when running bare `just`).
default:
    just --justfile "{{justfile()}}" --list

# Print the full CLI help with options and examples.
help:
    bash "{{justfile_directory()}}/scripts/sense.sh" help

# Install/update the Python environment from pyproject + uv.lock.
sync *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" sync {{args}}

# Dataset stats for both LabelMe and YOLO (counts, classes).
analyze *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" analyze {{args}}

# Dataset stats from the raw LabelMe JSON labels only.
analyze-labelme *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" analyze-labelme {{args}}

# Dataset stats from the converted YOLO-seg labels only.
analyze-yolo *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" analyze-yolo {{args}}

# Compare LabelMe vs YOLO labels to catch conversion drift.
analyze-compare *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" analyze-compare {{args}}

# Capture RGB (default) or RGBD (--rgbd). Presenter: -> object / <- background; either arrow saves in RGB mode.
capture *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" capture {{args}}

# Copy new captures DS/dataset_capture/rgb/ -> input_images/ for autolabeling (no overwrite).
stage *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" stage {{args}}

# Visualize a depth.npy map (e.g. just view-depth --depth-file path/depth.npy [--rgb]).
view-depth *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" view-depth {{args}}

# YOLO pre-label rgbd/ captures -> to_review_rgbd_yolo/ (skips already-labeled; --overwrite to redo).
prelabel-rgbd *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" prelabel-rgbd {{args}}

# Open LabelMe on the RGBD prelabels to fix masks before they feed the synth pipeline.
review-rgbd *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" review-rgbd {{args}}

# Write reviewed RGBD labels back as rgb.json inside each rgbd/<id>/ (input of synth-render).
promote-rgbd *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" promote-rgbd {{args}}

# Lift LabelMe masks on RGBD captures to 3D points (Phase 1B test).
lift-rgbd *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" lift-rgbd {{args}}

# View lift_*.ply aligned to the RealSense capture orientation.
view-lift *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" view-lift {{args}}

# Phase 1B (init + render, 2 steps). --backend REQUIRED: points->synth_render_point/, gsplat->synth_render_3dgs/.
synth-render *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" synth-render {{args}}

# Export synth_render_<backend> views -> LabelMe pairs in to_review_synth_<backend>/. --backend REQUIRED.
export-synth *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" export-synth {{args}}

# Open LabelMe to QC synthetic prelabels (object-on-black). QC only, not promoted. --backend REQUIRED.
review-synth *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" review-synth {{args}}

# Composite object views over a real background -> composited_views_<backend>/. --backend REQUIRED.
composite-synth *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" composite-synth {{args}}

# Export composited_views_<backend> -> LabelMe pairs in to_review_composite_<backend>/. --backend REQUIRED.
export-composite *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" export-composite {{args}}

# Open LabelMe on composited prelabels for review. --backend REQUIRED (points|gsplat).
review-composite *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" review-composite {{args}}

# Promote reviewed composited labels -> dataset_labeled_<backend>/composited_labelme/. --backend REQUIRED.
promote-composite *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" promote-composite {{args}}

# Convert LabelMe -> YOLO-seg. --data REQUIRED: real|point|3dgs|all (all = real+point+3dgs).
convert *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" convert {{args}}

# Train the YOLO segmentation model on the YOLO dataset.
train *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" train {{args}}

# Validate the latest trained model on the val split.
val *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" val {{args}}

# Re-validate every trained best.pt on the current val split and print a ranked table.
compare-models *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" compare-models {{args}}

# Run inference with the latest model (default: val images).
predict *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" predict {{args}}

# Run inference specifically on the val image folder.
predict-val *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" predict-val {{args}}

# Open LabelMe on dataset_labeled/manual_labelme/ for first-pass manual labeling.
labelme *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" labelme {{args}}

# Realtime 3D detection (RealSense + YOLO + Open3D viewer).
tabletop *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" tabletop {{args}}

# Realtime 3D detection without GUI, prints per-frame JSON.
tabletop-headless *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" tabletop-headless {{args}}

# Realtime 3D detection forced onto CPU inference.
tabletop-cpu *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" tabletop-cpu {{args}}

# --- YOLO pre-label pipeline (default backend; classes from DS/yolo_classes.yaml) ---
# Step 1: YOLO inference on input_images/ -> prelabels_yolo/ (extra flags pass through).
prelabel-yolo *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" prelabel-yolo {{args}}

# Step 2: YOLO prelabels_yolo/ -> LabelMe pairs in to_review_yolo/ (keeps existing reviews; --overwrite to force).
export-yolo *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" export-yolo {{args}}

# Step 3: open LabelMe on to_review_yolo/ to fix the YOLO prelabels.
review-yolo *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" review-yolo {{args}}

# Step 4: copy reviewed pairs -> dataset_labeled/reviewed_yolo_labelme/ (joins the training pool). [--overwrite] [--capture ID].
promote-yolo *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" promote-yolo {{args}}

# --- SAM3 pre-label pipeline (text-prompt backend; needs sam3.pt + 'uv sync --extra sam3') ---
# Step 1: SAM3 inference on input_images/ -> prelabels_sam3/ (extra flags pass through).
prelabel-sam3 *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" prelabel-sam3 {{args}}

# Step 2: SAM3 prelabels_sam3/ -> LabelMe pairs in to_review_sam3/ (keeps existing reviews; --overwrite to force).
export-sam3 *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" export-sam3 {{args}}

# Step 3: open LabelMe on to_review_sam3/ to inspect the SAM3 prelabels.
review-sam3 *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" review-sam3 {{args}}

# Step 4: copy reviewed pairs -> dataset_labeled/reviewed_sam3_labelme/ (joins the training pool). [--overwrite] [--capture ID].
promote-sam3 *args:
    bash "{{justfile_directory()}}/scripts/sense.sh" promote-sam3 {{args}}

# Escape hatch: forward raw args/options straight to sense.sh.
sense +args:
    bash "{{justfile_directory()}}/scripts/sense.sh" {{args}}
