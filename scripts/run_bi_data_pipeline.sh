#!/usr/bin/env bash
# run_bi_data_pipeline.sh — End-to-end data pipeline (Steps 1–4 + 4b)
#
# Usage:  bash scripts/run_bi_data_pipeline.sh [--from STEP]
#
# Steps:
#   1  Face scanning
#   2  Video selection + clip extraction
#   3  VHAP FLAME tracking
#   4  Build LMDB dataset
#   4b Extract reference frames (for Modes B & C)
# ──────────────────────────────────────────────────────────────

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

FROM_STEP="${1:-1}"
if [[ "$FROM_STEP" == "--from" ]]; then
    FROM_STEP="${2:-1}"
fi

log() { echo -e "\n\033[1;36m═══ Step $1: $2 ═══\033[0m\n"; }

# ── Step 1: Face Scanning ───────────────────────────────────

if [[ "$FROM_STEP" -le 1 ]]; then
    log 1 "Face Scanning"
    conda run -n vhap --no-banner python scripts/batch_face_scan.py \
        --video_dir data/presenters/bi \
        --output_dir runs/face_scan_bi \
        --sample_fps 0.5
fi

# ── Step 2: Select Videos + Extract Clips ───────────────────

if [[ "$FROM_STEP" -le 2 ]]; then
    log 2 "Video Selection + Clip Extraction"
    conda run -n vhap --no-banner python scripts/prepare_bi_clips.py \
        --scan_results runs/face_scan_bi/scan_results.json \
        --video_dir data/presenters/bi \
        --output_dir data/bi_training \
        --min_face_pct 30 --target_hours 10
fi

# ── Step 3: VHAP FLAME Tracking ────────────────────────────

if [[ "$FROM_STEP" -le 3 ]]; then
    log 3 "VHAP FLAME Tracking"
    conda run -n vhap --no-banner python scripts/batch_vhap.py \
        --manifest data/bi_training/clip_manifest.json \
        --vhap_dir module_A_offline/A1_flame_extraction/VHAP \
        --output_dir data/bi_training/flame_params \
        --num_epochs 1
fi

# ── Step 4: Build LMDB ─────────────────────────────────────

if [[ "$FROM_STEP" -le 4 ]]; then
    log 4 "Build LMDB Dataset"
    conda run -n vhap --no-banner python scripts/build_bi_lmdb.py \
        --manifest data/bi_training/clip_manifest.json \
        --flame_dir data/bi_training/flame_params \
        --output_dir data/bi_training/lmdb
fi

# ── Step 4b: Extract Reference Frames ──────────────────────

log "4b" "Extract Reference Frames"
conda run -n vhap --no-banner python scripts/extract_ref_frames.py \
    --manifest data/bi_training/clip_manifest.json \
    --output_dir data/bi_training/ref_frames \
    --strategy frontal_max_face

echo -e "\n\033[1;32m✓ Data pipeline complete.\033[0m"
