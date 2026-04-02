"""Offline presenter preprocessing — run once per presenter.

Usage:
    conda activate wan_audio
    python scripts/preprocess_presenter.py \
        --presenter   bi \
        --video_dir   data/presenters/bi/raw/ \
        --output_dir  data/presenters/bi/

Outputs:
    data/presenters/bi/masks/masks.npz   — SAM2 face + upper_body masks
    data/presenters/bi/ref_frame.png     — best frontal reference frame
    data/presenters/bi/clean_bg/         — ProPainter background (needs propainter env)

ProPainter step requires switching env:
    conda activate propainter
    python module_C_visual/propainter_wrapper.py \
        --video_dir  data/presenters/bi/raw/ \
        --mask_dir   data/presenters/bi/masks/ \
        --output_dir data/presenters/bi/clean_bg/
"""

import argparse
import os
import sys
import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description='Offline presenter preprocessing')
    parser.add_argument('--presenter', required=True, help='Presenter ID (e.g. bi)')
    parser.add_argument('--video_dir', required=True, help='Directory of raw videos')
    parser.add_argument('--output_dir', required=True, help='Presenter output directory')
    parser.add_argument('--sam2_checkpoint', default='data/models/sam2_hiera_large.pt')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--dilate_px', type=int, default=12)
    parser.add_argument('--video', default=None,
                        help='Process single video (default: first in video_dir)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find video to process
    if args.video:
        video_path = args.video
    else:
        videos = sorted(glob.glob(os.path.join(args.video_dir, '*.mp4')))
        if not videos:
            videos = sorted(glob.glob(os.path.join(args.video_dir, '*.MP4')))
        if not videos:
            print(f"[error] No videos found in {args.video_dir}")
            return
        video_path = videos[0]
        print(f"[preprocess] Using first video: {video_path}")

    # ── Step 1: SAM2 mask tracking ───────────────────────────
    print("\n[1/3] SAM2 mask tracking...")
    from module_C_visual.sam2_tracker import FaceTracker, extract_frames
    import numpy as np
    import cv2
    import shutil

    frames_dir = os.path.join(args.output_dir, '_frames_tmp')
    shutil.rmtree(frames_dir, ignore_errors=True)  # clean before extracting
    extract_frames(video_path, frames_dir, args.fps)

    frame_files = sorted(f for f in os.listdir(frames_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    print(f"  {len(frame_files)} frames extracted")

    try:
        tracker = FaceTracker(checkpoint=args.sam2_checkpoint)
        masks = tracker.track(frames_dir, dilate_px=args.dilate_px)

        # Save as npz
        masks_dir = os.path.join(args.output_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        face_arr = np.stack(masks['face_mask'])
        ub_arr = np.stack(masks['upper_body_mask'])
        npz_path = os.path.join(masks_dir, 'masks.npz')
        np.savez_compressed(npz_path, face_mask=face_arr, upper_body_mask=ub_arr)
        print(f"  Saved masks → {npz_path}")
        print(f"  face_mask: {face_arr.shape}, upper_body_mask: {ub_arr.shape}")
    except ImportError as e:
        print(f"  [skip] SAM2 not available: {e}")
        print(f"  Install sam2 or run manually.")

    # ── Step 2: Select reference frame ───────────────────────
    print("\n[2/3] Selecting reference frame...")
    # Pick the frame with the largest face detection as proxy for quality
    try:
        from module_C_visual.sam2_tracker import detect_face_bbox
        best_idx, best_area = 0, 0
        # Sample every 10th frame for speed
        sample_step = max(1, len(frame_files) // 30)
        for i in range(0, len(frame_files), sample_step):
            frame = cv2.imread(os.path.join(frames_dir, frame_files[i]))
            try:
                bbox = detect_face_bbox(frame)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > best_area:
                    best_area = area
                    best_idx = i
            except RuntimeError:
                continue

        ref_src = os.path.join(frames_dir, frame_files[best_idx])
        ref_dst = os.path.join(args.output_dir, 'ref_frame.png')
        shutil.copy2(ref_src, ref_dst)
        print(f"  Selected frame {best_idx} (area={best_area:.0f}) → {ref_dst}")
    except ImportError:
        # Fallback: use first frame
        ref_src = os.path.join(frames_dir, frame_files[0])
        ref_dst = os.path.join(args.output_dir, 'ref_frame.png')
        shutil.copy2(ref_src, ref_dst)
        print(f"  Fallback: frame 0 → {ref_dst}")

    # ── Step 3: ProPainter (print instructions) ──────────────
    print("\n[3/3] ProPainter background inpainting...")
    clean_bg_dir = os.path.join(args.output_dir, 'clean_bg')
    print(f"  ProPainter requires its own conda env. Run:")
    print(f"    conda activate propainter")
    print(f"    python module_C_visual/propainter_wrapper.py \\")
    print(f"        --video_dir  \"{video_path}\" \\")
    print(f"        --mask_dir   \"{os.path.join(args.output_dir, 'masks')}\" \\")
    print(f"        --output_dir \"{clean_bg_dir}\"")

    # Cleanup temp frames
    shutil.rmtree(frames_dir, ignore_errors=True)

    print(f"\n✓ Preprocessing done for presenter '{args.presenter}'")
    print(f"  Output: {args.output_dir}")


if __name__ == '__main__':
    main()
