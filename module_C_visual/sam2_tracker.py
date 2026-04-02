"""SAM2 face + upper_body dual-mask tracking.

Runs SAM2 once per video → outputs per-frame masks as .npz:
  face_mask      (T, H, W) uint8  — face region only
  upper_body_mask(T, H, W) uint8  — head + shoulders + arms

Also selects the best frontal reference frame (ref_frame.png).

Usage:
    python module_C_visual/sam2_tracker.py \
        --video data/presenters/bi/raw/BI_001.mp4 \
        --output_dir data/presenters/bi/ \
        --sam2_checkpoint data/models/sam2_hiera_large.pt
"""

import os
import sys
import numpy as np
import cv2
import argparse

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None


# ── InsightFace helpers ──────────────────────────────────────

def detect_face_bbox(frame: np.ndarray) -> np.ndarray:
    """Detect largest face bbox [x1, y1, x2, y2] via InsightFace."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(frame)
    if not faces:
        raise RuntimeError("No face detected in frame")
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return np.array(best.bbox, dtype=np.float32)


def expand_to_upper_body(bbox: np.ndarray, H: int, W: int,
                         up: float = 0.3, down: float = 1.2,
                         side: float = 0.8) -> np.ndarray:
    """Expand face bbox to cover upper body."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    return np.array([
        max(0, cx - w * (0.5 + side)),
        max(0, y1 - h * up),
        min(W, cx + w * (0.5 + side)),
        min(H, y2 + h * down),
    ], dtype=np.float32)


def select_best_ref_frame(frames_dir: str, face_bboxes: list) -> int:
    """Select the frame with largest frontal face (proxy for best quality)."""
    best_idx, best_area = 0, 0
    for i, bbox in enumerate(face_bboxes):
        if bbox is None:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > best_area:
            best_area = area
            best_idx = i
    return best_idx


# ── SAM2 tracker ─────────────────────────────────────────────

class FaceTracker:
    def __init__(self, model_cfg: str = 'sam2_hiera_l.yaml',
                 checkpoint: str = None):
        if build_sam2_video_predictor is None:
            raise ImportError("sam2 not installed")
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint

    def track(self, frames_dir: str, prompt_frame_idx: int = 0,
              dilate_px: int = 12) -> dict[str, list[np.ndarray]]:
        """Track face and upper body across all frames.

        Returns dict with 'face_mask' and 'upper_body_mask' lists,
        each element (H, W) uint8 [0, 255].
        """
        files = sorted(f for f in os.listdir(frames_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        prompt_frame = cv2.imread(os.path.join(frames_dir, files[prompt_frame_idx]))
        H, W = prompt_frame.shape[:2]

        face_bbox = detect_face_bbox(prompt_frame)
        ub_bbox = expand_to_upper_body(face_bbox, H, W)

        predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8) if dilate_px > 0 else None

        state = predictor.init_state(video_path=frames_dir)

        predictor.add_new_points_or_box(
            state, frame_idx=prompt_frame_idx,
            obj_id=1, box=face_bbox,
        )
        predictor.add_new_points_or_box(
            state, frame_idx=prompt_frame_idx,
            obj_id=2, box=ub_bbox,
        )

        raw = {}
        for fidx, obj_ids, logits in predictor.propagate_in_video(state):
            raw[fidx] = {}
            for i, oid in enumerate(obj_ids):
                raw[fidx][oid] = (logits[i] > 0).cpu().numpy().squeeze().astype(np.uint8)

        face_masks, ub_masks = [], []
        for idx in sorted(raw.keys()):
            fm = raw[idx].get(1, np.zeros((H, W), np.uint8))
            um = raw[idx].get(2, np.zeros((H, W), np.uint8))
            if kernel is not None:
                fm = cv2.dilate(fm, kernel)
                um = cv2.dilate(um, kernel)
            face_masks.append(fm * 255)
            ub_masks.append(um * 255)

        return {'face_mask': face_masks, 'upper_body_mask': ub_masks}


def mask_to_bbox(mask: np.ndarray, padding: int = 0) -> np.ndarray:
    """Convert binary mask to bbox [x1, y1, x2, y2]."""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        raise ValueError("Empty mask")
    h, w = mask.shape[:2]
    return np.array([
        max(0, xs.min() - padding), max(0, ys.min() - padding),
        min(w, xs.max() + padding), min(h, ys.max() + padding),
    ], dtype=np.float32)


# ── CLI ──────────────────────────────────────────────────────

def extract_frames(video: str, out_dir: str, fps: int = 25):
    """Extract frames as JPEG (SAM2 requires '<frame_index>.jpg' format)."""
    os.makedirs(out_dir, exist_ok=True)
    os.system(f'ffmpeg -i "{video}" -vf fps={fps} -q:v 2 '
              f'"{out_dir}/%06d.jpg" -y -hide_banner -loglevel error')


def main():
    parser = argparse.ArgumentParser(description='SAM2 dual-mask tracking')
    parser.add_argument('--video', required=True, help='Input video')
    parser.add_argument('--output_dir', required=True, help='Presenter output dir')
    parser.add_argument('--sam2_checkpoint', default='data/models/sam2_hiera_large.pt')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--dilate_px', type=int, default=12)
    args = parser.parse_args()

    # Extract frames
    frames_dir = os.path.join(args.output_dir, '_frames_tmp')
    print(f"[sam2] Extracting frames at {args.fps}fps...")
    extract_frames(args.video, frames_dir, args.fps)

    # Track
    print("[sam2] Running SAM2 tracking...")
    tracker = FaceTracker(checkpoint=args.sam2_checkpoint)
    masks = tracker.track(frames_dir, dilate_px=args.dilate_px)

    # Save as npz (compact)
    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)

    face_arr = np.stack(masks['face_mask'])       # (T, H, W)
    ub_arr = np.stack(masks['upper_body_mask'])    # (T, H, W)
    npz_path = os.path.join(masks_dir, 'masks.npz')
    np.savez_compressed(npz_path, face_mask=face_arr, upper_body_mask=ub_arr)
    print(f"[sam2] Saved {len(masks['face_mask'])} frames → {npz_path}")

    # Select and save ref_frame
    files = sorted(os.listdir(frames_dir))
    # Use frame 0 as default; could be improved with face quality scoring
    ref_idx = 0
    ref_src = os.path.join(frames_dir, files[ref_idx])
    ref_dst = os.path.join(args.output_dir, 'ref_frame.png')
    import shutil
    shutil.copy2(ref_src, ref_dst)
    print(f"[sam2] Ref frame: {ref_dst}")

    # Cleanup temp frames
    import shutil as sh
    sh.rmtree(frames_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
