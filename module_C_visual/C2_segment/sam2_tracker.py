"""SAM2 face/upper-body mask propagation.

Runs SAM2 once per video → produces per-frame masks reused by:
  - ProPainter (background inpainting)
  - Diffusion (VACE mask / FantasyTalking compositing)
  - Compositing (soft mask blending)

Two mask types:
  upper_body_mask — covers head + shoulders + arms (for ProPainter / Mode 1 compositing)
  face_mask       — covers face only (for Mode 2 differential strength map)
"""

import os
import sys
import numpy as np
import cv2

# Optional SAM2 import — path set by caller or env
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None


def detect_face_bbox(frame: np.ndarray) -> np.ndarray:
    """Use InsightFace to detect face bbox on a single frame.

    Returns (4,) array [x1, y1, x2, y2].
    """
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(frame)
    if not faces:
        raise RuntimeError("No face detected")
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return np.array(best.bbox, dtype=np.float32)


def expand_to_upper_body(bbox: np.ndarray, frame_h: int, frame_w: int,
                         expand_up: float = 0.3,
                         expand_down: float = 1.2,
                         expand_side: float = 0.8) -> np.ndarray:
    """Expand face bbox to cover upper body (head + shoulders + arms).

    Args:
        bbox: [x1, y1, x2, y2] face bbox.
        expand_up/down/side: expansion ratios relative to face height/width.

    Returns:
        [x1, y1, x2, y2] upper body bbox, clamped to frame bounds.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2

    ub_x1 = cx - w * (0.5 + expand_side)
    ub_x2 = cx + w * (0.5 + expand_side)
    ub_y1 = y1 - h * expand_up
    ub_y2 = y2 + h * expand_down

    return np.array([
        max(0, ub_x1), max(0, ub_y1),
        min(frame_w, ub_x2), min(frame_h, ub_y2),
    ], dtype=np.float32)


class FaceTracker:
    """SAM2 video predictor wrapper for face/upper-body tracking."""

    def __init__(self, model_cfg: str = 'sam2_hiera_large.yaml',
                 checkpoint: str = None):
        if build_sam2_video_predictor is None:
            raise ImportError("sam2 not found. Clone and install SAM2 first.")
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint

    def track(self, video_dir: str, prompt_frame_idx: int = 0,
              prompt_type: str = 'auto_face',
              dilate_px: int = 10) -> dict[str, list[np.ndarray]]:
        """Track face and upper body across all frames.

        Args:
            video_dir: directory of extracted frames (sorted PNGs).
            prompt_frame_idx: frame index for auto-prompting.
            prompt_type: 'auto_face' uses InsightFace bbox as SAM2 prompt.
            dilate_px: mask dilation in pixels.

        Returns:
            dict with keys:
                'face_mask': list of (H, W) uint8 [0, 255]
                'upper_body_mask': list of (H, W) uint8 [0, 255]
        """
        frame_files = sorted(f for f in os.listdir(video_dir)
                             if f.lower().endswith(('.png', '.jpg')))
        prompt_frame = cv2.imread(os.path.join(video_dir, frame_files[prompt_frame_idx]))
        H, W = prompt_frame.shape[:2]

        face_bbox = detect_face_bbox(prompt_frame)
        ub_bbox = expand_to_upper_body(face_bbox, H, W)

        predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint)

        # Track face (obj_id=1) and upper body (obj_id=2)
        with predictor.init_state(video_path=video_dir) as state:
            predictor.add_new_prompts(state, frame_idx=prompt_frame_idx,
                                      box=face_bbox, obj_id=1)
            predictor.add_new_prompts(state, frame_idx=prompt_frame_idx,
                                      box=ub_bbox, obj_id=2)

            raw_masks = {}
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                raw_masks[frame_idx] = {}
                for i, oid in enumerate(obj_ids):
                    raw_masks[frame_idx][oid] = (mask_logits[i] > 0).cpu().numpy().astype(np.uint8)

        sorted_idx = sorted(raw_masks.keys())
        face_masks, ub_masks = [], []

        for idx in sorted_idx:
            fm = raw_masks[idx].get(1, np.zeros((H, W), dtype=np.uint8))
            um = raw_masks[idx].get(2, np.zeros((H, W), dtype=np.uint8))

            if dilate_px > 0:
                kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
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
        max(0, xs.min() - padding),
        max(0, ys.min() - padding),
        min(w, xs.max() + padding),
        min(h, ys.max() + padding),
    ], dtype=np.float32)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Input video or frames dir')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--prompt_type', default='auto_face')
    parser.add_argument('--dilate_px', type=int, default=10)
    args = parser.parse_args()

    # If input is a video file, extract frames first
    frames_dir = args.video
    if os.path.isfile(args.video):
        frames_dir = os.path.join(args.output_dir, '_frames')
        os.makedirs(frames_dir, exist_ok=True)
        os.system(f'ffmpeg -i "{args.video}" -vf fps=25 -q:v 2 '
                  f'"{frames_dir}/%06d.png" -y -hide_banner -loglevel error')

    tracker = FaceTracker()
    masks = tracker.track(frames_dir, prompt_type=args.prompt_type,
                          dilate_px=args.dilate_px)

    for key in ('face_mask', 'upper_body_mask'):
        out = os.path.join(args.output_dir, key)
        os.makedirs(out, exist_ok=True)
        for i, m in enumerate(masks[key]):
            cv2.imwrite(os.path.join(out, f'{i:06d}.png'), m)
        print(f"Saved {len(masks[key])} {key} to {out}")
