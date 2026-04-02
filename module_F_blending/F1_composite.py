"""Compositing: blend generated upper body back onto original frame.

Mode 1: soft mask blending (gen_frame over orig_frame)
  result = orig * (1 - soft_mask) + gen * soft_mask

Mode 2: skip — VACE already preserves mask-exterior pixels.
  Only light feathering at boundary if needed.
"""

import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from module_C_visual.mask_utils import soft_mask


def composite_mode1(
    orig_frame: np.ndarray,
    gen_frame: np.ndarray,
    upper_body_mask: np.ndarray,
    blur_radius: int = 16,
) -> np.ndarray:
    """Mode 1 compositing: blend FantasyTalking output onto original frame.

    Args:
        orig_frame: (H, W, 3) BGR original frame.
        gen_frame: (H, W, 3) BGR generated upper body (may be different resolution).
        upper_body_mask: (H, W) uint8 mask, 255 = generated region.
        blur_radius: Gaussian blur radius for soft mask edge.

    Returns:
        (H, W, 3) BGR composited frame.
    """
    H, W = orig_frame.shape[:2]

    # Resize gen_frame to match original if needed
    if gen_frame.shape[:2] != (H, W):
        gen_frame = cv2.resize(gen_frame, (W, H), interpolation=cv2.INTER_LANCZOS4)

    weight = soft_mask(upper_body_mask, blur_radius)
    weight_3ch = weight[:, :, np.newaxis]

    result = (orig_frame.astype(np.float64) * (1.0 - weight_3ch) +
              gen_frame.astype(np.float64) * weight_3ch)

    return np.clip(result, 0, 255).astype(np.uint8)


def composite_mode2(
    vace_frame: np.ndarray,
    orig_frame: np.ndarray,
    upper_body_mask: np.ndarray,
    feather_px: int = 4,
) -> np.ndarray:
    """Mode 2 compositing: VACE output already preserves background.

    Only apply light feathering at mask boundary for any VAE decode artifacts.
    Pixels far from boundary are taken directly from VACE output.
    """
    if feather_px <= 0:
        return vace_frame

    # Build narrow boundary feather
    kernel = np.ones((feather_px * 2 + 1, feather_px * 2 + 1), np.uint8)
    dilated = cv2.dilate(upper_body_mask, kernel)
    eroded = cv2.erode(upper_body_mask, kernel)
    boundary = cv2.subtract(dilated, eroded)

    # Soft blend only at boundary
    blend = soft_mask(boundary, feather_px)
    blend_3ch = blend[:, :, np.newaxis]

    # At boundary: average VACE and original
    result = vace_frame.copy().astype(np.float64)
    boundary_region = boundary > 0
    if boundary_region.any():
        result[boundary_region] = (
            vace_frame[boundary_region].astype(np.float64) * 0.5 +
            orig_frame[boundary_region].astype(np.float64) * 0.5
        )

    return np.clip(result, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_bg', help='Clean background frames dir (unused in new arch)')
    parser.add_argument('--gen_frames', required=True, help='Generated frames dir')
    parser.add_argument('--orig_frames', required=True, help='Original frames dir')
    parser.add_argument('--mask', required=True, help='Upper body mask dir')
    parser.add_argument('--mode', type=int, choices=[1, 2], default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    gen_files = sorted(os.listdir(args.gen_frames))
    orig_files = sorted(os.listdir(args.orig_frames))
    mask_files = sorted(os.listdir(args.mask))
    os.makedirs(args.output, exist_ok=True)

    n = min(len(gen_files), len(orig_files), len(mask_files))
    for i in range(n):
        gen = cv2.imread(os.path.join(args.gen_frames, gen_files[i]))
        orig = cv2.imread(os.path.join(args.orig_frames, orig_files[i]))
        mask = cv2.imread(os.path.join(args.mask, mask_files[i]),
                          cv2.IMREAD_GRAYSCALE)

        if args.mode == 1:
            out = composite_mode1(orig, gen, mask)
        else:
            out = composite_mode2(gen, orig, mask)

        cv2.imwrite(os.path.join(args.output, f'{i:06d}.png'), out)

    print(f"Composited {n} frames (mode {args.mode}) → {args.output}")
