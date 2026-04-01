"""Poisson blending at mask boundary (Mode 1 only).

cv2.seamlessClone handles color/lighting mismatch at the compositing seam.
Mode 2 should not need this — VACE preserves original pixels outside mask.
"""

import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from module_C_visual.C2_segment.mask_utils import mask_centroid


def poisson_blend(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
    flags: int = cv2.NORMAL_CLONE,
) -> np.ndarray:
    """Apply Poisson blending (seamlessClone) at mask boundary.

    Args:
        src: (H, W, 3) source (generated face/upper body).
        dst: (H, W, 3) destination (composited frame).
        mask: (H, W) uint8 binary mask, 255 = blend region.
        flags: cv2.NORMAL_CLONE or cv2.MIXED_CLONE.

    Returns:
        (H, W, 3) blended frame.
    """
    if mask.sum() == 0:
        return dst

    center = mask_centroid(mask)
    blend_mask = (mask > 127).astype(np.uint8) * 255

    try:
        return cv2.seamlessClone(src, dst, blend_mask, center, flags)
    except cv2.error:
        # Fails if mask touches image border or is too small
        return dst


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True, help='Generated frames')
    parser.add_argument('--dst_dir', required=True, help='Composited frames')
    parser.add_argument('--mask_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    src_files = sorted(os.listdir(args.src_dir))
    dst_files = sorted(os.listdir(args.dst_dir))
    mask_files = sorted(os.listdir(args.mask_dir))
    os.makedirs(args.output, exist_ok=True)

    n = min(len(src_files), len(dst_files), len(mask_files))
    for i in range(n):
        src = cv2.imread(os.path.join(args.src_dir, src_files[i]))
        dst = cv2.imread(os.path.join(args.dst_dir, dst_files[i]))
        mask = cv2.imread(os.path.join(args.mask_dir, mask_files[i]),
                          cv2.IMREAD_GRAYSCALE)
        out = poisson_blend(src, dst, mask)
        cv2.imwrite(os.path.join(args.output, f'{i:06d}.png'), out)

    print(f"Poisson blended {n} frames → {args.output}")
