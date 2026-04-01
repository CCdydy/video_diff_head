"""Color harmonization via histogram matching.

Corrects color/lighting differences between generated and original regions
at the compositing boundary. Matches the generated region's color
distribution to the original frame's nearby pixels.
"""

import numpy as np
import cv2


def histogram_match_region(
    src: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Match color histogram of src (generated) to ref (original) in mask region.

    Works in LAB color space for perceptually uniform matching.

    Args:
        src: (H, W, 3) BGR generated frame.
        ref: (H, W, 3) BGR original frame.
        mask: (H, W) uint8 mask, 255 = region to harmonize.

    Returns:
        (H, W, 3) BGR color-corrected frame.
    """
    if mask.sum() == 0:
        return src

    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float64)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float64)

    mask_bool = mask > 127
    # Use a band around the mask boundary as reference for color stats
    kernel = np.ones((31, 31), np.uint8)
    dilated = cv2.dilate(mask, kernel)
    band = (dilated > 127) & (~mask_bool)

    if band.sum() < 100:
        # Not enough reference pixels — skip
        return src

    result = src_lab.copy()
    for c in range(3):
        src_vals = src_lab[:, :, c][mask_bool]
        ref_vals = ref_lab[:, :, c][band]

        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        ref_mean, ref_std = ref_vals.mean(), ref_vals.std() + 1e-6

        # Normalize src to match ref distribution
        normalized = (src_lab[:, :, c] - src_mean) * (ref_std / src_std) + ref_mean
        result[:, :, c] = np.where(mask_bool, normalized, src_lab[:, :, c])

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', required=True)
    parser.add_argument('--orig_dir', required=True)
    parser.add_argument('--mask_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    gen_files = sorted(os.listdir(args.gen_dir))
    orig_files = sorted(os.listdir(args.orig_dir))
    mask_files = sorted(os.listdir(args.mask_dir))
    os.makedirs(args.output, exist_ok=True)

    n = min(len(gen_files), len(orig_files), len(mask_files))
    for i in range(n):
        gen = cv2.imread(os.path.join(args.gen_dir, gen_files[i]))
        orig = cv2.imread(os.path.join(args.orig_dir, orig_files[i]))
        mask = cv2.imread(os.path.join(args.mask_dir, mask_files[i]),
                          cv2.IMREAD_GRAYSCALE)
        out = histogram_match_region(gen, orig, mask)
        cv2.imwrite(os.path.join(args.output, f'{i:06d}.png'), out)

    print(f"Color harmonized {n} frames → {args.output}")
