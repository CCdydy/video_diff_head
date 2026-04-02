"""ProPainter background inpainting + boundary seam repair.

1. Background inpainting: orig_frames + upper_body_mask(膨胀12px) → clean_bg
2. Boundary seam repair: composited_frames + seam_mask(20px) → repaired

Usage:
    conda activate propainter
    python module_C_visual/propainter_wrapper.py \
        --video_dir  data/presenters/bi/raw/ \
        --mask_dir   data/presenters/bi/masks/ \
        --output_dir data/presenters/bi/clean_bg/
"""

import os
import argparse


PROPAINTER_REPO = os.environ.get(
    'PROPAINTER_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'third_party', 'ProPainter')
)


def inpaint_video(
    video_path: str,
    mask_dir: str,
    output_dir: str,
    subvideo_length: int = 80,
    neighbor_length: int = 10,
    ref_stride: int = 10,
    fp16: bool = True,
) -> str:
    """Run ProPainter CLI on a single video.

    Args:
        video_path: input video file or frames directory.
        mask_dir: directory of mask PNGs (255 = inpaint region).
        output_dir: where to write inpainted frames.

    Returns output_dir.
    """
    script = os.path.join(PROPAINTER_REPO, 'inference_propainter.py')
    if not os.path.isfile(script):
        raise FileNotFoundError(
            f"ProPainter not found at {PROPAINTER_REPO}.\n"
            f"Set PROPAINTER_DIR env or clone:\n"
            f"  git clone https://github.com/sczhou/ProPainter.git {PROPAINTER_REPO}"
        )

    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        f'python "{script}"'
        f' --video "{video_path}"'
        f' --mask "{mask_dir}"'
        f' --output "{output_dir}"'
        f' --subvideo_length {subvideo_length}'
        f' --neighbor_length {neighbor_length}'
        f' --ref_stride {ref_stride}'
    )
    if fp16:
        cmd += ' --fp16'

    print(f"[propainter] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"ProPainter failed (exit {ret})")
    return output_dir


def inpaint_boundary_seam(
    composited_dir: str,
    seam_mask_dir: str,
    output_dir: str,
    subvideo_length: int = 80,
) -> str:
    """Post-compositing boundary seam repair using ProPainter."""
    return inpaint_video(composited_dir, seam_mask_dir, output_dir,
                         subvideo_length=subvideo_length)


def main():
    parser = argparse.ArgumentParser(description='ProPainter background inpainting')
    parser.add_argument('--video_dir', required=True,
                        help='Video file or frames directory')
    parser.add_argument('--mask_dir', required=True,
                        help='Mask directory (npz or PNGs)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--subvideo_length', type=int, default=80)
    parser.add_argument('--mask_key', default='upper_body_mask',
                        help='Key in masks.npz to use')
    args = parser.parse_args()

    import numpy as np
    import cv2

    # If mask_dir points to an npz, extract PNGs for ProPainter
    npz_path = os.path.join(args.mask_dir, 'masks.npz')
    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        masks = data[args.mask_key]  # (T, H, W)
        png_dir = os.path.join(args.output_dir, '_mask_pngs')
        os.makedirs(png_dir, exist_ok=True)
        for i in range(masks.shape[0]):
            cv2.imwrite(os.path.join(png_dir, f'{i:06d}.png'), masks[i])
        mask_dir = png_dir
        print(f"[propainter] Extracted {masks.shape[0]} masks from npz")
    else:
        mask_dir = args.mask_dir

    inpaint_video(args.video_dir, mask_dir, args.output_dir,
                  subvideo_length=args.subvideo_length)
    print(f"[propainter] Done → {args.output_dir}")


if __name__ == '__main__':
    main()
