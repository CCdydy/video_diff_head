"""ProPainter wrapper for background inpainting + boundary seam repair.

Two use cases:
  1. Background inpainting: remove upper body, fill with background texture.
     Input: frames + upper_body_mask → clean_bg frames.
  2. Boundary seam repair: after compositing, fix the ~20px seam at mask edge.
     Input: composited frames + boundary_seam_mask → repaired frames.

ProPainter processes fixed-length clips (default 80 frames) with overlap.
"""

import os
import sys
import numpy as np
import cv2

PROPAINTER_REPO = os.path.join(os.path.dirname(__file__), 'ProPainter')
if os.path.isdir(PROPAINTER_REPO):
    sys.path.insert(0, PROPAINTER_REPO)


def inpaint_video_cli(
    video_path: str,
    mask_dir: str,
    output_dir: str,
    subvideo_length: int = 80,
    neighbor_length: int = 10,
    ref_stride: int = 10,
    fp16: bool = True,
) -> str:
    """Run ProPainter via its CLI script (simplest integration).

    Calls ProPainter/inference_propainter.py directly.
    Returns path to output directory.
    """
    script = os.path.join(PROPAINTER_REPO, 'inference_propainter.py')
    if not os.path.isfile(script):
        raise FileNotFoundError(
            f"ProPainter not found at {PROPAINTER_REPO}. Clone it first:\n"
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

    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"ProPainter failed with exit code {ret}")
    return output_dir


def inpaint_boundary_seam(
    composited_dir: str,
    seam_mask_dir: str,
    output_dir: str,
    subvideo_length: int = 80,
) -> str:
    """Repair boundary seam artifacts after compositing.

    Uses ProPainter with the narrow seam mask (dilate - erode, ~20px wide).
    Only touches the seam region, interior and exterior untouched.
    """
    return inpaint_video_cli(
        video_path=composited_dir,
        mask_dir=seam_mask_dir,
        output_dir=output_dir,
        subvideo_length=subvideo_length,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ProPainter background inpainting')
    parser.add_argument('--video', required=True, help='Video path or frames dir')
    parser.add_argument('--mask', required=True, help='Mask directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--subvideo_length', type=int, default=80)
    args = parser.parse_args()

    inpaint_video_cli(args.video, args.mask, args.output,
                      subvideo_length=args.subvideo_length)
    print(f"Done. Output: {args.output}")
