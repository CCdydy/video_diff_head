"""Mask utilities: dilate, soft mask, boundary seam mask.

Used across compositing, ProPainter boundary repair, and
differential strength map generation.
"""

import numpy as np
import cv2


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate binary mask by px pixels."""
    if px <= 0:
        return mask
    kernel = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    return cv2.dilate(mask, kernel)


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode binary mask by px pixels."""
    if px <= 0:
        return mask
    kernel = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    return cv2.erode(mask, kernel)


def soft_mask(mask: np.ndarray, blur_radius: int = 16) -> np.ndarray:
    """Convert hard mask to soft Gaussian falloff at edges.

    Args:
        mask: (H, W) uint8, 255 = region.
        blur_radius: Gaussian kernel radius.

    Returns:
        (H, W) float32 in [0, 1].
    """
    ksize = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), blur_radius)
    return np.clip(blurred / 255.0, 0.0, 1.0)


def boundary_seam_mask(mask: np.ndarray, width: int = 20) -> np.ndarray:
    """Generate a narrow boundary seam mask for ProPainter edge repair.

    boundary = dilate(mask, width//2) - erode(mask, width//2)

    Only repairs the edge seam, leaves mask interior and exterior untouched.

    Args:
        mask: (H, W) uint8 binary mask.
        width: total seam width in pixels.

    Returns:
        (H, W) uint8 boundary mask, 255 = seam region.
    """
    half = width // 2
    dilated = dilate_mask(mask, half)
    eroded = erode_mask(mask, half)
    seam = cv2.subtract(dilated, eroded)
    return seam


def mask_centroid(mask: np.ndarray) -> tuple[int, int]:
    """Compute (cx, cy) centroid of mask region."""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return (w // 2, h // 2)
    return (int(xs.mean()), int(ys.mean()))


def build_region_strength_map(
    face_mask: np.ndarray,
    upper_body_mask: np.ndarray,
    lips_mask: np.ndarray = None,
    strength_lips: float = 0.50,
    strength_face: float = 0.35,
    strength_shoulder: float = 0.15,
) -> np.ndarray:
    """Build per-pixel strength map for Differential Diffusion (Mode 2).

    Args:
        face_mask: (H, W) uint8, 255 = face region.
        upper_body_mask: (H, W) uint8, 255 = upper body.
        lips_mask: (H, W) uint8 optional, 255 = lips. If None, uses lower
                   third of face_mask as rough lips region.
        strength_lips/face/shoulder: per-region editing strength [0, 1].

    Returns:
        (H, W) float32 strength map in [0, 1].
        background = 0.0 (no edit), lips = highest edit.
    """
    H, W = face_mask.shape[:2]
    smap = np.zeros((H, W), dtype=np.float32)

    # Shoulder/neck = upper_body minus face
    shoulder_region = (upper_body_mask > 127) & (face_mask <= 127)
    smap[shoulder_region] = strength_shoulder

    # Face (non-lips)
    face_region = face_mask > 127
    smap[face_region] = strength_face

    # Lips
    if lips_mask is not None:
        lips_region = lips_mask > 127
    else:
        # Approximate: lower 1/3 of face bbox
        ys, xs = np.where(face_region)
        if len(ys) > 0:
            y_top, y_bot = ys.min(), ys.max()
            lip_start = y_top + int((y_bot - y_top) * 0.6)
            lips_region = face_region.copy()
            lips_region[:lip_start, :] = False
        else:
            lips_region = np.zeros((H, W), dtype=bool)
    smap[lips_region] = strength_lips

    return smap
