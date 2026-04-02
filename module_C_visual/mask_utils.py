"""Mask utilities: dilate, soft_mask, boundary_seam, strength_map.

Supports Mode 1 (lip-only) and Mode 2 (upper-body) strength maps.
"""

import numpy as np
import cv2


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    kernel = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    return cv2.dilate(mask, kernel)


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    kernel = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    return cv2.erode(mask, kernel)


def soft_mask(mask: np.ndarray, blur_radius: int = 16) -> np.ndarray:
    """Hard mask → soft Gaussian falloff. Returns (H,W) float32 [0,1]."""
    ksize = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), blur_radius)
    return np.clip(blurred / 255.0, 0.0, 1.0)


def boundary_seam_mask(mask: np.ndarray, width: int = 20) -> np.ndarray:
    """Narrow seam mask at mask boundary for ProPainter edge repair.
    Returns (H,W) uint8, 255 = seam."""
    half = width // 2
    return cv2.subtract(dilate_mask(mask, half), erode_mask(mask, half))


def mask_centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return (w // 2, h // 2)
    return (int(xs.mean()), int(ys.mean()))


# ── Strength maps for Differential Diffusion ─────────────────

def build_strength_map_mode2(
    face_mask: np.ndarray,
    upper_body_mask: np.ndarray,
    lip_strength: float = 0.55,
    face_strength: float = 0.35,
    body_strength: float = 0.15,
    lips_mask: np.ndarray = None,
) -> np.ndarray:
    """Mode 2 strength map: upper body complete rebuild.

    唇 0.55 / 脸 0.35 / 上半身 0.15 / 背景 0.00

    Args:
        face_mask: (H,W) uint8, 255=face.
        upper_body_mask: (H,W) uint8, 255=upper body.
        lips_mask: optional (H,W) uint8, 255=lips. If None, approximate
                   from lower 40% of face region.

    Returns:
        (H,W) float32 in [0, 1].
    """
    H, W = face_mask.shape[:2]
    smap = np.zeros((H, W), dtype=np.float32)

    # Shoulder/neck = upper_body minus face
    shoulder = (upper_body_mask > 127) & (face_mask <= 127)
    smap[shoulder] = body_strength

    # Face (non-lip)
    face = face_mask > 127
    smap[face] = face_strength

    # Lips
    if lips_mask is not None:
        lips = lips_mask > 127
    else:
        ys, xs = np.where(face)
        if len(ys) > 0:
            y_top, y_bot = ys.min(), ys.max()
            lip_start = y_top + int((y_bot - y_top) * 0.6)
            lips = face.copy()
            lips[:lip_start, :] = False
        else:
            lips = np.zeros((H, W), dtype=bool)
    smap[lips] = lip_strength

    return smap


def build_strength_map_mode1(
    face_mask: np.ndarray,
    lip_strength: float = 0.55,
    face_strength: float = 0.20,
    lips_mask: np.ndarray = None,
) -> np.ndarray:
    """Mode 1 strength map: lip-only local replacement.

    唇 0.55 / 脸其余 0.20 / 其他 0.00
    """
    H, W = face_mask.shape[:2]
    smap = np.zeros((H, W), dtype=np.float32)

    face = face_mask > 127
    smap[face] = face_strength

    if lips_mask is not None:
        lips = lips_mask > 127
    else:
        ys, xs = np.where(face)
        if len(ys) > 0:
            y_top, y_bot = ys.min(), ys.max()
            lip_start = y_top + int((y_bot - y_top) * 0.6)
            lips = face.copy()
            lips[:lip_start, :] = False
        else:
            lips = np.zeros((H, W), dtype=bool)
    smap[lips] = lip_strength

    return smap


def build_strength_map(
    face_mask: np.ndarray,
    upper_body_mask: np.ndarray,
    mode: int = 2,
    lip_strength: float = 0.55,
    face_strength: float = 0.35,
    body_strength: float = 0.15,
) -> np.ndarray:
    """Dispatch to Mode 1 or Mode 2 strength map builder."""
    if mode == 1:
        return build_strength_map_mode1(face_mask, lip_strength,
                                        face_strength=0.20)
    else:
        return build_strength_map_mode2(face_mask, upper_body_mask,
                                        lip_strength, face_strength,
                                        body_strength)
