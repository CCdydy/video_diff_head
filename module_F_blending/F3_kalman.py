"""Kalman filter landmark temporal smoothing (Mode 1 only).

Suppresses jitter from FantasyTalking's per-chunk generation by
smoothing face landmark trajectories and applying corrective warp.

Mode 2 (VACE) inherently has temporal consistency from joint 3D DiT
denoising — this module is not needed.

Parameters from README: sigma_process=0.01, sigma_measurement=0.1
"""

import numpy as np
import cv2


class LandmarkKalmanFilter:
    """Per-landmark 2D Kalman filter. State: [x, y, vx, vy]."""

    def __init__(self, n_landmarks: int,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        self.filters = []
        for _ in range(n_landmarks):
            kf = cv2.KalmanFilter(4, 2)
            kf.transitionMatrix = np.eye(4, dtype=np.float32)
            kf.transitionMatrix[0, 2] = 1.0
            kf.transitionMatrix[1, 3] = 1.0
            kf.measurementMatrix = np.zeros((2, 4), dtype=np.float32)
            kf.measurementMatrix[0, 0] = 1.0
            kf.measurementMatrix[1, 1] = 1.0
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
            kf.errorCovPost = np.eye(4, dtype=np.float32)
            self.filters.append(kf)

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        """Predict + correct. Returns smoothed (N, 2) landmarks."""
        smoothed = np.zeros_like(landmarks)
        for i, kf in enumerate(self.filters):
            kf.predict()
            kf.correct(landmarks[i].astype(np.float32).reshape(2, 1))
            smoothed[i] = [kf.statePost[0, 0], kf.statePost[1, 0]]
        return smoothed


def detect_landmarks_mediapipe(frame: np.ndarray) -> np.ndarray | None:
    """Detect 468 face landmarks using MediaPipe Face Mesh.

    Returns (468, 2) float32 array or None.
    """
    try:
        import mediapipe as mp
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        H, W = frame.shape[:2]
        lm = result.multi_face_landmarks[0].landmark
        return np.array([[l.x * W, l.y * H] for l in lm], dtype=np.float32)
    except ImportError:
        return None


def kalman_smooth_frames(
    frames: list[np.ndarray],
    face_masks: list[np.ndarray],
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
) -> list[np.ndarray]:
    """Apply Kalman smoothing to face region across frames.

    Detects landmarks per frame, smooths trajectories, applies affine warp
    to correct jitter in the face region only.
    """
    kf = None
    result = []

    for frame, mask in zip(frames, face_masks):
        lm = detect_landmarks_mediapipe(frame)
        if lm is None:
            result.append(frame)
            continue

        if kf is None:
            kf = LandmarkKalmanFilter(len(lm), process_noise, measurement_noise)

        smoothed_lm = kf.update(lm)
        displacement = np.linalg.norm(smoothed_lm - lm, axis=1).mean()

        if displacement < 0.5:
            result.append(frame)
            continue

        # Affine warp using 3 stable points (left eye, right eye, nose)
        # MediaPipe indices: 33 (left eye), 263 (right eye), 1 (nose tip)
        idx = [33, 263, 1] if len(lm) >= 468 else [0, 1, 2]
        src = lm[idx].astype(np.float32)
        dst = smoothed_lm[idx].astype(np.float32)

        M = cv2.getAffineTransform(src, dst)
        H, W = frame.shape[:2]
        warped = cv2.warpAffine(frame, M, (W, H),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

        # Blend: warp only inside face mask
        from module_C_visual.C2_segment.mask_utils import soft_mask as make_soft
        weight = make_soft(mask, blur_radius=15)[:, :, np.newaxis]
        blended = warped.astype(np.float64) * weight + frame.astype(np.float64) * (1 - weight)
        result.append(np.clip(blended, 0, 255).astype(np.uint8))

    return result


if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--masks', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    frame_files = sorted(os.listdir(args.input))
    mask_files = sorted(os.listdir(args.masks))

    frames = [cv2.imread(os.path.join(args.input, f)) for f in frame_files]
    masks = [cv2.imread(os.path.join(args.masks, f), cv2.IMREAD_GRAYSCALE)
             for f in mask_files]

    smoothed = kalman_smooth_frames(frames, masks)

    os.makedirs(args.output, exist_ok=True)
    for i, f in enumerate(smoothed):
        cv2.imwrite(os.path.join(args.output, f'{i:06d}.png'), f)
    print(f"Smoothed {len(smoothed)} frames → {args.output}")
