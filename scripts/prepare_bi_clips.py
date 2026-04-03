"""Prepare training clips from BI presenter videos.

Scans raw videos, detects face-on-screen segments, extracts 81-frame clips
at 25fps with audio.  Output structure for train_audio_adapter.py:

    data/training/
        clip_000/
            frames/000000.png ... 000080.png   (832x480 BGR)
            audio.wav                           (16kHz mono)
        clip_001/
        ...

Usage:
    python scripts/prepare_bi_clips.py \
        --video_dir  "data/raw/笔吧视频数据" \
        --output_dir  data/training \
        --max_clips   500 \
        --skip_existing
"""

import argparse
import os
import sys
import subprocess
import random

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def find_videos(video_dir: str) -> list[str]:
    """Recursively find all .mp4 files."""
    videos = []
    for root, _, files in os.walk(video_dir):
        for f in sorted(files):
            if f.endswith('.mp4'):
                videos.append(os.path.join(root, f))
    return videos


def detect_face_segments(video_path: str, sample_every: int = 50,
                         min_face_ratio: float = 0.02) -> list[tuple[int, int]]:
    """Detect contiguous segments where a face occupies enough of the frame.

    Returns list of (start_frame, end_frame) at the video's native fps.
    Uses OpenCV's DNN face detector (fast, no extra deps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    area = W * H

    # Use Haar cascade (ships with opencv)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face_frames = []
    for i in range(0, total, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) > 0:
            # Check largest face size
            largest = max(faces, key=lambda f: f[2] * f[3])
            face_area = largest[2] * largest[3]
            if face_area / area >= min_face_ratio:
                face_frames.append(i)

    cap.release()

    if not face_frames:
        return []

    # Merge into contiguous segments (gap tolerance = 2 * sample_every)
    segments = []
    seg_start = face_frames[0]
    seg_end = face_frames[0]
    gap_tol = 2 * sample_every

    for f in face_frames[1:]:
        if f - seg_end <= gap_tol:
            seg_end = f
        else:
            segments.append((seg_start, seg_end + sample_every))
            seg_start = f
            seg_end = f
    segments.append((seg_start, seg_end + sample_every))

    return segments


def extract_clip(video_path: str, start_sec: float, duration_sec: float,
                 output_dir: str, target_fps: int = 25,
                 target_size: tuple = (832, 480)) -> bool:
    """Extract one 81-frame clip: frames + audio."""
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, 'audio.wav')

    W, H = target_size

    # Extract frames with ffmpeg (fast, handles seeking well)
    ret = subprocess.run([
        'ffmpeg', '-ss', f'{start_sec:.3f}',
        '-i', video_path,
        '-t', f'{duration_sec:.3f}',
        '-vf', f'fps={target_fps},scale={W}:{H}',
        '-q:v', '2',
        '-frames:v', '81',
        os.path.join(frames_dir, '%06d.png'),
        '-y', '-hide_banner', '-loglevel', 'error'
    ], capture_output=True)

    if ret.returncode != 0:
        return False

    # Check we got 81 frames
    n_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if n_frames < 81:
        return False

    # Extract audio
    ret = subprocess.run([
        'ffmpeg', '-ss', f'{start_sec:.3f}',
        '-i', video_path,
        '-t', f'{duration_sec:.3f}',
        '-vn', '-ar', '16000', '-ac', '1',
        audio_path,
        '-y', '-hide_banner', '-loglevel', 'error'
    ], capture_output=True)

    if ret.returncode != 0:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare training clips')
    parser.add_argument('--video_dir', required=True,
                        help='Directory containing raw videos')
    parser.add_argument('--output_dir', default='data/training',
                        help='Output directory for clips')
    parser.add_argument('--max_clips', type=int, default=500,
                        help='Maximum number of clips to extract')
    parser.add_argument('--clip_frames', type=int, default=81)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--min_face_ratio', type=float, default=0.02,
                        help='Minimum face-to-frame area ratio')
    args = parser.parse_args()

    clip_duration = args.clip_frames / args.fps  # 81/25 = 3.24s

    videos = find_videos(args.video_dir)
    print(f"Found {len(videos)} videos in {args.video_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    clip_idx = 0

    # Count existing clips
    if args.skip_existing:
        existing = [d for d in os.listdir(args.output_dir)
                    if os.path.isdir(os.path.join(args.output_dir, d))]
        clip_idx = len(existing)
        print(f"Skipping {clip_idx} existing clips")

    random.shuffle(videos)  # Diverse sampling

    for vi, vpath in enumerate(videos):
        if clip_idx >= args.max_clips:
            break

        print(f"\n[{vi+1}/{len(videos)}] {os.path.basename(vpath)}")

        # Get video duration
        probe = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', vpath
        ], capture_output=True, text=True)
        try:
            float(probe.stdout.strip())
        except ValueError:
            print(f"  Cannot read duration, skipping")
            continue

        # Get native fps
        probe2 = subprocess.run([
            'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'csv=p=0', vpath
        ], capture_output=True, text=True)
        try:
            num, den = probe2.stdout.strip().split('/')
            native_fps = int(num) / int(den)
        except (ValueError, ZeroDivisionError):
            native_fps = 30

        # Detect face-on-screen segments
        segments = detect_face_segments(
            vpath, sample_every=int(native_fps * 2),
            min_face_ratio=args.min_face_ratio)

        if not segments:
            print(f"  No face segments found")
            continue

        print(f"  {len(segments)} face segments, "
              f"total {sum(e-s for s,e in segments)/native_fps:.0f}s of face")

        # Extract clips from face segments
        for seg_start, seg_end in segments:
            if clip_idx >= args.max_clips:
                break

            seg_start_sec = seg_start / native_fps
            seg_end_sec = seg_end / native_fps
            seg_duration = seg_end_sec - seg_start_sec

            # Skip segments shorter than one clip
            if seg_duration < clip_duration + 0.5:
                continue

            # Sample clips from this segment (non-overlapping)
            t = seg_start_sec
            while t + clip_duration <= seg_end_sec and clip_idx < args.max_clips:
                clip_name = f'clip_{clip_idx:04d}'
                clip_dir = os.path.join(args.output_dir, clip_name)

                if args.skip_existing and os.path.isdir(clip_dir):
                    clip_idx += 1
                    t += clip_duration
                    continue

                ok = extract_clip(vpath, t, clip_duration, clip_dir,
                                  target_fps=args.fps)
                if ok:
                    print(f"  ✓ {clip_name} ({t:.1f}s–{t+clip_duration:.1f}s)")
                    clip_idx += 1
                else:
                    # Clean up failed clip
                    import shutil
                    if os.path.isdir(clip_dir):
                        shutil.rmtree(clip_dir)

                t += clip_duration  # non-overlapping

    print(f"\nDone: {clip_idx} clips in {args.output_dir}")


if __name__ == '__main__':
    main()
