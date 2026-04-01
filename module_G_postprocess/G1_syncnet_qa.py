"""SyncNet lip-sync quality gate.

Computes sync_conf score between video and audio.
  sync_conf > 3.0 → pass
  sync_conf < 3.0 → flag for review

Uses joonson/syncnet_python as backend.
"""

import os
import subprocess
import json


def syncnet_qa(
    video_dir: str,
    audio_path: str,
    threshold: float = 3.0,
    syncnet_dir: str = None,
) -> dict:
    """Run SyncNet evaluation and return quality metrics.

    Args:
        video_dir: directory of composited frames (or video file).
        audio_path: path to audio WAV.
        threshold: sync_conf threshold for pass/fail.
        syncnet_dir: path to syncnet_python repo (auto-detect if None).

    Returns:
        dict with 'sync_conf', 'sync_dist', 'pass' keys.
    """
    if syncnet_dir is None:
        syncnet_dir = os.environ.get('SYNCNET_DIR', '')

    # Mux frames + audio into a temp video for SyncNet
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.close()

    if os.path.isdir(video_dir):
        mux_cmd = (
            f'ffmpeg -framerate 25 -i "{video_dir}/%06d.png" '
            f'-i "{audio_path}" -c:v libx264 -pix_fmt yuv420p '
            f'-c:a aac -shortest "{tmp.name}" -y -hide_banner -loglevel error'
        )
        os.system(mux_cmd)
        video_path = tmp.name
    else:
        video_path = video_dir

    # Run SyncNet evaluation
    # Adapt this to your SyncNet installation
    result = {'sync_conf': 0.0, 'sync_dist': 0.0, 'pass': False, 'video': video_path}

    try:
        script = os.path.join(syncnet_dir, 'run_pipeline.py') if syncnet_dir else ''
        if os.path.isfile(script):
            output = subprocess.check_output(
                ['python', script, '--videofile', video_path],
                stderr=subprocess.STDOUT, text=True
            )
            # Parse SyncNet output (format depends on implementation)
            for line in output.splitlines():
                if 'conf' in line.lower():
                    parts = line.split()
                    for j, p in enumerate(parts):
                        if 'conf' in p.lower() and j + 1 < len(parts):
                            try:
                                result['sync_conf'] = float(parts[j + 1].strip(','))
                            except ValueError:
                                pass
        else:
            print(f"[syncnet_qa] SyncNet not found at {script}")
            print(f"[syncnet_qa] Set SYNCNET_DIR env var or pass --syncnet_dir")
            result['sync_conf'] = -1.0
    except subprocess.CalledProcessError as e:
        print(f"[syncnet_qa] SyncNet failed: {e}")
        result['sync_conf'] = -1.0

    result['pass'] = result['sync_conf'] >= threshold
    status = "PASS ✓" if result['pass'] else "FAIL ✗"
    print(f"[syncnet_qa] sync_conf={result['sync_conf']:.2f} "
          f"threshold={threshold} → {status}")

    # Cleanup temp file
    if os.path.isdir(video_dir) and os.path.exists(tmp.name):
        os.unlink(tmp.name)

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--threshold', type=float, default=3.0)
    args = parser.parse_args()

    result = syncnet_qa(args.video, args.audio, args.threshold)
    print(json.dumps(result, indent=2))
