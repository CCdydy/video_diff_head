"""FFmpeg audio/video mux — final output encoding.

Settings: -c:v libx264 -crf 18 -c:a aac (from README)
"""

import os
import subprocess


def mux_output(
    video_input: str,
    audio_path: str,
    output_path: str,
    fps: int = 25,
    crf: int = 18,
) -> str:
    """Mux video frames/video + audio into final output.

    Args:
        video_input: frames directory (%06d.png) or video file.
        audio_path: audio WAV path.
        output_path: output MP4 path.
        fps: framerate (if input is frames directory).
        crf: x264 quality (lower = better, 18 = visually lossless).

    Returns:
        output_path.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if os.path.isdir(video_input):
        video_arg = f'-framerate {fps} -i "{video_input}/%06d.png"'
    else:
        video_arg = f'-i "{video_input}"'

    cmd = (
        f'ffmpeg {video_arg} -i "{audio_path}" '
        f'-c:v libx264 -crf {crf} -pix_fmt yuv420p '
        f'-c:a aac -b:a 192k -shortest '
        f'"{output_path}" -y -hide_banner -loglevel error'
    )

    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"FFmpeg mux failed (exit {ret})")
    print(f"[mux] Output: {output_path}")
    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Frames dir or video file')
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--crf', type=int, default=18)
    args = parser.parse_args()

    mux_output(args.video, args.audio, args.output, args.fps, args.crf)
