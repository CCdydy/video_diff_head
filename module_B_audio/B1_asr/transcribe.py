"""FunASR transcription stub. Requires conda activate funasr."""
import json, os

def transcribe(video_path: str, output_path: str):
    raise NotImplementedError(
        "transcribe() not yet implemented. Run in funasr env:\n"
        "  conda activate funasr\n"
        "  python module_B_audio/B1_asr/transcribe.py --video input.mp4 --output asr.json"
    )

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--output', required=True)
    transcribe(p.parse_args().video, p.parse_args().output)
