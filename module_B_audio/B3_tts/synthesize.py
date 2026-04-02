"""CosyVoice TTS synthesis stub. Requires conda activate cosyvoice."""
import os

def synthesize(text_path: str, output_path: str, voice_prompt: str = None):
    raise NotImplementedError(
        "synthesize() not yet implemented. Run in cosyvoice env:\n"
        "  conda activate cosyvoice\n"
        "  python module_B_audio/B3_tts/synthesize.py --text trans.json --output new_audio.wav"
    )

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--text', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--voice_prompt', default=None)
    a = p.parse_args()
    synthesize(a.text, a.output, a.voice_prompt)
