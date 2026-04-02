"""LLM-based translation stub."""
import json, os

def translate(input_path: str, output_path: str, target_lang: str = 'ja'):
    raise NotImplementedError(
        "translate() not yet implemented."
    )

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--target', default='ja')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    translate(a.input, a.output, a.target)
