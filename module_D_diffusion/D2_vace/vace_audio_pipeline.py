"""VACE + FantasyTalking audio adapter: masked V2V with audio conditioning.

Mode 2 core: loads VACE pipeline, grafts FantasyTalking's audio cross-attention
processors onto all 40 DiT blocks, then runs masked V2V with differential
strength noising.

Key components from FantasyTalking (reused, not modified):
  - AudioProjModel: Linear(768→2048) + LayerNorm
  - WanCrossAttentionProcessor: zero-init k/v proj, per-frame audio attention
  - split_audio_sequence(): maps audio to per-latent-frame windows

New in this file:
  - build_vace_audio_model(): load VACE + graft audio processors
  - run_vace_audio(): end-to-end masked V2V with audio + strength map
"""

import os
import sys
import numpy as np
import torch

# FantasyTalking audio modules — imported from cloned repo
FT_REPO = os.path.join(os.path.dirname(__file__), '..', 'D1_fantasytalking', 'fantasy-talking')
if os.path.isdir(FT_REPO):
    sys.path.insert(0, FT_REPO)

from diff_strength_map import (
    build_strength_map_latent,
    apply_differential_noise,
    apply_blended_denoise_step,
)


def load_audio_modules(ft_checkpoint: str, device: str = 'cuda'):
    """Load FantasyTalking's audio projection and processor weights.

    Args:
        ft_checkpoint: path to fantasytalking_model.ckpt
        device: target device.

    Returns:
        audio_proj: AudioProjModel instance.
        audio_processor_state: dict of per-block k/v projection weights.
    """
    from models.wan_audio import AudioProjModel

    state = torch.load(ft_checkpoint, map_location='cpu')

    audio_proj = AudioProjModel()
    proj_state = {k.replace('audio_proj.', ''): v
                  for k, v in state.items() if k.startswith('audio_proj')}
    audio_proj.load_state_dict(proj_state)
    audio_proj = audio_proj.to(device)
    audio_proj.eval()

    # Processor weights (k_proj, v_proj per block)
    proc_state = {k: v for k, v in state.items()
                  if 'audio' in k and not k.startswith('audio_proj')}

    return audio_proj, proc_state


def extract_wav2vec2_features(audio_path: str,
                              model_name: str = 'facebook/wav2vec2-base-960h',
                              device: str = 'cuda') -> torch.Tensor:
    """Extract wav2vec2 features from audio file.

    Args:
        audio_path: path to 16kHz mono WAV.
        model_name: HuggingFace model name or local path.

    Returns:
        (T_audio, 768) tensor of audio features.
    """
    import torchaudio
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.mean(0)  # mono

    inputs = processor(waveform.numpy(), sampling_rate=16000,
                       return_tensors='pt', padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    return outputs.last_hidden_state.squeeze(0)  # (T_a, 768)


def install_audio_processors(transformer, proc_state: dict,
                             audio_scale: float = 1.0):
    """Graft FantasyTalking's WanCrossAttentionProcessor onto VACE DiT blocks.

    Installs audio cross-attention into all 40 blocks with zero-init k/v.
    Loads trained weights from proc_state if available.
    """
    from models.wan_audio import WanCrossAttentionProcessor, set_audio_processor

    set_audio_processor(transformer, audio_scale=audio_scale)

    # Load trained k/v projection weights per block
    if proc_state:
        missing = []
        for name, param in transformer.named_parameters():
            if 'audio' in name and name in proc_state:
                param.data.copy_(proc_state[name])
            elif 'audio' in name:
                missing.append(name)
        if missing:
            print(f"[warn] {len(missing)} audio params not found in checkpoint"
                  f" (expected if using zero-init)")


def build_vace_audio_model(
    vace_checkpoint: str,
    ft_checkpoint: str,
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16,
    audio_scale: float = 1.0,
):
    """Build VACE pipeline with grafted audio cross-attention.

    Args:
        vace_checkpoint: path to VACE-Wan2.1-14B directory.
        ft_checkpoint: path to fantasytalking_model.ckpt.
        device: CUDA device.
        dtype: model precision (bfloat16 recommended).
        audio_scale: scaling factor for audio attention contribution.

    Returns:
        pipe: VACE pipeline with audio processors installed.
        audio_proj: AudioProjModel for wav2vec2→DiT projection.
    """
    # Import VACE pipeline (adapt path after cloning)
    try:
        from diffsynth import VACEVideoPipeline
        pipe = VACEVideoPipeline.from_pretrained(vace_checkpoint,
                                                  torch_dtype=dtype)
    except ImportError:
        # Fallback: try loading as a generic diffusers pipeline
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(vace_checkpoint,
                                                  torch_dtype=dtype)

    audio_proj, proc_state = load_audio_modules(ft_checkpoint, device)

    install_audio_processors(pipe.transformer, proc_state,
                             audio_scale=audio_scale)

    pipe = pipe.to(device)
    return pipe, audio_proj


def run_vace_audio(
    pipe,
    audio_proj,
    src_video_path: str,
    src_mask_dir: str,
    audio_path: str,
    ref_frame_path: str,
    strength_map: np.ndarray,
    num_steps: int = 25,
    n_latent_frames: int = 21,
    device: str = 'cuda',
) -> list[np.ndarray]:
    """Run VACE masked V2V with audio conditioning.

    Args:
        pipe: VACE pipeline with audio processors.
        audio_proj: AudioProjModel.
        src_video_path: path to source video or frames directory.
        src_mask_dir: directory of upper_body_mask PNGs.
        audio_path: path to new audio WAV (16kHz mono).
        ref_frame_path: reference frame PNG for identity anchoring.
        strength_map: (H, W) float32 per-pixel editing strength [0, 1].
        num_steps: number of denoising steps.
        n_latent_frames: number of latent frames (81 video frames → 21).

    Returns:
        list of (H, W, 3) BGR uint8 output frames.
    """
    from models.wan_audio import split_audio_sequence

    # 1. Extract and project audio features
    audio_feat = extract_wav2vec2_features(audio_path, device=device)
    audio_cond = audio_proj(audio_feat.unsqueeze(0))  # (1, T_a, 2048)
    audio_windows = split_audio_sequence(audio_cond, n_latent_frames)
    # → (1, n_latent_frames, window_len, 2048)

    # 2. Encode source video through VAE
    # (implementation depends on VACE's specific API)
    z_orig = pipe.encode_video(src_video_path)  # (1, 16, T_lat, H_lat, W_lat)

    # 3. Build latent-space strength map
    _, _, _, lat_h, lat_w = z_orig.shape
    smap_latent = build_strength_map_latent(strength_map, lat_h, lat_w)
    smap_latent = smap_latent.to(device)

    # 4. Apply differential noise
    z_noisy, t_starts = apply_differential_noise(z_orig, smap_latent)

    # 5. Run VACE denoising with audio conditioning
    # The audio_windows are passed to WanCrossAttentionProcessors via pipe's
    # internal transformer forward pass. VACE's context adapter handles
    # src_video and src_mask for background preservation.
    output = pipe(
        latents=z_noisy,
        src_video=z_orig,
        src_mask=src_mask_dir,
        src_ref_images=[ref_frame_path],
        audio_feat=audio_windows,
        num_inference_steps=num_steps,
    )

    # 6. Decode latents to frames
    if hasattr(output, 'frames'):
        return output.frames
    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='VACE + audio masked V2V')
    parser.add_argument('--src_video', required=True)
    parser.add_argument('--src_mask', required=True)
    parser.add_argument('--audio_path', required=True)
    parser.add_argument('--ref_frame', required=True)
    parser.add_argument('--vace_ckpt', required=True)
    parser.add_argument('--ft_ckpt', required=True)
    parser.add_argument('--strength_map', default='lips:0.5,face:0.35,shoulder:0.15')
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()

    import cv2
    from module_C_visual.C2_segment.mask_utils import build_region_strength_map

    pipe, audio_proj = build_vace_audio_model(args.vace_ckpt, args.ft_ckpt)

    # Build strength map from masks (simplified — use actual masks in production)
    # For CLI, parse strength string
    strengths = dict(s.split(':') for s in args.strength_map.split(','))

    print(f"Running VACE + audio, {args.num_steps} steps...")
    # Placeholder: actual strength_map built from mask files
    frames = run_vace_audio(
        pipe, audio_proj,
        args.src_video, args.src_mask, args.audio_path, args.ref_frame,
        strength_map=np.zeros((480, 832), dtype=np.float32),  # placeholder
        num_steps=args.num_steps,
    )

    os.makedirs(args.output_path, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(args.output_path, f'{i:06d}.png'), f)
    print(f"Saved {len(frames)} frames to {args.output_path}")
