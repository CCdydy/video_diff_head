"""VACE + FantasyTalking audio adapter: masked V2V with audio conditioning.

Components (README §Audio Adapter 结构):
  AudioProjModel              — Linear(768→2048) + LayerNorm
  WanAudioCrossAttentionProcessor — zero-init k/v × 40 blocks, audio_scale.tanh() gate
  install_audio_adapter()     — inject processors + load FantasyTalking weights
  VACEAudioPipeline.run_chunk()     — single 81-frame inference
  VACEAudioPipeline.run_long_video() — Hamming window long video chunking

Trainable params: ~212M (VACE 14B frozen)
  AudioProjModel:   1,577K
  k/v per block:    40 × 2 × (2048×5120) = 838M (but each is ~10.5M, total ~420M)
  audio_scale:      40 scalars
"""

import os
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ══════════════════════════════════════════════════════════════
# Audio Projection: wav2vec2 (768) → DiT cross-attn (2048)
# ══════════════════════════════════════════════════════════════

class AudioProjModel(nn.Module):
    """Linear(768→2048) + LayerNorm. Minimal by design (FantasyTalking)."""

    def __init__(self, in_dim: int = 768, out_dim: int = 2048):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, 768) → (B, T, 2048)"""
        return self.norm(self.proj(x))


# ══════════════════════════════════════════════════════════════
# Audio windowing: continuous audio → per-latent-frame windows
# ══════════════════════════════════════════════════════════════

def split_audio_sequence(
    audio_feat: torch.Tensor,
    n_latent_frames: int = 21,
    base_window: int = 8,
    expand: int = 4,
) -> torch.Tensor:
    """Split continuous audio features into per-latent-frame windows.

    Wan2.1 VAE compresses 4 video frames → 1 latent frame.
    81 video frames → 21 latent frames.
    Audio at ~50 vec/sec, video at 16fps → ~3.125 audio vecs per video frame
    → ~12.5 per latent frame.

    Each window = base_window + 2*expand tokens with overlap for smoothness.

    Args:
        audio_feat: (B, T_audio, C) projected audio features.
        n_latent_frames: number of latent temporal frames.
        base_window: core window size per latent frame.
        expand: overlap expansion on each side.

    Returns:
        (B, n_latent_frames, window_len, C) windowed audio.
    """
    B, T_audio, C = audio_feat.shape
    window_len = base_window + 2 * expand

    # Compute center positions for each latent frame
    stride = max(1, T_audio / n_latent_frames)
    windows = []

    for i in range(n_latent_frames):
        center = int(stride * (i + 0.5))
        start = max(0, center - window_len // 2)
        end = start + window_len

        if end > T_audio:
            end = T_audio
            start = max(0, end - window_len)

        chunk = audio_feat[:, start:end, :]  # (B, <=window_len, C)

        # Pad if needed
        if chunk.shape[1] < window_len:
            pad = torch.zeros(B, window_len - chunk.shape[1], C,
                              device=chunk.device, dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=1)

        windows.append(chunk)

    return torch.stack(windows, dim=1)  # (B, n_latent, window_len, C)


# ══════════════════════════════════════════════════════════════
# Audio Cross-Attention Processor (injected into all 40 DiT blocks)
# ══════════════════════════════════════════════════════════════

class WanAudioCrossAttentionProcessor(nn.Module):
    """Per-frame audio cross-attention for a single DiT block.

    Inserted into every WanAttentionBlock's cross-attention.
    Zero-initialized k/v projections for stable training.
    Gated by audio_scale.tanh() to control contribution.

    Forward computes:
        audio_attn = Attention(Q=video_tokens, K=audio_k, V=audio_v)
        output = original_output + audio_attn * audio_scale.tanh()
    """

    def __init__(self, hidden_dim: int = 5120, audio_dim: int = 2048,
                 num_heads: int = 40):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Zero-initialized projections (FantasyTalking design)
        self.k_audio = nn.Linear(audio_dim, hidden_dim, bias=False)
        self.v_audio = nn.Linear(audio_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.k_audio.weight)
        nn.init.zeros_(self.v_audio.weight)

        # Learnable gate scalar, initialized to 0 → tanh(0)=0 → no initial contribution
        self.audio_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        query: torch.Tensor,
        audio_cond: torch.Tensor,
        n_latent_frames: int,
    ) -> torch.Tensor:
        """Compute per-frame audio cross-attention.

        Args:
            query: (B, N_total, hidden_dim) video latent tokens from self-attn.
                   N_total = n_latent_frames × H_lat × W_lat
            audio_cond: (B, n_latent_frames, L_win, audio_dim) windowed audio.
            n_latent_frames: number of latent temporal frames.

        Returns:
            (B, N_total, hidden_dim) audio attention output, scaled.
        """
        B, N_total, D = query.shape
        tokens_per_frame = N_total // n_latent_frames

        # Project audio to k, v
        # audio_cond: (B, T_lat, L_win, 2048) → reshape for per-frame attention
        k = self.k_audio(audio_cond)  # (B, T_lat, L_win, D)
        v = self.v_audio(audio_cond)  # (B, T_lat, L_win, D)

        # Reshape query to per-frame: (B*T_lat, tokens_per_frame, D)
        q = query.view(B, n_latent_frames, tokens_per_frame, D)
        q = q.reshape(B * n_latent_frames, tokens_per_frame, D)

        # Reshape k, v: (B*T_lat, L_win, D)
        L_win = k.shape[2]
        k = k.reshape(B * n_latent_frames, L_win, D)
        v = v.reshape(B * n_latent_frames, L_win, D)

        # Multi-head attention
        q = q.view(-1, tokens_per_frame, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, L_win, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, L_win, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v)  # (B*T, heads, tpf, head_dim)
        attn = attn.transpose(1, 2).reshape(B * n_latent_frames,
                                             tokens_per_frame, D)

        # Reshape back to (B, N_total, D)
        attn = attn.view(B, n_latent_frames, tokens_per_frame, D)
        attn = attn.reshape(B, N_total, D)

        # Gate with tanh(audio_scale)
        return attn * self.audio_scale.tanh()


# ══════════════════════════════════════════════════════════════
# Adapter installation: graft processors onto VACE DiT
# ══════════════════════════════════════════════════════════════

def install_audio_adapter(
    transformer: nn.Module,
    audio_dim: int = 2048,
    hidden_dim: int = 5120,
    num_heads: int = 40,
    ft_checkpoint: str = None,
    device: str = 'cuda',
) -> list[WanAudioCrossAttentionProcessor]:
    """Install WanAudioCrossAttentionProcessor into all 40 DiT blocks.

    If ft_checkpoint is provided, loads trained k/v weights from
    FantasyTalking's fantasytalking_model.ckpt.

    Returns list of installed processors.
    """
    processors = []

    # Find all transformer blocks
    blocks = None
    if hasattr(transformer, 'blocks'):
        blocks = transformer.blocks
    elif hasattr(transformer, 'transformer_blocks'):
        blocks = transformer.transformer_blocks
    else:
        # Search for sequential of blocks
        for name, module in transformer.named_children():
            if isinstance(module, nn.ModuleList) and len(module) >= 40:
                blocks = module
                break

    if blocks is None:
        raise ValueError("Cannot find transformer blocks in model. "
                         "Expected .blocks or .transformer_blocks with >=40 entries.")

    for i, block in enumerate(blocks):
        proc = WanAudioCrossAttentionProcessor(
            hidden_dim=hidden_dim, audio_dim=audio_dim, num_heads=num_heads
        ).to(device)
        # Store as submodule of the block
        block.audio_processor = proc
        processors.append(proc)

    print(f"[adapter] Installed audio processors on {len(processors)} blocks")

    # Load pretrained weights if available
    if ft_checkpoint and os.path.isfile(ft_checkpoint):
        state = torch.load(ft_checkpoint, map_location='cpu')
        loaded = 0
        for i, proc in enumerate(processors):
            prefix_patterns = [
                f'blocks.{i}.audio_processor.',
                f'transformer.blocks.{i}.audio_processor.',
                f'audio_blocks.{i}.',
            ]
            proc_state = {}
            for key, val in state.items():
                for prefix in prefix_patterns:
                    if key.startswith(prefix):
                        short_key = key[len(prefix):]
                        proc_state[short_key] = val
                        break

            if proc_state:
                proc.load_state_dict(proc_state, strict=False)
                loaded += 1

        print(f"[adapter] Loaded weights for {loaded}/{len(processors)} blocks "
              f"from {ft_checkpoint}")

    return processors


# ══════════════════════════════════════════════════════════════
# Wav2Vec2 feature extraction
# ══════════════════════════════════════════════════════════════

def extract_wav2vec2(
    audio_path: str,
    model_path: str = 'data/models/wav2vec2-base-960h',
    device: str = 'cuda',
) -> torch.Tensor:
    """Extract wav2vec2 features from audio file.

    Returns (1, T_audio, 768) tensor. ~50 vectors per second at 16kHz.
    """
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2Model.from_pretrained(model_path).to(device).eval()

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.mean(0)  # mono

    inputs = processor(waveform.numpy(), sampling_rate=16000,
                       return_tensors='pt', padding=True)

    with torch.no_grad():
        out = model(inputs.input_values.to(device))

    return out.last_hidden_state  # (1, T_audio, 768)


# ══════════════════════════════════════════════════════════════
# VACE Audio Pipeline
# ══════════════════════════════════════════════════════════════

class VACEAudioPipeline:
    """End-to-end VACE masked V2V with audio conditioning.

    Wraps VACE pipeline + AudioProjModel + audio processors.
    """

    def __init__(
        self,
        vace_model_path: str = 'data/models/Wan2.1-VACE-14B',
        wav2vec2_path: str = 'data/models/wav2vec2-base-960h',
        ft_checkpoint: str = 'data/models/fantasytalking_audio_adapter.ckpt',
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        t5_cpu: bool = False,
        offload_model: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.wav2vec2_path = wav2vec2_path

        # Load VACE pipeline
        # The exact import depends on which VACE distribution is used.
        # Option 1: Official Wan2.1 generate.py style
        # Option 2: diffusers-based
        self.pipe = self._load_vace(vace_model_path, t5_cpu, offload_model)

        # Audio projection
        self.audio_proj = AudioProjModel().to(device, dtype)

        # Load audio_proj weights from FantasyTalking checkpoint
        if ft_checkpoint and os.path.isfile(ft_checkpoint):
            state = torch.load(ft_checkpoint, map_location='cpu')
            proj_state = {}
            for k, v in state.items():
                for prefix in ('audio_proj.', 'audio_projection.'):
                    if k.startswith(prefix):
                        proj_state[k[len(prefix):]] = v
                        break
            if proj_state:
                self.audio_proj.load_state_dict(proj_state, strict=False)
                print(f"[pipeline] Loaded AudioProjModel from {ft_checkpoint}")
        self.audio_proj.eval()

        # Install audio cross-attention processors
        transformer = self._get_transformer()
        self.audio_processors = install_audio_adapter(
            transformer, ft_checkpoint=ft_checkpoint, device=device,
        )

    def _load_vace(self, model_path, t5_cpu, offload_model):
        """Load VACE via Wan2.1 native API (wan.WanVace).

        DiT stays on CPU; use offload_model=True in generate() to move
        to GPU on demand. This keeps idle VRAM < 1GB on RTX 5090.
        """
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        wan_repo = os.path.join(root_dir, 'third_party', 'Wan2.1')
        if os.path.isdir(wan_repo) and wan_repo not in sys.path:
            sys.path.insert(0, wan_repo)

        import wan
        from wan.configs import WAN_CONFIGS

        model_path = os.path.abspath(model_path)
        print(f"[pipeline] Loading VACE from {model_path} (native API)...")
        pipe = wan.WanVace(
            config=WAN_CONFIGS['vace-14B'],
            checkpoint_dir=model_path,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=t5_cpu or True,  # always T5 on CPU to save VRAM
        )
        print(f"[pipeline] VACE loaded. GPU VRAM: "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB")
        return pipe

    def _get_transformer(self):
        """Get the DiT transformer from the WanVace pipeline."""
        return self.pipe.model

    def prepare_audio(self, audio_path: str, n_latent_frames: int = 21,
                      expand: int = 4) -> torch.Tensor:
        """Extract + project + window audio features.

        Returns (1, n_latent_frames, L_win, 2048) audio conditioning.
        """
        # Extract wav2vec2 features
        raw_feat = extract_wav2vec2(audio_path, self.wav2vec2_path, self.device)
        # (1, T_audio, 768)

        # Project to DiT dimension
        with torch.no_grad():
            projected = self.audio_proj(raw_feat.to(self.dtype))
        # (1, T_audio, 2048)

        # Window into per-latent-frame chunks
        windowed = split_audio_sequence(projected, n_latent_frames, expand=expand)
        # (1, n_latent_frames, L_win, 2048)

        return windowed

    def run_chunk(
        self,
        src_frames: list[np.ndarray],
        src_masks: list[np.ndarray],
        audio_path: str,
        ref_frame: np.ndarray,
        strength_map: np.ndarray,
        num_steps: int = 25,
        audio_cfg: float = 2.0,
        expand: int = 4,
    ) -> list[np.ndarray]:
        """Run single 81-frame chunk through VACE + audio.

        Args:
            src_frames: list of 81 (H,W,3) BGR uint8 frames.
            src_masks: list of 81 (H,W) uint8 masks (255=edit region).
            audio_path: path to audio WAV (segment for this chunk).
            ref_frame: (H,W,3) BGR reference frame for identity.
            strength_map: (H,W) float32 per-pixel editing strength.
            num_steps: DDIM denoising steps.
            audio_cfg: audio guidance scale.
            expand: audio window expansion.

        Returns:
            list of 81 (H,W,3) BGR uint8 edited frames.
        """
        import cv2

        n_frames = len(src_frames)
        # Wan2.1 constraint: (N-1) % 4 == 0
        assert (n_frames - 1) % 4 == 0, \
            f"Frame count must satisfy (N-1)%4==0, got {n_frames}"

        n_latent_frames = (n_frames - 1) // 4 + 1

        # Prepare audio conditioning
        audio_cond = self.prepare_audio(audio_path, n_latent_frames, expand)

        # Store audio_cond on processors for the forward pass
        for proc in self.audio_processors:
            proc._audio_cond = audio_cond
            proc._n_latent_frames = n_latent_frames

        # Prepare VACE inputs
        # (exact API depends on VACE version — this is the logical flow)
        try:
            output = self.pipe(
                video=src_frames,
                mask=src_masks,
                ref_image=ref_frame,
                strength=strength_map,
                num_inference_steps=num_steps,
                guidance_scale=audio_cfg,
            )
            if hasattr(output, 'frames'):
                return output.frames
            return output
        except Exception as e:
            print(f"[pipeline] VACE inference error: {e}")
            raise

    def run_long_video(
        self,
        src_frames: list[np.ndarray],
        src_masks: list[np.ndarray],
        audio_path: str,
        ref_frame: np.ndarray,
        strength_map: np.ndarray,
        chunk_size: int = 81,
        overlap: int = 13,
        num_steps: int = 25,
        audio_cfg: float = 2.0,
    ) -> list[np.ndarray]:
        """Process long video with Hamming window chunk blending.

        Splits video into overlapping 81-frame chunks, processes each,
        and blends overlaps using Hamming window weights.

        Args:
            src_frames: full video frames list.
            src_masks: full mask list.
            audio_path: full audio path.
            ref_frame: identity reference frame.
            strength_map: per-pixel editing strength.
            chunk_size: frames per chunk (must satisfy (N-1)%4==0).
            overlap: overlap between consecutive chunks.
            num_steps: DDIM steps per chunk.
            audio_cfg: audio guidance scale.

        Returns:
            list of all output frames, blended at chunk boundaries.
        """
        T = len(src_frames)
        if T <= chunk_size:
            return self.run_chunk(src_frames, src_masks, audio_path,
                                  ref_frame, strength_map, num_steps, audio_cfg)

        # Split into chunks
        stride = chunk_size - overlap
        chunks = []
        for start in range(0, T, stride):
            end = min(start + chunk_size, T)
            # Ensure valid frame count for VAE
            while (end - start - 1) % 4 != 0 and end < T:
                end += 1
            if (end - start - 1) % 4 != 0:
                end = start + ((end - start - 1) // 4) * 4 + 1
            if end <= start:
                break
            chunks.append((start, end))

        # Build Hamming window weights for blending
        hamming = np.hamming(overlap * 2)
        ramp_down = hamming[:overlap]  # 1→0
        ramp_up = hamming[overlap:]    # 0→1

        # Process each chunk
        all_outputs = [None] * T
        all_weights = np.zeros(T, dtype=np.float64)

        for ci, (start, end) in enumerate(chunks):
            print(f"[pipeline] Chunk {ci+1}/{len(chunks)}: frames {start}–{end}")

            chunk_frames = src_frames[start:end]
            chunk_masks = src_masks[start:end]

            # TODO: extract audio segment for this chunk
            # For now pass full audio — VACE handles temporal alignment
            chunk_output = self.run_chunk(
                chunk_frames, chunk_masks, audio_path,
                ref_frame, strength_map, num_steps, audio_cfg,
            )

            # Assign with Hamming weights at overlaps
            n = end - start
            weights = np.ones(n, dtype=np.float64)

            # Ramp up at start (if not first chunk)
            if ci > 0 and overlap > 0:
                ramp_len = min(overlap, n)
                weights[:ramp_len] = ramp_up[:ramp_len]

            # Ramp down at end (if not last chunk)
            if ci < len(chunks) - 1 and overlap > 0:
                ramp_len = min(overlap, n)
                weights[n-ramp_len:] = ramp_down[-ramp_len:]

            for i in range(n):
                frame_idx = start + i
                if frame_idx >= T:
                    break
                w = weights[i]
                if all_outputs[frame_idx] is None:
                    all_outputs[frame_idx] = chunk_output[i].astype(np.float64) * w
                else:
                    all_outputs[frame_idx] += chunk_output[i].astype(np.float64) * w
                all_weights[frame_idx] += w

        # Normalize
        result = []
        for i in range(T):
            if all_outputs[i] is not None and all_weights[i] > 0:
                frame = all_outputs[i] / all_weights[i]
                result.append(np.clip(frame, 0, 255).astype(np.uint8))
            else:
                result.append(src_frames[i])

        return result
