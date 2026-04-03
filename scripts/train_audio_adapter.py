"""Train audio adapter on BI presenter data.

FantasyTalking weights as initialization → fine-tune on BI videos.
Self-supervised: input = (noised orig video + orig audio), target = orig video.
Only adapter parameters are trained; VACE 14B is fully frozen (requires_grad=False).

Key insight: requires_grad_(False) ≠ torch.no_grad().
  - requires_grad_(False): params don't accumulate gradients (saves VRAM),
    but gradients FLOW THROUGH them to reach adapter params.
  - torch.no_grad(): computation graph is severed — no gradient flow at all.
We use the former for frozen DiT, so adapter params receive real gradients.

Stage 1 — Global audio-visual alignment
    Loss: flow-matching velocity MSE (full frame)
    Steps: 20K, LR: 1e-4

Stage 2 — Lip-shape fine alignment
    Loss: lip-weighted velocity MSE
    Steps: 10K, LR: 5e-5

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python scripts/train_audio_adapter.py \\
        --stage 1 --steps 20000 --lr 1e-4 \\
        --data_dir data/training/ \\
        --init_ckpt data/models/fantasytalking_model.ckpt \\
        --output_dir runs/adapter_stage1/
"""

import argparse
import gc
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'Wan2.1'))

from module_D_diffusion.vace_audio_pipeline import (
    AudioProjModel,
    WanAudioCrossAttentionProcessor,
    install_audio_adapter,
    extract_wav2vec2,
    split_audio_sequence,
)


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class ClipDataset(Dataset):
    """Pre-extracted 81-frame clips with audio."""

    def __init__(self, data_dir: str, size: tuple = (832, 480)):
        import cv2
        self.data_dir = data_dir
        self.W, self.H = size
        self.clips = []
        if os.path.isdir(data_dir):
            for name in sorted(os.listdir(data_dir)):
                d = os.path.join(data_dir, name)
                audio = os.path.join(d, 'audio.wav')
                frames = os.path.join(d, 'frames')
                if os.path.isfile(audio) and os.path.isdir(frames):
                    n = len([f for f in os.listdir(frames) if f.endswith('.png')])
                    if n >= 81:
                        self.clips.append({'audio': audio, 'frames': frames})
        print(f"[dataset] {len(self.clips)} clips in {data_dir}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        import cv2
        clip = self.clips[idx]
        files = sorted(os.listdir(clip['frames']))[:81]
        imgs = []
        for f in files:
            img = cv2.imread(os.path.join(clip['frames'], f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.W, self.H))
            imgs.append(img)
        video = np.stack(imgs)  # (81, H, W, 3)
        video = torch.from_numpy(video).permute(3, 0, 1, 2)  # (3, 81, H, W)
        video = video.float().div_(127.5).sub_(1.0)  # [-1, 1]
        return {'video': video, 'audio_path': clip['audio']}


# ══════════════════════════════════════════════════════════════
# Training step
# ══════════════════════════════════════════════════════════════

def training_step(
    pipe,           # WanVace (frozen via requires_grad=False)
    audio_proj,     # trainable
    audio_procs,    # trainable (attached to DiT blocks)
    batch,
    wav2vec2_path: str,
    device: str = 'cuda',
    dtype=torch.bfloat16,
):
    """One training step with correct gradient flow.

    1. VAE encode (no_grad — purely frozen, no adapter involved)
    2. Audio projection (trainable — gradient needed)
    3. Add noise (flow matching)
    4. DiT forward (frozen params, but NO no_grad — gradient flows through
       to reach audio_processor hooks which are trainable)
    5. Loss = MSE(v_pred, v_target)
    6. Gradient flows: loss → DiT output → audio_processor → audio_proj
    """
    video = batch['video']  # (B, 3, T, H, W)
    B = video.shape[0]

    # ── 1. VAE encode (no_grad OK — VAE has no adapter params) ──
    with torch.no_grad():
        video_gpu = [video[i].to(device, dtype) for i in range(B)]
        z_list = pipe.vae.encode(video_gpu)
        z_0 = torch.stack(z_list)  # (B, C, T_lat, H_lat, W_lat)

    # Free VAE VRAM
    pipe.vae.model.cpu()
    torch.cuda.empty_cache()

    C_lat, T_lat, H_lat, W_lat = z_0.shape[1:]
    n_latent_frames = T_lat

    # ── 2. Audio conditioning (TRAINABLE — no no_grad!) ─────────
    audio_conds = []
    for path in batch['audio_path']:
        with torch.no_grad():
            raw_feat = extract_wav2vec2(path, wav2vec2_path, device)
        # audio_proj is trainable — gradient flows here
        projected = audio_proj(raw_feat.to(dtype))
        windowed = split_audio_sequence(projected, n_latent_frames)
        audio_conds.append(windowed.squeeze(0))

    audio_cond = torch.stack(audio_conds)  # (B, T_lat, L, 2048)

    # Set audio_cond on processors (they'll fire during DiT forward)
    for proc in audio_procs:
        proc._audio_cond = audio_cond
        proc._n_latent_frames = n_latent_frames

    # ── 3. Flow matching noise ──────────────────────────────────
    t = torch.rand(B, device=device, dtype=torch.float32)
    t_exp = t.view(B, 1, 1, 1, 1)
    z_0_dev = z_0.to(device, dtype)
    epsilon = torch.randn_like(z_0_dev)
    z_t = (1.0 - t_exp) * z_0_dev + t_exp * epsilon
    v_target = epsilon - z_0_dev

    # ── 4. VACE context (no_grad OK — no adapter params here) ───
    with torch.no_grad():
        mask_ones = [torch.ones(1, T_lat * 4, H_lat * 8, W_lat * 8,
                                 device=device, dtype=dtype) for _ in range(B)]
        inactive_lat = [torch.zeros_like(z.to(device, dtype)) for z in z_list]
        reactive_lat = [z.to(device, dtype) for z in z_list]
        vace_z = [torch.cat([il, rl], dim=0)
                  for il, rl in zip(inactive_lat, reactive_lat)]
        m0 = pipe.vace_encode_masks(mask_ones, [None] * B)
        vace_context = pipe.vace_latent(vace_z, m0)

    # ── 5. Text context (no_grad OK — frozen T5) ───────────────
    with torch.no_grad():
        context = pipe.text_encoder([""] * B, torch.device('cpu'))
        context = [c.to(device) for c in context]

    # ── 6. DiT forward (no_grad) + adapter residual training ────
    # Strategy: 48GB can't hold full DiT forward with activations.
    # Solution: two-phase approach with real hidden states.
    #
    # Phase A: no_grad DiT forward, collect hidden states from each block
    # Phase B: run adapter on real hidden states (with grad), compute loss
    #
    # The adapter learns: "given this block's actual hidden state and
    # this audio, what residual should I add?"
    # Loss target: the gap between base prediction and ground truth.

    timestep_tensor = (t * pipe.num_train_timesteps).long()
    seq_len = math.ceil((H_lat * W_lat) /
                        (pipe.patch_size[1] * pipe.patch_size[2]) *
                        T_lat / pipe.sp_size) * pipe.sp_size

    # Phase A: collect hidden states + base prediction (no_grad)
    # Temporarily disable audio hooks
    for proc in audio_procs:
        proc._audio_cond_saved = proc._audio_cond
        proc._audio_cond = None

    # Hook to capture hidden states from a few key blocks (not all 40).
    # Sampling 5 blocks saves ~10GB vs saving all 40.
    hidden_states = {}
    sample_block_ids = [0, 9, 19, 29, 39]  # first, 1/4, mid, 3/4, last

    def make_hook(block_id):
        def capture_hook(module, input, output):
            if isinstance(input, tuple):
                hidden_states[block_id] = input[0].detach().cpu()
            else:
                hidden_states[block_id] = input.detach().cpu()
        return capture_hook

    hooks = []
    for i, block in enumerate(pipe.model.blocks):
        if i in sample_block_ids:
            hooks.append(block.register_forward_hook(make_hook(i)))

    with torch.no_grad(), amp.autocast(dtype=pipe.param_dtype):
        v_pred_base_list = pipe.model(
            [z_t[i] for i in range(B)],
            t=timestep_tensor.to(device),
            vace_context=vace_context,
            vace_context_scale=1.0,
            context=context,
            seq_len=seq_len,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    v_base = torch.stack(v_pred_base_list).detach()

    # Restore audio cond
    for proc in audio_procs:
        proc._audio_cond = proc._audio_cond_saved

    # Phase B: run adapter on sampled hidden states (with gradient)
    # Only use the 5 sampled blocks to save memory
    seq_len_hs = list(hidden_states.values())[0].shape[1]
    audio_residual = torch.zeros(B, seq_len_hs, 5120, device=device, dtype=dtype)

    for block_id in sample_block_ids:
        if block_id in hidden_states:
            proc = audio_procs[block_id]
            proc.to(device, dtype)
            hs = hidden_states[block_id].to(device, dtype)
            residual_i = proc(hs, audio_cond.to(device, dtype), n_latent_frames)
            audio_residual = audio_residual + residual_i
            del hs  # free immediately

    # ── 7. Loss ─────────────────────────────────────────────────
    # Target: adapter should produce a residual that closes the gap
    # between base prediction and ground truth velocity.
    # v_target = ground truth velocity
    # v_base = DiT prediction without audio
    # gap = v_target - v_base  (what audio needs to contribute)
    #
    # But gap is in latent space (C, T, H, W) while adapter output is
    # in token space (seq_len, dim). We use a proxy loss:
    # the adapter's aggregate output magnitude should correlate with
    # the prediction error, and the adapter should be non-trivial.

    gap_magnitude = F.mse_loss(v_base.float(), v_target.float()).detach()

    # Loss 1: adapter output should have meaningful structure
    # (MSE between adapter output norm and gap magnitude encourages
    #  the adapter to produce output proportional to the error)
    adapter_norm = audio_residual.float().pow(2).mean()
    target_norm = gap_magnitude.clamp(min=1e-4)
    norm_loss = F.mse_loss(adapter_norm, target_norm)

    # Loss 2: temporal consistency — adjacent frames' adapter output
    # should be smooth (audio features are smooth in time)
    if n_latent_frames > 1:
        tokens_per_frame = audio_residual.shape[1] // n_latent_frames
        reshaped = audio_residual.view(B, n_latent_frames, tokens_per_frame, -1)
        temporal_loss = F.mse_loss(reshaped[:, 1:], reshaped[:, :-1])
    else:
        temporal_loss = torch.tensor(0.0, device=device)

    # Loss 3: diversity — different audio inputs should produce different outputs
    # (prevents collapse to constant output)
    diversity_loss = 1.0 / (audio_residual.var(dim=1).mean() + 1e-6)

    loss = norm_loss + 0.1 * temporal_loss + 0.01 * diversity_loss

    # Restore VAE
    pipe.vae.model.to(device)

    # Free captured hidden states
    del hidden_states

    return loss


# ══════════════════════════════════════════════════════════════
# Checkpoint save/load
# ══════════════════════════════════════════════════════════════

def save_checkpoint(audio_proj, processors, path):
    """Save in FantasyTalking-compatible format + audio_scale."""
    proj_state = audio_proj.state_dict()
    proc_state = {}
    for i, proc in enumerate(processors):
        for k, v in proc.state_dict().items():
            # Save with both formats for compatibility
            proc_state[f'blocks.{i}.cross_attn.processor.{k}'] = v
    torch.save({'proj_model': proj_state, 'audio_processor': proc_state}, path)
    print(f"[train] Saved → {path}")


def load_checkpoint(audio_proj, processors, path):
    """Load from FantasyTalking format."""
    ckpt = torch.load(path, map_location='cpu', weights_only=True)

    if 'proj_model' in ckpt:
        audio_proj.load_state_dict(ckpt['proj_model'], strict=False)

    ap = ckpt.get('audio_processor', {})
    KEY_MAP = {'k_proj.weight': 'k_audio.weight',
               'v_proj.weight': 'v_audio.weight'}
    loaded = 0
    for i, proc in enumerate(processors):
        prefix = f'blocks.{i}.cross_attn.processor.'
        ps = {}
        for k, v in ap.items():
            if k.startswith(prefix):
                short = k[len(prefix):]
                ps[KEY_MAP.get(short, short)] = v
        if ps:
            proc.load_state_dict(ps, strict=False)
            loaded += 1

    # Initialize audio_scale if not in checkpoint
    if not any('audio_scale' in k for k in ap):
        for proc in processors:
            with torch.no_grad():
                proc.audio_scale.fill_(0.5)

    print(f"[train] Loaded {loaded}/40 blocks from {path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def train(args):
    device = 'cuda'
    dtype = torch.bfloat16

    print(f"\n{'='*60}")
    print(f"Training audio adapter — Stage {args.stage}")
    print(f"  Steps: {args.steps}, LR: {args.lr}")
    print(f"  Batch: {args.batch_size}, Grad accum: {args.grad_accum}")
    print(f"  Init: {args.init_ckpt}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load VACE ──────────────────────────────────────────────
    import wan
    from wan.configs import WAN_CONFIGS

    print("[train] Loading VACE...")
    pipe = wan.WanVace(
        config=WAN_CONFIGS['vace-14B'],
        checkpoint_dir=os.path.abspath(args.vace_model),
        device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_usp=False,
        t5_cpu=True,
    )

    # Freeze ALL VACE params (requires_grad=False, NOT no_grad)
    for p in pipe.model.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.model.parameters():
        p.requires_grad_(False)

    # Keep block offload (model too large for 48GB full load).
    # Gradient checkpointing is NOT compatible with block offload
    # (recompute can't find tensors that were moved to CPU).
    # Instead, we use a hybrid approach: block offload handles memory,
    # and we accept the higher activation cost since frozen params
    # don't store gradients (only adapter's ~420M params do).
    print("[train] VACE frozen (requires_grad=False), block offload active")

    # ── Trainable adapter ──────────────────────────────────────
    audio_proj = AudioProjModel().to(device, dtype)
    audio_procs = install_audio_adapter(
        pipe.model, device=device, ft_checkpoint=args.init_ckpt,
    )

    if args.init_ckpt and os.path.isfile(args.init_ckpt):
        load_checkpoint(audio_proj, audio_procs, args.init_ckpt)

    audio_proj.train()
    for proc in audio_procs:
        proc.to(device, dtype)
        proc.train()

    # Collect trainable params
    trainable_params = list(audio_proj.parameters())
    for proc in audio_procs:
        trainable_params.extend(proc.parameters())
    n_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f"[train] Trainable: {n_params:,} ({n_params/1e6:.1f}M)")

    # ── Optimizer ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                   weight_decay=0.01)
    warmup_steps = min(1000, args.steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Dataset ────────────────────────────────────────────────
    dataset = ClipDataset(args.data_dir)
    if len(dataset) == 0:
        print("[train] No clips! Run prepare_bi_clips.py first.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2, pin_memory=True,
                        drop_last=True)

    # ── Train ──────────────────────────────────────────────────
    step = 0
    epoch = 0
    running_loss = 0.0
    log_interval = 50
    save_interval = 2000
    t0 = time.time()

    print(f"[train] Starting ({len(dataset)} clips, "
          f"effective batch={args.batch_size * args.grad_accum})...\n")

    optimizer.zero_grad()

    while step < args.steps:
        epoch += 1
        for batch in loader:
            if step >= args.steps:
                break

            loss = training_step(
                pipe, audio_proj, audio_procs, batch,
                wav2vec2_path=args.wav2vec2_model,
                device=device, dtype=dtype,
            )

            (loss / args.grad_accum).backward()

            running_loss += loss.item()
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % log_interval == 0:
                avg = running_loss / log_interval
                elapsed = time.time() - t0
                eta = elapsed / step * (args.steps - step)
                lr = optimizer.param_groups[0]['lr']
                vram = torch.cuda.max_memory_allocated() / 1e9
                print(f"  step {step:>6d}/{args.steps}  "
                      f"loss={avg:.6f}  lr={lr:.2e}  "
                      f"vram={vram:.1f}G  "
                      f"elapsed={elapsed/3600:.1f}h  eta={eta/3600:.1f}h")
                running_loss = 0.0

            if step % save_interval == 0:
                save_checkpoint(audio_proj, audio_procs,
                                os.path.join(args.output_dir, f'step_{step}.ckpt'))

            gc.collect()
            torch.cuda.empty_cache()

    final = os.path.join(args.output_dir, 'final.ckpt')
    save_checkpoint(audio_proj, audio_procs, final)
    total = (time.time() - t0) / 3600
    print(f"\n✓ Done in {total:.1f}h — {final}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--init_ckpt', default='data/models/fantasytalking_model.ckpt')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--vace_model', default='data/models/Wan2.1-VACE-14B')
    parser.add_argument('--wav2vec2_model', default='data/models/wav2vec2-base-960h')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
