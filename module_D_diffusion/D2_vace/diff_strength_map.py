"""Differential Diffusion: region-selective noise injection for VACE Mode 2.

Implements per-pixel strength-controlled noising for masked V2V editing.
Each pixel has its own editing strength [0, 1]:
  lips     → 0.50 (maximum edit, change lip shape)
  face     → 0.35 (expression/head motion)
  shoulder → 0.15 (slight follow-through)
  bg/body  → 0.00 (completely preserved)

At each denoising step, the strength map is binarized with a decreasing
threshold, enabling spatially graded control. This is training-free.

Reference: Differential Diffusion (github.com/exx8/differential-diffusion),
merged into HuggingFace diffusers.
"""

import numpy as np
import torch
import torch.nn.functional as F


def build_strength_map_latent(
    strength_map: np.ndarray,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """Downsample pixel-space strength map to latent resolution.

    Wan2.1 VAE compresses spatial 8×. The soft downsampling naturally
    diffuses sharp mask edges into smooth latent-space transitions.

    Args:
        strength_map: (H, W) float32 in [0, 1], pixel resolution.
        latent_h, latent_w: target latent spatial dimensions.

    Returns:
        (1, 1, latent_h, latent_w) float tensor on CPU.
    """
    t = torch.from_numpy(strength_map).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(latent_h, latent_w), mode='bilinear',
                      align_corners=False)
    return t.clamp(0.0, 1.0)


def apply_differential_noise(
    z_orig: torch.Tensor,
    strength_map_latent: torch.Tensor,
    noise: torch.Tensor = None,
    max_timestep: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-pixel differential noise to original latents.

    For Wan2.1's flow-matching, the noising interpolation is linear:
        z_noisy = (1 - t) * z_orig + t * noise
    where t = strength_map * t_max / T

    Args:
        z_orig: (B, C, T_lat, H_lat, W_lat) original VAE-encoded latents.
        strength_map_latent: (1, 1, H_lat, W_lat) per-pixel strength [0, 1].
        noise: optional pre-generated noise, same shape as z_orig.
        max_timestep: maximum diffusion timestep (typically 1000).

    Returns:
        z_noisy: noised latents with per-pixel strength.
        t_starts: (1, 1, H_lat, W_lat) per-pixel starting timestep.
    """
    if noise is None:
        noise = torch.randn_like(z_orig)

    # Expand strength map to match z_orig: (B, C, T_lat, H_lat, W_lat)
    # strength_map applies uniformly across batch, channel, and temporal dims
    smap = strength_map_latent.to(z_orig.device, z_orig.dtype)
    # Broadcast: (1, 1, 1, H_lat, W_lat) over (B, C, T_lat, H_lat, W_lat)
    smap = smap.unsqueeze(2)  # (1, 1, 1, H_lat, W_lat)

    # Per-pixel interpolation factor (flow matching: linear)
    t = smap  # in [0, 1], where 0 = no noise, 1 = full noise

    z_noisy = (1.0 - t) * z_orig + t * noise

    # t_starts for the scheduler: pixels with strength=0 start at step 0 (skip),
    # pixels with strength=0.5 start at step 500, etc.
    t_starts = (smap.squeeze(2) * max_timestep).long()

    return z_noisy, t_starts


def apply_blended_denoise_step(
    z_denoised: torch.Tensor,
    z_orig: torch.Tensor,
    strength_map_latent: torch.Tensor,
    current_step: int,
    total_steps: int,
) -> torch.Tensor:
    """Blended Latent Diffusion: at each step, blend denoised with original.

    For pixels where current progress > their strength, use original latent.
    For pixels still being edited, use denoised result.

    This is the step-wise blending from Avrahami et al. (ACM TOG 2023).
    """
    progress = current_step / total_steps  # 0→1 as denoising proceeds
    smap = strength_map_latent.to(z_denoised.device, z_denoised.dtype)
    smap = smap.unsqueeze(2)  # (1, 1, 1, H, W)

    # Binary decision: if pixel's strength < progress, it's "done" → use original
    # This threshold decreases over time, progressively locking in low-strength regions
    blend_mask = (smap >= progress).float()

    return z_denoised * blend_mask + z_orig * (1.0 - blend_mask)


if __name__ == '__main__':
    # Quick test with synthetic data
    B, C, T, H, W = 1, 16, 21, 60, 104
    z = torch.randn(B, C, T, H, W)
    smap = torch.rand(1, 1, H, W) * 0.5  # max 0.5 strength

    z_noisy, t_starts = apply_differential_noise(z, smap)
    print(f"z_noisy shape: {z_noisy.shape}")
    print(f"t_starts range: [{t_starts.min()}, {t_starts.max()}]")
    print(f"Background pixels (strength=0) preserved: "
          f"{(z_noisy[0, 0, 0] == z[0, 0, 0]).float().mean():.1%}")
