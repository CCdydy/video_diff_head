"""Differential Diffusion: per-pixel strength-controlled noise for VACE.

Mode 1: 唇 0.55 / 脸 0.20 / 其他 0.00
Mode 2: 唇 0.55 / 脸 0.35 / 上半身 0.15 / 背景 0.00

Wan2.1 flow-matching: z_t = (1-t)*z_0 + t*ε, where t = strength per pixel.
Training-free — works at inference time only.

Reference: Differential Diffusion (arXiv:2306.00950, merged into diffusers).
"""

import numpy as np
import torch
import torch.nn.functional as F


def strength_map_to_latent(
    strength_map: np.ndarray,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """Downsample pixel-space strength map to latent resolution (÷8).

    Bilinear interpolation naturally softens sharp mask edges.

    Returns (1, 1, latent_h, latent_w) float tensor.
    """
    t = torch.from_numpy(strength_map).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(latent_h, latent_w), mode='bilinear',
                      align_corners=False)
    return t.clamp(0.0, 1.0)


def apply_differential_noise(
    z_orig: torch.Tensor,
    strength_latent: torch.Tensor,
    noise: torch.Tensor = None,
) -> torch.Tensor:
    """Apply per-pixel differential noise to original latents.

    Wan2.1 flow-matching interpolation (linear):
        z_noisy = (1 - s) * z_orig + s * noise
    where s = per-pixel strength from strength_map.

    Args:
        z_orig: (B, C, T_lat, H_lat, W_lat) VAE-encoded original video.
        strength_latent: (1, 1, H_lat, W_lat) per-pixel strength [0,1].
        noise: optional, same shape as z_orig.

    Returns:
        z_noisy: differentially noised latents.
    """
    if noise is None:
        noise = torch.randn_like(z_orig)

    s = strength_latent.to(z_orig.device, z_orig.dtype)
    # Broadcast: (1, 1, 1, H, W) over (B, C, T, H, W)
    s = s.unsqueeze(2)

    z_noisy = (1.0 - s) * z_orig + s * noise
    return z_noisy


def blended_denoise_step(
    z_denoised: torch.Tensor,
    z_orig: torch.Tensor,
    strength_latent: torch.Tensor,
    current_step: int,
    total_steps: int,
) -> torch.Tensor:
    """Blended Latent Diffusion step: progressively lock low-strength regions.

    At step t/T:
      - pixels with strength < t/T → use z_orig (editing done)
      - pixels with strength >= t/T → use z_denoised (still editing)
    """
    progress = current_step / total_steps
    s = strength_latent.to(z_denoised.device, z_denoised.dtype).unsqueeze(2)
    edit_mask = (s >= progress).float()
    return z_denoised * edit_mask + z_orig * (1.0 - edit_mask)
