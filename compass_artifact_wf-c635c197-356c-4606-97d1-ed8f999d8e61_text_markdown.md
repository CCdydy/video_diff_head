# Audio-driven upper-body re-animation via Wan2.1 video diffusion

**The most viable architecture for translating presenter videos combines SDEdit-style masked denoising on Wan2.1's DiT backbone with audio cross-attention adapted from FantasyTalking or OmniAvatar's pixel-wise injection.** Both FantasyTalking and OmniAvatar use wav2vec2-base-960h for audio encoding but differ fundamentally in injection strategy—cross-attention vs. additive embedding—and in their choice of I2V vs. T2V backbone. A practical pipeline chains SAM2 mask propagation → masked SDEdit with audio conditioning → composite back, processing 81-frame (5-second) chunks with 8–16 frame overlaps. On an RTX 5090 (32GB), Wan2.1-14B in FP8 with block offloading is feasible at 480P, yielding roughly **0.6–1.4 minutes of compute per second of output video** with aggressive optimization.

---

## FantasyTalking injects audio via zero-initialized cross-attention into every DiT block

**Repository:** `github.com/Fantasy-AMAP/fantasy-talking` (~1.6k stars) | **Paper:** arXiv 2504.04842 (ACM MM 2025)
**Base model:** Wan2.1-I2V-14B-720P (Image-to-Video, 14B parameters)

The audio pipeline starts with **facebook/wav2vec2-base-960h**, producing 768-dimensional features at ~50 vectors/second from 16kHz audio. These features pass through a deliberately minimal `AudioProjModel`—a single `Linear(768→2048)` followed by `LayerNorm`—defined in `model.py`. The projected features are then temporally segmented by `split_audio_sequence()`, which maps continuous audio to per-latent-frame windows. Since Wan2.1's causal 3D VAE compresses every 4 video frames into 1 latent frame, 81 video frames become 21 latent frames; the audio sequence is split into 21 overlapping windows of ~17 tokens each (8 base + 4 expand on each side).

Audio enters the DiT through `WanCrossAttentionProcessor`, installed in **all 40 transformer blocks** via `set_audio_processor()`. Each processor contains learned `k_proj` and `v_proj` projections (`Linear(2048→5120)`, zero-initialized for stable training). The forward pass computes three separate attention outputs that are summed:

```
x = text_attn + img_attn + audio_attn × audio_scale
```

The 4D audio tensor `[B, 21, L, C]` enables **per-frame audio attention**: queries are reshaped to `[B×21, ...]`, attention is computed independently per latent frame, then reshaped back. This is the primary mode, ensuring each generated frame attends only to its temporally aligned audio window.

**Identity preservation** uses a facial-focused cross-attention module rather than a ReferenceNet. The CLIP image encoder (open-clip-xlm-roberta-large-vit-huge-14) produces 257 tokens for the reference image, processed through dedicated `k_img`/`v_img` projections already present in Wan2.1's `WanI2VCrossAttention`. The paper explicitly states this approach "concentrates on modeling facial regions" while allowing flexible motion generation.

**Training uses a dual-stage approach (DAVA):** Stage 1 runs ~80,000 steps for clip-level audio-visual alignment across the entire scene. Stage 2 runs ~20,000 steps with a **lip-tracing mask** that focuses loss computation on the mouth region for precise synchronization. Training ran on 64× A100 GPUs with learning rate 1e-4 and 0.1 dropout probability for each condition (image, audio, text) independently.

**Critical limitation: FantasyTalking is strictly I2V.** The inference script (`infer.py`) takes `--image_path` (single portrait), `--audio_path` (WAV), and `--prompt` (text). There is no video-to-video mode. Output is 81 frames at 512×512, 23 FPS, with 30 denoising steps. A follow-up paper, **FantasyTalking2** (arXiv 2508.11255, AAAI 2026), adds timestep-layer adaptive preference optimization via LoRA experts (rank 128) but remains I2V.

---

## OmniAvatar replaces cross-attention with pixel-wise additive audio embedding

**Repository:** `github.com/Omni-Avatar/OmniAvatar` (~1.8k stars) | **Paper:** arXiv 2506.18866 (ZJU + Alibaba)
**Base model:** Wan2.1-**T2V**-14B (Text-to-Video, not I2V)

The architectural choice to use T2V rather than I2V is deliberate. OmniAvatar's **Audio Pack module** works as follows: wav2vec2-base-960h extracts 768-dim features, which are padded to length T+3 (matching VAE requirements), grouped at compression rate 4 (matching the VAE's temporal stride), rearranged so 4 consecutive audio vectors are packed together, then projected via a **linear layer** to the video latent dimension. The result is added **element-wise** at the pixel level:

```
z'_{i,t} = z_{i,t} + P_a(z_a)
```

This additive injection occurs at **DiT blocks between layer 2 and the middle layer** (~layer 20), with **unshared projection weights per layer**. Audio is deliberately excluded from the first and final layers to prevent it from dominating the latent representation. This multi-hierarchical injection contrasts sharply with FantasyTalking's uniform injection across all 40 blocks.

Since T2V has no native image cross-attention, identity is preserved by encoding the reference frame through the VAE, repeating it temporally, and **concatenating it channel-wise** with the video latent at each timestep. This is simpler than FantasyTalking's CLIP-based cross-attention but works because LoRA adaptation (rank=128, alpha=64, applied to Q/K/V projections and FFN layers) teaches the model to extract identity features from the concatenated reference.

| Aspect | FantasyTalking | OmniAvatar |
|--------|---------------|------------|
| **Backbone** | Wan2.1-I2V-14B | Wan2.1-T2V-14B |
| **Audio injection** | Cross-attention (all 40 blocks) | Pixel-wise addition (layers 2–20) |
| **Audio projections** | Linear(768→2048) + per-block k/v(2048→5120) | Grouped linear per-layer (unshared) |
| **Identity method** | CLIP 257 tokens via cross-attention | VAE latent concatenation |
| **Training** | Full adapter fine-tuning (64× A100) | LoRA rank=128 (64× A100) |
| **Training data** | Undisclosed | AVSpeech filtered: 774K clips, ~1320 hours |
| **Benchmark FVD** | 780 | **664** (SOTA) |
| **Sync-C** | 3.14 | **7.12** (SOTA) |

OmniAvatar **can condition on video frames** through its prefix latent mechanism: during inference, the last 13 frames from a previous batch serve as conditioning for the next batch, enabling temporal continuity for long videos. However, it remains fundamentally a generation model, not a video editor—it produces new video conditioned on reference image + audio + text, rather than editing existing footage.

**Memory benchmarks** (A800 GPU, 14B BF16): 36GB with unlimited persistent params (16.0s/it), 21GB with 7B persistent params (19.4s/it), or **8GB with full offloading** (22.1s/it). The input format is `[prompt]@@[img_path]@@[audio_path]` in a text file, processed by `scripts/inference.py`.

---

## Masked SDEdit enables region-selective re-animation with per-pixel strength control

SDEdit applied to video DiTs works by encoding all frames into latent space, adding noise at strength `s = t_start/T`, then jointly denoising with new audio conditioning. For Wan2.1's flow-matching framework, the interpolation is linear: `x_t = (1−t)·x_0 + t·ε`. The strength parameter directly controls the edit-fidelity tradeoff:

| Edit target | Strength (s) | t_start (1000-step) | Behavior |
|-------------|-------------|---------------------|----------|
| Lip-only | 0.15–0.30 | 150–300 | Subtle texture/shape changes; identity fully preserved |
| Full face | 0.35–0.55 | 350–550 | Expression, gaze, mouth rewritten; broad identity preserved |
| Head + upper body | 0.55–0.80 | 550–800 | Pose, head angle, shoulder position change |

For **region-masked editing**, five approaches exist with increasing sophistication:

**Blended Latent Diffusion** (Avrahami et al., ACM TOG 2023) is the foundation. At each denoising step: denoise to get `z_fg` (edited foreground), add noise to original latent at current level to get `z_bg` (background), then blend via `z_t = z_fg ⊙ mask + z_bg ⊙ (1−mask)`. Progressive mask shrinking (dilated early → exact late) prevents thin-mask artifacts.

**Differential Diffusion** (`github.com/exx8/differential-diffusion`, merged into HuggingFace diffusers) is the most directly applicable approach. It accepts a **grayscale change map** where each pixel specifies its own editing strength 0–1. At each step, the change map is binarized with a decreasing threshold, enabling spatially graded control: lips at 0.5, cheeks at 0.3, shoulders at 0.15, background at 0.0. This generalizes SDEdit from scalar to spatial and is training-free.

**VACE** (`ali-vilab/VACE`, integrated into Wan2.1) is the native solution. It adds a **Video Condition Unit** and **Context Adapter** with parallel `vace_blocks` to the DiT, supporting masked video-to-video editing with `src_video`, `src_mask`, and `src_ref_images` inputs. Masks are binary (white=edit, black=preserve) with "Grow Mask With Blur" for feathered boundaries. Available as `Wan2.1-VACE-14B`.

**NC-SDEdit** (`github.com/yangqy1110/NC-SDEdit`, ECCV 2024) addresses structure preservation by replacing low-frequency noise components via FFT calibration. The calibrated noise preserves coarse spatial layout (positions, colors) while allowing high-frequency regeneration. Threshold frequency ν=0.5 is typical.

For boundary handling between edited face and unedited body, **feathered masks in latent space** (Gaussian blur with 8–16px radius before downsampling to latent resolution) combined with step-wise blending produces smooth transitions. The Wan2.1 VAE's 8× spatial compression naturally softens pixel-space boundaries in latent space.

---

## Wan2.1's DiT architecture reveals specific insertion points for audio conditioning

The Wan2.1 14B model comprises **40 transformer blocks** (`WanAttentionBlock`), each containing:

1. **WanSelfAttention** — Full spatiotemporal self-attention with 3D Rotary Position Embeddings (RoPE), 40 heads, dim=5120
2. **Cross-attention** — `WanT2VCrossAttention` (T2V) or `WanI2VCrossAttention` (I2V) processing umT5-XXL text embeddings (4096-dim, max 512 tokens) and optionally CLIP ViT-H image embeddings
3. **FFN** — Two-layer MLP with ffn_dim=13824

**I2V conditioning uses channel concatenation**: the reference image is VAE-encoded and concatenated along the channel dimension, changing `in_dim` from 16 (T2V) to **36** (16 latent + 16 image latent + 4 mask channels). CLIP embeddings are additionally injected via cross-attention for high-level semantics. A binary mask indicates which latent frames are "given" (reference) vs. "to generate."

**The 3D Causal VAE** compresses at stride `(4, 8, 8)` — temporal 4×, spatial 8×8 — producing 16-channel latents. For an 81-frame 832×480 video: latent shape is `[1, 16, 21, 60, 104]`, yielding **32,760 tokens** after patching with size `(1, 2, 2)`. Full attention (no windowing by default) means O(N²) scaling, though NABLA sparse attention with Sliding Tile Attention achieves **2.7× speedup** with negligible quality loss.

**Extending I2V to video-to-video** can be done through VACE (official), DiffSynth-Studio's vid2vid pipeline, or by encoding all source frames through the VAE and using SDEdit. The frame count must satisfy `(N−1) mod 4 = 0` due to temporal compression (valid lengths: 17, 33, 49, 65, 81, 97, 113, 129, 161).

**Timestep conditioning** uses a shared MLP (Linear→SiLU→Linear) that produces 6 modulation parameters (scale/shift for pre-norm, post-norm, and gate) with per-block learned biases, following the adaptive layer norm approach.

---

## Temporal consistency relies on TREPA, overlapping chunks, and inherent 3D attention

**TREPA (Temporal REPresentation Alignment)**, introduced in LatentSync, uses a frozen **VideoMAE-v2** encoder to extract temporal representations from both generated and ground-truth frame sequences, then applies MSE loss between them: `L_TREPA = MSE(T(gen_seq), T(gt_seq))`. Unlike per-frame losses (LPIPS), TREPA captures inter-frame temporal correlations. LatentSync also uses mixed noise: `ε_f = ε_shared + ε_f_ind`, where shared noise across all 16 frames provides baseline temporal coherence. TREPA was shown to outperform temporal self-attention layers, which actually harmed lip-sync accuracy by distributing gradients across temporal parameters.

For **long video generation**, both FantasyTalking and OmniAvatar process 81-frame chunks (5 seconds at 16 FPS). OmniAvatar's overlap strategy uses the **last 13 frames** from the previous batch as prefix latents for the next batch, with the reference frame embedding repeated throughout for identity anchoring. The state-of-the-art for chunk blending comes from unified long video inpainting work (arXiv 2511.03272), which uses **Hamming window weighting** during co-denoising of overlapping temporal windows, trained on Wan2.1 with LoRA.

Joint video denoising with 3D DiTs (Wan2.1, HunyuanVideo) **inherently enforces temporal consistency** because every spatiotemporal token attends to all others. Frame-by-frame SDEdit fails catastrophically—independent noise per frame produces flickering—but joint denoising over the full latent tensor naturally smooths temporal transitions. At medium noise levels (s ≈ 0.4–0.6), the model preserves input motion patterns while allowing appearance changes. Above s > 0.7, temporal structure begins breaking down.

**Boundary handling between edited face and unedited body** uses: soft masking with Gaussian blur in latent space, step-wise blending (Blended Latent Diffusion), Poisson blending at compositing boundaries, and FantasyTalking's lip-tracing masks that follow mouth motion to minimize the edited region. The key insight is that the VAE's 8× spatial downsampling naturally diffuses sharp pixel-space mask edges into smooth latent-space transitions.

---

## The practical V2V pipeline chains SAM2, masked diffusion, and ProPainter compositing

**ProPainter** (`github.com/sczhou/ProPainter`, ICCV 2023, 6.6k stars) provides the inpainting backbone with three components: Recurrent Flow Completion (RFC) for optical flow, Dual-Domain Propagation (DDP) combining image-domain and feature-domain warping, and Mask-Guided Sparse Video Transformer (MSVT) for spatiotemporal refinement. At 720×480 with fp16, it requires **~8GB VRAM** for 80 frames. The `--subvideo_length` parameter (default 80) decouples memory from total video length. Processing speed is ~80× faster than previous methods.

**SAM2** (`github.com/facebookresearch/sam2`) generates temporally consistent masks via its streaming memory architecture. Initialize with a point/bbox on the face in frame 0, then `propagate_in_video()` produces per-frame masks. The `sam2_hiera_large` (224M params) runs at ~44 FPS. Bidirectional propagation improves consistency; the occlusion head prevents mask drift during head turns.

**Recommended end-to-end pipeline:**

1. **Preprocessing**: Extract frames at 16 FPS → face detection (RetinaFace) → SAM2 mask propagation → dilate masks 4–8px
2. **Per 81-frame chunk**: Encode original frames through Wan2.1 VAE → apply Differential Diffusion with per-region strength map (lips 0.4, face 0.3, shoulders 0.15, background 0.0) → denoise with audio cross-attention from FantasyTalking adapter → decode
3. **Compositing**: Alpha-blend diffusion output using SAM2 masks with feathered edges → color harmonization via histogram matching → ProPainter for boundary artifact cleanup
4. **Temporal stitching**: Hamming window blending across 8–16 frame overlaps between chunks

**RTX 5090 (32GB) feasibility**: Wan2.1-14B in BF16 with `--offload_model` and `--t5_cpu` fits at 480P/81 frames using ~24–28GB peak VRAM. FP8 quantization (`torch.float8_e4m3fn`) halves model memory to ~14GB for weights, making 720P feasible. WanGP achieves 14B non-quantized at 480P/81 frames in just **8GB** through aggressive async offloading. FantasyTalking's `num_persistent_param_in_dit=7B` mode uses 20GB at 32.8s/iteration on A100; on RTX 5090 with ~10–12B persistent params, expect ~25–30GB usage.

**Throughput estimates** (RTX 5090, 480P, FP8 + TeaCache + reduced steps):

| Stage | Time per 5s chunk |
|-------|-------------------|
| SAM2 mask propagation | ~3s |
| Diffusion (25 steps, FP8, TeaCache) | ~3–7 min |
| ProPainter boundary cleanup | ~10s |
| Compositing | ~3s |
| **Total** | **~4–8 min per 5s** |

---

## Conclusion: the optimal architecture for audio-driven video translation

The most promising approach combines VACE's native masked video-to-video capability with FantasyTalking's audio cross-attention mechanism. VACE already supports `src_video` + `src_mask` + `src_ref_images` inputs on Wan2.1-14B, providing the V2V foundation that neither FantasyTalking nor OmniAvatar natively offer. Grafting FantasyTalking's `WanCrossAttentionProcessor` (with its zero-initialized audio k/v projections across all 40 blocks and per-frame audio windowing) onto VACE's masked editing pipeline would yield audio-conditioned masked V2V in a single model.

OmniAvatar's pixel-wise additive injection is computationally cheaper and achieves better benchmark scores (FVD 664 vs. 780, Sync-C 7.12 vs. 3.14), suggesting the additive Audio Pack approach may be superior for lip-sync quality. Its LoRA-based training (rank 128) is also more parameter-efficient. However, OmniAvatar uses the T2V backbone, which lacks the native image conditioning channels that VACE and FantasyTalking's I2V backbone provide.

**The key unsolved challenge remains transitioning from I2V generation to true V2V editing**—where the original video's non-face regions are maximally preserved. Differential Diffusion's per-pixel strength maps, combined with VACE's mask-aware context adapter, offer the most direct path. EditYourself (Pipio AI, 2026) has demonstrated this is achievable on LTX-Video's DiT, using windowed audio conditioning and forward-backward RoPE for identity stability. Replicating that approach on Wan2.1's more capable backbone, leveraging the existing FantasyTalking/OmniAvatar audio adapters, represents the frontier implementation target.
