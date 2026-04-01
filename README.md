# Video Translation Pipeline

> Audio-driven video dubbing pipeline for product review presenters.  
> Chinese → Japanese (and multilingual) face-synchronized video translation.

---

## Overview

This pipeline translates presenter videos by replacing both speech and facial animation. It supports three rendering backends with increasing capability:

| Mode | Rendering Backend | Head Motion | Lip Sync | Upper Body | Identity Fidelity |
|------|-------------------|-------------|----------|------------|-------------------|
| **A** | 3DMM → GaussianAvatars | ✅ Full control | ✅ | ❌ | ✅ High (personalized) |
| **B** | 3DMM → LatentSync (Hybrid) | ✅ Pose warp | ✅ Diffusion | ❌ | ✅ High |
| **C** | EchoMimic V2 / FantasyTalking | ✅ Audio-driven | ✅ Diffusion | ✅ | ⚡ One-shot |

**Recommended for production**: Mode B (best balance of control + quality).  
**Recommended for evaluation/comparison**: run all three, select by SyncNet score.

---

## Architecture

### Mode A — 3DMM + GaussianAvatars (Original)

```text
┌──────────────────────────────────────────────────────────────────────┐
│  Offline (per presenter, one-time)                                   │
│                                                                      │
│  Video corpus ──► VHAP → FLAME params                                │
│                      ├─► GaussianAvatars training (3DGS avatar)      │
│                      └─► Audio→3DMM fine-tuning data                 │
│  Voice samples ──► CosyVoice voice prompt extraction                 │
└──────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Online (per new video)                                              │
│                                                                      │
│  Input video                                                         │
│    ├─ Audio stream:                                                  │
│    │    FunASR → translate → CosyVoice TTS → new_audio               │
│    │                                                                 │
│    └─ Visual stream:                                                 │
│         InsightFace detect → SAM2 mask                               │
│         ProPainter → clean_bg                                   ①    │
│                                                                      │
│  new_audio ──► Audio→3DMM model → exp + pose + jaw                   │
│                 (Keyness mechanism: per-frame motion amplitude)       │
│                                   │                                  │
│              FLAME params ──────── GaussianAvatars → rendered_face   │
│                                   │                                  │
│  Three-layer compositing:                                            │
│    Layer 0: clean_bg              ──── background                    │
│    Layer 1: original body         ──── torso / hands (no head)       │
│    Layer 2: rendered_face RGBA    ──── new face + Poisson blend       │
│                                   │                                  │
│              Kalman smoothing → CodeFormer SR → SyncNet QA           │
│                                   │                                  │
│                            FFmpeg mux → output                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Mode B — 3DMM + LatentSync (Hybrid, Recommended)

The Audio→3DMM model's **Keyness mechanism** drives head pose warping; LatentSync handles
diffusion-based lip generation on the warped frames. GaussianAvatars is not required.

```text
┌──────────────────────────────────────────────────────────────────────┐
│  Offline (per presenter, one-time)                                   │
│                                                                      │
│  Video corpus ──► VHAP → FLAME params                                │
│                      └─► Audio→3DMM fine-tuning data                 │
│  Voice samples ──► CosyVoice voice prompt extraction                 │
│                                                                      │
│  ★ GaussianAvatars training NOT required in this mode                │
└──────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Online (per new video)                                              │
│                                                                      │
│  Input video                                                         │
│    ├─ Audio stream:                                                  │
│    │    FunASR → translate → CosyVoice TTS → new_audio               │
│    │                                                                 │
│    └─ Visual stream:                                                 │
│         InsightFace detect → SAM2 mask                               │
│         ProPainter → clean_bg                                   ①    │
│                                                                      │
│  new_audio ──► Audio→3DMM model → head_pose (6D)                     │
│                 (exp params discarded; pose used for warping only)    │
│                                   │                                  │
│              head_pose ──────────► FLAME-guided affine warp          │
│                          original_body_frames → posed_frames    ②    │
│                                                                      │
│  posed_frames ② + new_audio ──► LatentSync                           │
│                                  (audio-conditioned LDM lip sync)    │
│                                   │                                  │
│                            lip_synced_face_region               ③    │
│                                                                      │
│  Three-layer compositing:                                            │
│    Layer 0: clean_bg (from ①)     ──── background                    │
│    Layer 1: original body (no head) ── torso / hands                 │
│    Layer 2: lip_synced_face (from ③) ─ new face + Poisson blend      │
│                                   │                                  │
│              Kalman smoothing → SyncNet QA → FFmpeg mux → output     │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Mode C — End-to-End Diffusion (Upper Body)

Direct audio-to-upper-body generation. No explicit 3DMM intermediate required.
Use for scenarios requiring natural hand gestures or when presenter has heavy head movement.

```text
┌──────────────────────────────────────────────────────────────────────┐
│  Offline (per presenter, one-time)                                   │
│                                                                      │
│  Video corpus ──► select 1–3 high-quality reference frames           │
│  Voice samples ──► CosyVoice voice prompt extraction                 │
└──────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Online (per new video)                                              │
│                                                                      │
│  Input video                                                         │
│    ├─ Audio stream:                                                  │
│    │    FunASR → translate → CosyVoice TTS → new_audio               │
│    │                                                                 │
│    └─ Visual stream:                                                 │
│         InsightFace → ref_frame (anchor appearance frame)            │
│         SAM2 → background_mask                                       │
│         ProPainter → clean_bg                                   ①    │
│                                                                      │
│  ref_frame + new_audio ──► EchoMimic V2  (half-body)                 │
│                         or FantasyTalking (head, Wan2.1 14B)         │
│                         or OmniAvatar    (full-body, Wan2.1 14B)     │
│                                   │                                  │
│                            generated_upper_body RGBA            ②    │
│                                                                      │
│  Two-layer compositing:                                              │
│    Layer 0: clean_bg (from ①)         ──── background                │
│    Layer 1: generated_upper_body (from ②) ─ presenter + Poisson      │
│                                   │                                  │
│              SyncNet QA → FFmpeg mux → output                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```text
video_trans/
│
├── scripts/                         Data processing & utility scripts
│   ├── batch_face_scan.py               Scan videos for face presence ratio
│   ├── prepare_bi_clips.py              Select videos + extract 25fps clips + audio
│   ├── batch_vhap.py                    Batch VHAP FLAME tracking
│   ├── build_bi_lmdb.py                 Build LMDB training dataset
│   ├── extract_audio_features.py        BEATs + WavLM per-frame embeddings
│   ├── render_flame_mesh.py             FLAME mesh visualization (OpenCV)
│   ├── run_inference_standalone.py      3-stage inference without LMDB
│   ├── scan_face_segments.py            Find face-present intervals
│   ├── rename_dataset.py                Rename Chinese dirs to indexed IDs
│   ├── run_bi_data_pipeline.sh          End-to-end data pipeline
│   ├── train_bi.sh                      Training launch script
│   └── setup_envs.sh                    Conda environment setup
│
├── data/                            Local data (git-ignored)
│   ├── raw/bi/                      Raw BI presenter videos (278)
│   ├── raw/ji/                      Raw JI presenter videos (57)
│   ├── presenters/bi/               Renamed video archive (symlinks)
│   ├── presenters/bi_archive/       FLAME shape + params for BI
│   └── bi_training/                 ★ Prepared training data
│       ├── clips/                       53 selected videos (25fps + 16kHz FLAC)
│       ├── flame_params/                VHAP FLAME coefs per video
│       ├── ref_frames/                  ★ NEW: anchor frames for diffusion modes
│       └── lmdb/                        Training LMDB datasets
│           ├── BI-full-intensity/           coef + audio + intensity
│           ├── audio_emb_lmdb/              BEATs + WavLM embeddings
│           ├── text_vad_lmdb/               Whisper + VAD
│           ├── stats_train/                 Normalization statistics
│           ├── train.txt / val.txt          Split files
│           └── keys.txt
│
├── runs/                            Experiment outputs (git-ignored)
│   ├── face_scan_bi/                Face scan results (235 videos)
│   └── demo/                        Demo renders
│
├── core_model/                      Audio→3DMM model (git submodule)
│   ├── main_keyness.py              Keyness predictor training
│   ├── main_stage1.py               Stage 1: head pose diffusion
│   ├── main_stage2.py               Stage 2: expression diffusion
│   ├── models/                      Model definitions
│   ├── data/                        LMDB data loaders
│   ├── options/                     Training configs
│   ├── scripts_for_dataset/         LMDB dataset preparation tools
│   ├── experiments/                 Checkpoints (local only)
│   └── datasets/                    LMDB data (local only)
│
├── module_A_offline/                Offline presenter processing
│   ├── A1_flame_extraction/
│   │   ├── VHAP/                        Video head tracking → FLAME
│   │   └── spectre/                     Speech-aware FLAME (legacy)
│   ├── A2_3dgs_avatar/
│   │   └── GaussianAvatars/             FLAME-rigged 3DGS  [Mode A only]
│   └── A3_voice_clone/
│       └── CosyVoice/                   Zero-shot TTS
│
├── module_B_audio/                  Audio processing
│   ├── B1_asr/                      FunASR transcription
│   ├── B2_translate/                Translation (LLM-based)
│   └── B3_tts/                      CosyVoice synthesis
│
├── module_C_visual/                 Visual preprocessing
│   ├── C1_detect/                   InsightFace face detection
│   ├── C2_segment/                  SAM2 mask generation
│   └── C3_inpaint/                  ProPainter background inpainting
│
├── module_D_diffusion/              ★ NEW: Diffusion rendering backends
│   ├── D1_latentsync/
│   │   └── LatentSync/              [Mode B] Audio-conditioned LDM lip sync
│   ├── D2_echomimic/
│   │   └── echomimic_v2/            [Mode C] Half-body audio-driven animation
│   ├── D3_fantasytalking/
│   │   └── fantasy-talking/         [Mode C alt] Wan2.1-based portrait animation
│   └── D4_omniavatar/
│       └── OmniAvatar/              [Mode C alt] Full-body, Wan2.1 14B
│
├── module_F_blending/               Compositing & enhancement
│   ├── F1_pose_warp.py              FLAME-guided affine head warp  [Mode B]
│   ├── F2_composite.py              Three-layer compositing + Poisson blend
│   ├── F3_kalman.py                 Temporal smoothing (Kalman filter)
│   └── F4_sr.py                     CodeFormer super-resolution  [Mode A/B]
│
└── module_G_postprocess/            QA & encoding
    ├── G1_syncnet_qa.py             SyncNet lip-sync quality gate
    └── G2_mux.py                    FFmpeg audio/video mux
```

---

## Hardware & Environment

> **GPU**: NVIDIA GeForce RTX 5090 (Blackwell, sm_120)  
> Requires **PyTorch >= 2.8.0 + cu128**. All envs use `--index-url https://download.pytorch.org/whl/cu128`.

### Conda Environments

| Env | Module | Used In | Status |
|-----|--------|---------|--------|
| `vhap` | VHAP + data pipeline + core_model training | Offline / Mode A | ✅ Working |
| `gaussian-avatars` | GaussianAvatars 3DGS | Mode A only | ✅ Compiled |
| `cosyvoice` | CosyVoice TTS | All modes | ⚠️ Partial |
| `funasr` | FunASR ASR | All modes | ✅ Working |
| `sam2` | SAM2 segmentation | All modes | ✅ Ready |
| `propainter` | ProPainter inpainting | All modes | ✅ Ready |
| `latentsync` | LatentSync 1.6 lip sync | Mode B | 🆕 To setup |
| `echomimic` | EchoMimic V2 half-body | Mode C | 🆕 To setup |
| `fantasytalking` | FantasyTalking / OmniAvatar | Mode C alt | 🆕 To setup |
| `codeformer` | CodeFormer SR | Mode A/B | ✅ Ready |
| `syncnet` | SyncNet lip-sync QA | All modes | ✅ Ready |

### RTX 5090 Build Notes

pytorch3d, diff-gaussian-rasterization, nvdiffrast all require system nvcc 13.2 + torch CUDA check patch.  
See `scripts/setup_envs.sh` for details.

LatentSync, EchoMimic V2, FantasyTalking were tested on CUDA 12.4 (cu124). Use the cu128 index URL
and verify compatibility with `torch.version.cuda` before running.

---

## Setting Up New Diffusion Environments

### LatentSync (Mode B)

```bash
conda create -n latentsync python=3.10 -y
conda activate latentsync
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r module_D_diffusion/D1_latentsync/LatentSync/requirements.txt

# Download checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 \
    --local-dir module_D_diffusion/D1_latentsync/LatentSync/checkpoints
```

Inference requires ~6.5 GB VRAM. Fine-tuning stage 2 (efficient) requires ~20 GB VRAM.

### EchoMimic V2 (Mode C)

```bash
conda create -n echomimic python=3.10 -y
conda activate echomimic
pip install torch==2.5.1 torchvision torchaudio xformers==0.0.28.post3 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r module_D_diffusion/D2_echomimic/echomimic_v2/requirements.txt
pip install --no-deps facenet_pytorch==2.6.0

# Download checkpoints
huggingface-cli download BadToBest/EchoMimicV2 \
    --local-dir module_D_diffusion/D2_echomimic/echomimic_v2/pretrained_weights
```

### FantasyTalking (Mode C alt)

```bash
conda create -n fantasytalking python=3.10 -y
conda activate fantasytalking
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r module_D_diffusion/D3_fantasytalking/fantasy-talking/requirements.txt

# Requires Wan2.1-I2V-14B-720P (~28 GB download)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
    --local-dir module_D_diffusion/D3_fantasytalking/fantasy-talking/models/Wan2.1-I2V-14B-720P
huggingface-cli download acvlab/FantasyTalking fantasytalking_model.ckpt \
    --local-dir module_D_diffusion/D3_fantasytalking/fantasy-talking/models
```

---

## Data Pipeline (for New Presenters)

This pipeline is shared across all rendering modes. Steps 1–4 are always required.
Step 5 (GaussianAvatars training) is only required for Mode A.

### Step 1: Face Scanning

```bash
conda activate vhap
python scripts/batch_face_scan.py \
    --video_dir data/presenters/bi \
    --output_dir runs/face_scan_bi \
    --sample_fps 0.5
```

Output: `scan_results.json` with per-video face percentage.

### Step 2: Select Videos + Extract Clips

```bash
python scripts/prepare_bi_clips.py \
    --scan_results runs/face_scan_bi/scan_results.json \
    --video_dir data/presenters/bi \
    --output_dir data/bi_training \
    --min_face_pct 30 --target_hours 10
```

Output: 25fps MP4 + 16kHz FLAC per video.

### Step 3: VHAP FLAME Tracking

```bash
python scripts/batch_vhap.py \
    --manifest data/bi_training/clip_manifest.json \
    --vhap_dir module_A_offline/A1_flame_extraction/VHAP \
    --output_dir data/bi_training/flame_params \
    --num_epochs 1
```

Output: per-video `coef.npz` (pose 6D + exp 50D + shape 100D).

### Step 4: Build LMDB

```bash
python scripts/build_bi_lmdb.py \
    --manifest data/bi_training/clip_manifest.json \
    --flame_dir data/bi_training/flame_params \
    --output_dir data/bi_training/lmdb
```

Output: 3 LMDB databases + stats + train/val split.

### Step 4b: Extract Reference Frames ★ NEW (Modes B and C)

```bash
python scripts/extract_ref_frames.py \
    --manifest data/bi_training/clip_manifest.json \
    --output_dir data/bi_training/ref_frames \
    --strategy frontal_max_face          # selects clearest frontal frame per video
```

Output: one high-quality reference PNG per presenter, used as the identity anchor
for LatentSync inpainting and EchoMimic V2 generation.

### Step 5: GaussianAvatars Training (Mode A only)

```bash
conda activate gaussian-avatars
bash module_A_offline/A2_3dgs_avatar/train_avatar.sh \
    --flame_params data/bi_training/flame_params/BI_0112/coef.npz \
    --video data/bi_training/clips/BI_0112.mp4 \
    --output runs/avatars/BI_0112
```

### Full Pipeline (one command)

```bash
bash scripts/run_bi_data_pipeline.sh   # Steps 1–4 + 4b
```

---

## Online Inference

### Mode B — Hybrid (Recommended)

```bash
# Step 1: Audio pipeline
conda activate funasr
python module_B_audio/B1_asr/transcribe.py \
    --video input.mp4 --output runs/asr/

python module_B_audio/B2_translate/translate.py \
    --asr_result runs/asr/result.json --target_lang ja

conda activate cosyvoice
python module_B_audio/B3_tts/synthesize.py \
    --text runs/translate/result.json \
    --voice_prompt data/bi_training/voice_prompts/BI.pt \
    --output runs/tts/new_audio.wav

# Step 2: Visual preprocessing
conda activate sam2
python module_C_visual/preprocess.py \
    --video input.mp4 \
    --output_dir runs/visual/ \
    --mode inpaint_bg                    # runs InsightFace → SAM2 → ProPainter

# Step 3: Head pose generation (Keyness model)
conda activate vhap
python core_model/run_inference_standalone.py \
    --audio runs/tts/new_audio.wav \
    --output_dir runs/3dmm/ \
    --mode pose_only                     # only head_pose output needed for Mode B

# Step 4: FLAME-guided head warp
python module_F_blending/F1_pose_warp.py \
    --video input.mp4 \
    --pose runs/3dmm/head_pose.npy \
    --mask runs/visual/face_mask/ \
    --output runs/posed_frames/

# Step 5: LatentSync lip generation
conda activate latentsync
python module_D_diffusion/D1_latentsync/run_latentsync.py \
    --video runs/posed_frames/ \
    --audio runs/tts/new_audio.wav \
    --output runs/lip_synced/

# Step 6: Compositing
conda activate propainter
python module_F_blending/F2_composite.py \
    --clean_bg runs/visual/clean_bg/ \
    --body_frames runs/visual/body_frames/ \
    --face_frames runs/lip_synced/ \
    --output runs/composite/

# Step 7: Temporal smoothing + QA + mux
python module_F_blending/F3_kalman.py --input runs/composite/ --output runs/smoothed/
conda activate syncnet
python module_G_postprocess/G1_syncnet_qa.py \
    --video runs/smoothed/ --audio runs/tts/new_audio.wav
python module_G_postprocess/G2_mux.py \
    --video runs/smoothed/ --audio runs/tts/new_audio.wav \
    --output output_translated.mp4
```

### Mode C — EchoMimic V2 (Full Upper Body)

```bash
# Steps 1–2 same as Mode B (audio pipeline + ProPainter bg)

# Step 3: EchoMimic V2 upper body generation
conda activate echomimic
python module_D_diffusion/D2_echomimic/run_echomimic.py \
    --ref_frame data/bi_training/ref_frames/BI_ref.png \
    --audio runs/tts/new_audio.wav \
    --output runs/echomimic_out/

# Step 4: Two-layer compositing (no body layer needed)
python module_F_blending/F2_composite.py \
    --clean_bg runs/visual/clean_bg/ \
    --face_frames runs/echomimic_out/ \
    --mode two_layer \
    --output runs/composite/

# Steps 5–6 same as Mode B (QA + mux)
```

### Mode A — 3DMM + GaussianAvatars (Full Control)

```bash
# Steps 1–2 same as Mode B (audio pipeline + ProPainter bg)

# Step 3: Full 3DMM inference (exp + pose + jaw)
conda activate vhap
python core_model/run_inference_standalone.py \
    --audio runs/tts/new_audio.wav \
    --output_dir runs/3dmm/ \
    --mode full

# Step 4: GaussianAvatars rendering
conda activate gaussian-avatars
python module_A_offline/A2_3dgs_avatar/render.py \
    --avatar runs/avatars/BI_0112/ \
    --params runs/3dmm/ \
    --output runs/rendered_face/

# Step 5: Three-layer compositing
python module_F_blending/F2_composite.py \
    --clean_bg runs/visual/clean_bg/ \
    --body_frames runs/visual/body_frames/ \
    --face_frames runs/rendered_face/ \
    --output runs/composite/

# Step 6: CodeFormer SR + Kalman + SyncNet QA + mux
conda activate codeformer
python module_F_blending/F4_sr.py --input runs/composite/ --output runs/sr/
python module_F_blending/F3_kalman.py --input runs/sr/ --output runs/smoothed/
conda activate syncnet
python module_G_postprocess/G1_syncnet_qa.py \
    --video runs/smoothed/ --audio runs/tts/new_audio.wav
python module_G_postprocess/G2_mux.py \
    --video runs/smoothed/ --audio runs/tts/new_audio.wav \
    --output output_translated.mp4
```

---

## Compositing Order (Critical)

The correct layering order is non-negotiable. Incorrect ordering (e.g., inpainting after
face compositing) causes artifacts at the face boundary.

```
① ProPainter background inpainting
     Input:  original frames + face masks
     Output: clean_bg (head region filled with background texture)
     ↓
② Face rendering (Mode A: GaussianAvatars / Mode B: LatentSync / Mode C: EchoMimic)
     Input:  new_audio (+ posed_frames for Mode B)
     Output: face_rgba with alpha channel
     ↓
③ Three-layer alpha compositing
     Layer 0 (bottom): clean_bg
     Layer 1 (mid):    original body frames, face region zeroed out by mask
     Layer 2 (top):    face_rgba, alpha-blended onto Layer 1
     ↓
④ Poisson blending at face boundary seam
     ↓
⑤ Kalman filter temporal smoothing (landmark-based, suppresses jitter)
     ↓
⑥ CodeFormer SR (Mode A/B only; Mode C output is already high-res)
     ↓
⑦ SyncNet QA gate (reject frames with sync score < threshold)
```

---

## LMDB Dataset Format

Three LMDB databases per dataset, matching HDTF_TFHP format.

### 1. Full-Intensity LMDB

```python
key/metadata: {"n_frames": int, "intensity_mean": float}
key/000: {
    "audio": bytes,          # FLAC 16kHz mono
    "coef": {
        "pose":  (T, 6),     # [head_rot(3), jaw(3)] axis-angle
        "exp":   (T, 50),    # FLAME expression coefficients
        "shape": (T, 100),   # FLAME shape (identity, near-constant)
    },
    "intensity": (T,),       # Keyness motion amplitude [0, 1]
}
```

### 2. Audio Embedding LMDB

```python
key/000: {
    "emb": {
        "beats": (T, 768),   # UniSpeech-SAT / BEATs embeddings
        "wavlm": (T, 768),   # WavLM-Large embeddings
    },
    "target_len": int,
}
```

### 3. Text/VAD LMDB

```python
key/000: {
    "texts":         list[str],           # Whisper transcripts per window
    "vad":           tensor(N, 3),        # VAD logits (speech / silence / noise)
    "window_ranges": list[tuple],         # (start_frame, end_frame)
    "window_frames": 100,
    "n_frames":      int,
}
```

### Normalization Stats

```
stats_train/
├── pose_mean.npy    (6,)
├── pose_std.npy     (6,)
├── exp_mean.npy     (50,)
├── exp_std.npy      (50,)
├── shape_mean.npy   (100,)
└── shape_std.npy    (100,)
```

---

## BI Presenter Data Status

### Face Scan Results (235 videos scanned)

- Total video: 57h, estimated face content: 13.8h
- Average face rate: 23% (product review format, mostly screen recording)
- Videos with >30% face: **53** | Videos with >50% face: **14**

### Selected for Training: 53 videos

- 25fps clips + 16kHz FLAC extracted to `data/bi_training/clips/`
- Total duration: ~13.4h

### VHAP Tracking Completed: 2 videos

| Video | Duration | Epochs | Quality |
|-------|----------|--------|---------|
| BI_0112 | 60s | 30 | High |
| BI_0190 | 3min | 1 | Baseline |

### Known Issue: FLAME Parameter Space Mismatch

HDTF training data uses SPECTRE-extracted FLAME params; our data uses VHAP-extracted FLAME params.
Value ranges differ by ~10–20× in pose. Consequences:

- Pretrained model cannot be directly denormalized with VHAP stats
- Fine-tuning on 2 videos causes high-frequency jitter in pose output
- **Resolution path**: collect 10+ videos of VHAP tracking, retrain normalization stats, then continue fine-tuning

This mismatch is a **Mode A/B concern only** — Mode C (EchoMimic/FantasyTalking) bypasses 3DMM
entirely and is not affected.

---

## Audio→3DMM Model Checkpoints

### Pretrained on HDTF

| Component | Path | Iterations |
|-----------|------|------------|
| Keyness predictor | `experiments/keyness/v1/.../iter_0200000.pt` | 200K |
| Stage 1 (head pose) | `experiments/stage1/stage1end/iter_0131500.pt` | 131.5K |
| Stage 2 (expression) | `stage2/experiments/DPT/dpt_headproxy-260123/iter_0050000.pt` | 50K |

### Fine-tuned on BI (2 videos, experimental)

| Component | Path | Delta |
|-----------|------|-------|
| Keyness predictor | `experiments/keyness/v1/.../iter_0205000.pt` | +5K |
| Stage 1 (head pose) | `experiments/stage1/stage1end/iter_0136000.pt` | +4.5K |
| Stage 2 (expression) | `experiments/stage2/stage2_bi-260401_204133/iter_0055000.pt` | +5K |

### Key Experimental Results (HDTF_TFHP, 20 videos, 4855 frames)

| Metric | Improvement | Notes |
|--------|-------------|-------|
| Head–ExpVel Correlation | **+13.2%** | Primary Keyness validation |
| GT-Keyness Ablation | **+37.7%** (0.175 → 0.241) | Strongest result; centerpiece claim |
| Head L1 | **−8.3%** | Significant |
| Exp FGD | −1.7% | Marginal; do not overstate |

---

## Rendering Backend Comparison

| Criterion | Mode A (3DGS) | Mode B (LatentSync) | Mode C (EchoMimic V2) |
|-----------|--------------|---------------------|----------------------|
| Identity fidelity | ★★★★★ Personalized | ★★★★ Preserves original | ★★★ One-shot ref |
| Lip sync quality | ★★★★ FLAME jaw | ★★★★★ LDM + SyncNet | ★★★★ Audio-driven |
| Head motion | ★★★★★ Full 3DMM control | ★★★★ Pose warp | ★★★ Audio implicit |
| Upper body / gestures | ✗ | ✗ | ✅ Half-body |
| Temporal consistency | ★★★★ (Kalman helps) | ★★★★ (TREPA layer) | ★★★ (can flicker) |
| Offline setup cost | ★ Heavy (3DGS train) | ★★★★★ None | ★★★★ Ref frame only |
| Inference speed | ★★★ | ★★★★ | ★★ (14B option slow) |
| FLAME param dependency | Required | Pose only | None |
| Recommended for | Research / paper | **Production** | Long takes / gestures |

---

## External Dependencies

### Core (all modes)
- **[FunASR](https://github.com/modelscope/FunASR)** — Chinese ASR
- **[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)** — Zero-shot TTS / voice cloning
- **[SAM2](https://github.com/facebookresearch/sam2)** — Video object segmentation
- **[ProPainter](https://github.com/sczhou/ProPainter)** — Video inpainting
- **[SyncNet](https://github.com/joonson/syncnet_python)** — Lip-sync QA

### Mode A
- **[VHAP](https://github.com/ShenhanQian/VHAP)** — FLAME tracking
- **[GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)** — FLAME-rigged 3DGS
- **[CodeFormer](https://github.com/sczhou/CodeFormer)** — Face super-resolution
- **FLAME 2020** — `core_model/models/data/FLAME2020/`

### Mode B (adds)
- **[LatentSync](https://github.com/bytedance/LatentSync)** (ByteDance, CVPR 2025 submitted) — Audio-conditioned LDM lip sync, no intermediate motion representation, TREPA temporal consistency

### Mode C (replaces A+B rendering)
- **[EchoMimic V2](https://github.com/antgroup/echomimic_v2)** (Ant Group, CVPR 2025) — Half-body audio-driven animation, Audio-Pose Dynamic Harmonization
- **[FantasyTalking](https://github.com/Fantasy-AMAP/fantasy-talking)** (Alibaba, ACM MM 2025) — Wan2.1 I2V 14B-based portrait animation
- **[OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar)** (ZJU + Alibaba, 2025) — Full-body, Wan2.1 T2V 14B backbone

---

## References

```bibtex
@inproceedings{zeng2026keyness,
  title={Keyness-Aware Multi-Stage Diffusion for Expressive Audio-Driven Facial Animation},
  author={Zeng, Ziyue and Xie, Weijing and Watanabe, Hiroshi},
  booktitle={Proc. Interspeech},
  year={2026}
}

@article{meng2024echomimicv2,
  title={EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation},
  author={Meng, Rang and Zhang, Xingyu and Li, Yuming and Ma, Chenguang},
  journal={arXiv preprint arXiv:2411.10061},
  year={2024}
}

@article{li2024latentsync,
  title={LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision},
  author={Li, Chunyu and Zhang, Chao and Xu, Weikai and others},
  journal={arXiv preprint arXiv:2412.09262},
  year={2024}
}

@inproceedings{cui2024hallo3,
  title={Hallo3: Highly Dynamic and Realistic Portrait Image Animation with Video Diffusion Transformer},
  author={Cui, Jiahao and Li, Hui and Zhan, Yun and others},
  journal={arXiv preprint arXiv:2412.00733},
  year={2024}
}

@inproceedings{xu2025hunyuanportrait,
  title={HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation},
  author={Xu, Zunnan and Yu, Zhentao and others},
  booktitle={CVPR},
  year={2025}
}

@article{wang2025fantasytalking,
  title={FantasyTalking: Realistic Talking Portrait Generation via Coherent Motion Synthesis},
  author={Wang, Mengchao and Wang, Qiang and others},
  journal={arXiv preprint arXiv:2504.04842},
  year={2025}
}

@misc{gan2025omniavatar,
  title={OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation},
  author={Gan, Qijun and Yang, Ruizi and Zhu, Jianke and others},
  eprint={2506.18866},
  year={2025}
}
```
