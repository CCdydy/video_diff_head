# Video Translation Pipeline

> 给固定主播的评测视频换语种（中→日等）。
> 新 TTS 音频驱动上半身重生成，原始视频背景/身体区域最大复用。

---

## 一句话架构

```
原始视频 + 新音频
    │
    ├─ 背景 / 身体区域  →  直接复用（0 计算）
    └─ 上半身 / 人脸区域 →  VACE masked V2V + 音频 cross-attention 重生成
                               │
                         ProPainter 边界修复 → 合成 → 输出
```

核心思路：**原始视频是先验，新音频只驱动差异**。不从零生成，只编辑需要变化的区域。

---

## 技术路线选择

### 为什么是 VACE + FantasyTalking 音频适配器

| 方案 | 能否 V2V | 上半身 | 音频条件 | 身份保真 |
|------|---------|--------|---------|---------|
| LatentSync | ✅（仅唇部） | ❌ | ✅ | ✅ |
| EchoMimic V2 | ❌（I2V生成） | ✅ | ✅ | ⚡ one-shot |
| FantasyTalking | ❌（I2V生成） | ✅ | ✅ | ✅ CLIP |
| OmniAvatar | ❌（T2V生成） | ✅ | ✅ | ✅ VAE concat |
| **VACE + 音频** | **✅** | **✅** | **✅（需接入）** | **✅ 原始帧** |

VACE（`ali-vilab/VACE-Wan2.1-14B`）是 Wan2.1 官方的 masked V2V 编辑工具，原生支持
`src_video + src_mask + src_ref_images` 输入。FantasyTalking 证明了在 Wan2.1 DiT 的
40 个 block 里插入 audio cross-attention 是可行且效果好的。两者结合 = 有音频条件的
masked 视频编辑，正是这个任务所需要的。

### 两种运行模式

```
Mode 1 (快速验证):  FantasyTalking 直接推理
    ref_frame + new_audio → 生成新上半身视频
    → 用 SAM2 mask 合成回原始背景
    缺点: 上半身姿态与原始视频不同步，手/肩位置可能对不上

Mode 2 (目标方案):  VACE masked V2V + 音频条件
    原始视频 + face_mask + new_audio → 只编辑脸部区域
    → 背景/身体完全保留原始帧
    优点: 除人脸外的所有内容 pixel-perfect 保留
```

**Mode 1 现在就能跑，Mode 2 需要把音频适配器接入 VACE。**

---

## 完整流程

```
┌─────────────────────────────────────────────────────────────────────┐
│  预处理（每个视频只跑一次）                                           │
│                                                                     │
│  原始视频 (25fps)                                                    │
│    ├─ InsightFace  →  face_bbox per frame                           │
│    ├─ SAM2         →  upper_body_mask (T, H, W)  ← 主要用这个        │
│    │                   face_mask (T, H, W)        ← 精细编辑用       │
│    └─ ProPainter   →  clean_bg (T, H, W, 3)                        │
│                        input:  frames + upper_body_mask              │
│                        output: 上半身区域填充背景纹理                  │
│                                                                     │
│  新 TTS 音频 (16kHz mono WAV)                                        │
│    └─ wav2vec2-base-960h  →  audio_feat (T_a, 768)                  │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Mode 1: FantasyTalking 推理（快速验证路线）                          │
│                                                                     │
│  输入:                                                               │
│    ref_frame    = 原始视频第 0 帧（或人工选最清晰帧）                   │
│    audio_path   = new_audio.wav                                     │
│    prompt       = "a person talking, upper body, natural motion"    │
│                                                                     │
│  调用:                                                               │
│    python infer.py \                                                │
│        --image_path ref_frame.png \                                 │
│        --audio_path new_audio.wav \                                 │
│        --prompt "..." \                                             │
│        --num_inference_steps 30 \                                   │
│        --audio_cfg 4.0 \                                            │
│        --output_path gen_upperbody/                                 │
│                                                                     │
│  输出: 81帧 @ 512×512, ~23fps                                        │
│  chunk 策略: 每次 81 帧，最后 13 帧作为下一 chunk 的 prefix            │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼  (或走 Mode 2)
┌─────────────────────────────────────────────────────────────────────┐
│  Mode 2: VACE + 音频条件 masked V2V（目标路线）                       │
│                                                                     │
│  ① 对原始帧做区域差异化加噪（Differential Diffusion 思路）            │
│                                                                     │
│     strength_map per pixel:                                         │
│       lips region     →  0.50  (最大编辑，换唇型)                    │
│       rest of face    →  0.35  (表情/头动)                          │
│       neck/shoulder   →  0.15  (轻微跟随)                           │
│       body/background →  0.00  (完全保留)                           │
│                                                                     │
│     z_noisy[t] = sqrt(a_bar(t_start)) * z_orig[t] + sqrt(1-a_bar)*e│
│              where t_start = strength_map * T_MAX                   │
│     face region only: 只对 face_mask 内的 latent patch 加噪          │
│                                                                     │
│  ② VACE context adapter 接收原始帧信息                               │
│                                                                     │
│     src_video   = z_orig         原始 VAE 编码                      │
│     src_mask    = upper_body_mask (binary, H/8 x W/8)              │
│     src_ref     = z_orig[0]      第0帧作为身份锚定                   │
│                                                                     │
│     VACE 的 vace_blocks 把 src_video 的未遮罩区域信息                 │
│     注入到每个 DiT block 的 context，保证背景/身体 pixel-perfect      │
│                                                                     │
│  ③ 音频 cross-attention（移植自 FantasyTalking）                     │
│                                                                     │
│     audio_feat  →  AudioProjModel: Linear(768→2048) + LayerNorm     │
│                →  split_audio_sequence(): 按 VAE 压缩比切窗           │
│                    21 latent frames → 21 audio windows               │
│                →  WanCrossAttentionProcessor: 插入全部 40 个 block   │
│                    per-frame attention: 每帧只关注对应时间窗口音频     │
│                    zero-init k/v proj → 训练稳定，初期不破坏原模型    │
│                                                                     │
│  输出: 只有 upper_body_mask 内区域被重新生成，其余与原始帧完全一致     │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  合成（Mode 1 / Mode 2 共用）                                        │
│                                                                     │
│  ① 区域混合                                                          │
│     soft_mask = GaussianBlur(upper_body_mask, radius=16px)          │
│     result    = orig_frame * (1 - soft_mask)                        │
│               + gen_frame  * soft_mask                              │
│     [Mode 2 跳过：mask 外区域已 pixel-perfect]                        │
│                                                                     │
│  ② Poisson 边界融合（仅 Mode 1）                                     │
│     cv2.seamlessClone(gen_frame, orig_frame, mask, center)          │
│                                                                     │
│  ③ ProPainter 边界修复                                               │
│     boundary_seam_mask = dilate(mask) - erode(mask)  宽度~20px      │
│     只修复边缘接缝，不动内部                                           │
│                                                                     │
│  ④ Kalman landmark 平滑（仅 Mode 1）                                 │
│     sigma_process=0.01, sigma_measurement=0.1                       │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  后处理                                                              │
│  SyncNet QA:  sync_conf > 3.0 pass / < 3.0 flag                    │
│  FFmpeg mux:  -c:v libx264 -crf 18 -c:a aac                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 仓库结构

```
video_trans/
│
├── scripts/
│   ├── batch_face_scan.py
│   ├── prepare_bi_clips.py
│   ├── extract_ref_frames.py        NEW: 给每个主播选最佳参考帧
│   ├── extract_audio_features.py
│   ├── run_translate_video.py       NEW: 一键翻译入口
│   ├── run_bi_data_pipeline.sh
│   └── setup_envs.sh
│
├── data/
│   ├── raw/bi/                      原始 BI 视频 (278 个)
│   ├── raw/ji/                      原始 JI 视频 (57 个)
│   ├── presenters/
│   │   ├── bi/
│   │   └── bi_ref_frames/           NEW: 每主播精选参考帧 (PNG)
│   └── bi_training/
│       ├── clips/                   53 个视频 (25fps + 16kHz FLAC)
│       ├── flame_params/            VHAP 参数（旧架构遗留）
│       └── lmdb/                    旧训练数据（保留备用）
│
├── module_A_offline/
│   ├── A1_flame_extraction/VHAP/   仅旧架构
│   ├── A2_3dgs_avatar/              仅旧架构，新流程不用
│   └── A3_voice_clone/CosyVoice/   TTS（所有模式共用）
│
├── module_B_audio/
│   ├── B1_asr/                      FunASR
│   ├── B2_translate/                翻译（LLM）
│   └── B3_tts/                      CosyVoice 合成
│
├── module_C_visual/
│   ├── C1_detect/                   InsightFace
│   ├── C2_segment/
│   │   ├── sam2_tracker.py          SAM2 mask 传播（全视频一次）
│   │   └── mask_utils.py            dilate / soft_mask / boundary_mask
│   └── C3_inpaint/
│       └── propainter_wrapper.py    背景修复 + 边界接缝修复
│
├── module_D_diffusion/
│   ├── D1_fantasytalking/
│   │   └── fantasy-talking/         Mode 1 推理（直接用原仓库）
│   │       ├── infer.py
│   │       ├── models/
│   │       │   ├── wan_audio.py     WanCrossAttentionProcessor
│   │       │   │                    AudioProjModel: Linear(768→2048)+LN
│   │       │   └── audio_utils.py   split_audio_sequence()
│   │       └── pretrained_weights/
│   │           ├── Wan2.1-I2V-14B-720P/
│   │           ├── wav2vec2-base-960h/
│   │           └── fantasytalking_model.ckpt
│   │
│   ├── D2_vace/
│   │   ├── VACE-Wan2.1-14B/         Mode 2 基础
│   │   ├── vace_audio_pipeline.py   NEW: VACE + 音频适配器集成
│   │   └── diff_strength_map.py     NEW: 区域差异化加噪
│   │
│   └── D3_omniavatar/               对比基线
│       └── OmniAvatar/
│
├── module_F_blending/
│   ├── F1_composite.py              合成 + Gaussian soft mask
│   ├── F2_poisson.py                seamlessClone 边界融合
│   ├── F3_kalman.py                 landmark 时序平滑（Mode 1）
│   └── F4_color_harmonize.py        直方图匹配
│
└── module_G_postprocess/
    ├── G1_syncnet_qa.py
    └── G2_mux.py
```

---

## 环境配置

> GPU: RTX 5090 (Blackwell, sm_120, 32GB VRAM) — PyTorch >= 2.8.0 + cu128

```bash
# FantasyTalking / VACE 共用环境
conda create -n wan_audio python=3.10 -y && conda activate wan_audio

pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
pip install xformers==0.0.28.post3 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r module_D_diffusion/D1_fantasytalking/fantasy-talking/requirements.txt
pip install diffusers transformers accelerate

# 模型权重（Mode 1: FantasyTalking）
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
    --local-dir module_D_diffusion/D1_fantasytalking/fantasy-talking/models/Wan2.1-I2V-14B-720P
huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir module_D_diffusion/D1_fantasytalking/fantasy-talking/models/wav2vec2-base-960h
huggingface-cli download acvlab/FantasyTalking fantasytalking_model.ckpt \
    --local-dir module_D_diffusion/D1_fantasytalking/fantasy-talking/models

# 模型权重（Mode 2: VACE）
huggingface-cli download ali-vilab/VACE-Wan2.1-14B \
    --local-dir module_D_diffusion/D2_vace/VACE-Wan2.1-14B
```

### VRAM 参考（RTX 5090, 32GB）

| 配置 | 峰值 VRAM | 速度（81帧） |
|------|----------|------------|
| FantasyTalking FP16 全量 | ~28GB | ~45s |
| FantasyTalking FP8 + `--num_persistent 7B` | ~18GB | ~70s |
| VACE FP16 + `--offload_model` | ~24GB | ~60s |
| VACE FP8 + TeaCache | ~16GB | ~40s |
| SAM2 hiera_large | ~4GB | 44fps |
| ProPainter FP16 (80帧 720p) | ~8GB | ~3s/帧 |

---

## 一键运行

```bash
conda activate wan_audio

python scripts/run_translate_video.py \
    --input       data/raw/bi/BI_0112.mp4 \
    --presenter   bi \
    --target_lang ja \
    --mode        1 \
    --output      output/BI_0112_ja.mp4
```

| 参数 | 说明 |
|------|------|
| `--mode 1/2` | 1=FantasyTalking（快），2=VACE+音频（精确） |
| `--presenter bi` | 加载对应参考帧和声音 prompt |
| `--target_lang ja` | 目标语种，传给 CosyVoice |
| `--chunk_size 81` | 每次处理帧数（5s @ 16fps） |
| `--overlap 13` | chunk 间重叠帧 |
| `--audio_cfg 4.0` | 音频引导强度（3–7） |
| `--strength_face 0.35` | Mode 2 人脸加噪强度 |
| `--strength_lips 0.50` | Mode 2 唇部加噪强度 |

---

## 分步运行

```bash
# 1. 音频管线
conda activate funasr
python module_B_audio/B1_asr/transcribe.py --video input.mp4 --output runs/asr/
python module_B_audio/B2_translate/translate.py \
    --input runs/asr/result.json --target ja --output runs/translate/
conda activate cosyvoice
python module_B_audio/B3_tts/synthesize.py \
    --text runs/translate/result.json \
    --voice_prompt data/presenters/bi/voice_prompt.pt \
    --output runs/tts/new_audio.wav

# 2. 视觉预处理（只跑一次）
conda activate sam2
python module_C_visual/C2_segment/sam2_tracker.py \
    --video input.mp4 --output_dir runs/masks/ --prompt_type auto_face

conda activate propainter
python module_C_visual/C3_inpaint/propainter_wrapper.py \
    --video input.mp4 --mask runs/masks/upper_body_mask/ \
    --output runs/clean_bg/ --subvideo_length 80

# 3a. 上半身重生成（Mode 1）
conda activate wan_audio
python module_D_diffusion/D1_fantasytalking/fantasy-talking/infer.py \
    --image_path data/presenters/bi_ref_frames/BI_ref.png \
    --audio_path runs/tts/new_audio.wav \
    --prompt "a presenter talking, upper body shot, natural gestures" \
    --num_inference_steps 30 --audio_cfg 4.0 \
    --num_persistent_param_in_dit 7000000000 \
    --output_path runs/gen_upperbody/

# 3b. 上半身重生成（Mode 2）
python module_D_diffusion/D2_vace/vace_audio_pipeline.py \
    --src_video input.mp4 \
    --src_mask  runs/masks/upper_body_mask/ \
    --audio_path runs/tts/new_audio.wav \
    --ref_frame  data/presenters/bi_ref_frames/BI_ref.png \
    --strength_map lips:0.5,face:0.35,shoulder:0.15 \
    --output_path runs/vace_out/

# 4. 合成 + QA + 输出
python module_F_blending/F1_composite.py \
    --clean_bg runs/clean_bg/ --gen_frames runs/gen_upperbody/ \
    --orig_frames input_frames/ --mask runs/masks/upper_body_mask/ \
    --mode 1 --output runs/composite/

conda activate syncnet
python module_G_postprocess/G1_syncnet_qa.py \
    --video runs/composite/ --audio runs/tts/new_audio.wav --threshold 3.0
python module_G_postprocess/G2_mux.py \
    --video runs/composite/ --audio runs/tts/new_audio.wav \
    --output output/BI_0112_ja.mp4
```

---

## Mode 2 开发计划

改动约 300 行，分 4 个文件。

### 需要修改的文件

| 文件 | 来源 | 改动 |
|------|------|------|
| `vace_audio_pipeline.py` | 新写 | 加载 VACE 后调用 `set_audio_processor()` |
| `fantasy-talking/models/wan_audio.py` | FantasyTalking | 直接复用，不改 |
| `diff_strength_map.py` | 新写 | 区域差异化加噪 → `z_noisy + t_starts` |
| `VACE/pipeline_vace.py` | VACE 官方 | 加 `audio_feat` 参数透传 |

### 核心代码

```python
# vace_audio_pipeline.py
from fantasy_talking.models.wan_audio import (
    AudioProjModel,               # Linear(768→2048) + LayerNorm
    WanCrossAttentionProcessor,   # per-frame audio cross-attention
    split_audio_sequence,         # 切成 per-latent-frame 窗口
)

def build_vace_audio_model(vace_ckpt, ft_ckpt, device):
    pipe = VACEPipeline.from_pretrained(vace_ckpt, torch_dtype=torch.bfloat16)
    audio_proj = AudioProjModel()
    state = torch.load(ft_ckpt)
    audio_proj.load_state_dict(
        {k: v for k, v in state.items() if k.startswith('audio_proj')}
    )
    # 插入音频 cross-attention 到所有 40 个 DiT block
    for block in pipe.transformer.blocks:
        block.attn2.set_processor(WanCrossAttentionProcessor(audio_scale=1.0))
    return pipe, audio_proj

def run_vace_audio(pipe, audio_proj, src_video, src_mask,
                   audio_wav, ref_frame, strength_map):
    audio_feat    = wav2vec2_extract(audio_wav)              # (T_a, 768)
    audio_cond    = audio_proj(audio_feat)                   # (T_a, 2048)
    audio_windows = split_audio_sequence(audio_cond, n_latent_frames=21)
    z_orig        = pipe.vae.encode(src_video)
    z_noisy, t_starts = apply_diff_strength(z_orig, src_mask, strength_map)
    return pipe(
        src_video=z_noisy, src_mask=src_mask, src_ref=ref_frame,
        audio_feat=audio_windows, t_starts=t_starts, num_steps=25,
    )
```

### Fine-tune（零样本效果不好时）

只训练 `AudioProjModel`（1.6M 参数），冻结其他所有权重：

```bash
python train_audio_proj.py \
    --data_dir  data/bi_training/clips/ \
    --vace_ckpt module_D_diffusion/D2_vace/VACE-Wan2.1-14B/ \
    --ft_ckpt   module_D_diffusion/D1_fantasytalking/.../fantasytalking_model.ckpt \
    --lr 1e-4 --steps 5000 --batch_size 1 --grad_accum 8
# 53 个 BI 视频自监督训练，约 2-3 小时
```

---

## 数据现状（BI 主播）

| 项目 | 状态 |
|------|------|
| 原始视频 | 278 个 |
| 含脸 >30% 视频 | 53 个（~13.4h） |
| 参考帧提取 | 待执行 |
| SAM2 mask 生成 | 待执行 |
| FantasyTalking Mode 1 测试 | 待执行 |
| VHAP 完成 | 2 个（旧架构遗留） |

---

## 外部依赖

| 项目 | 用途 | 链接 |
|------|------|------|
| FantasyTalking | Mode 1 + 音频适配器来源 | `Fantasy-AMAP/fantasy-talking` |
| VACE | Mode 2 masked V2V 基础 | `ali-vilab/VACE` |
| OmniAvatar | 对比基线 | `Omni-Avatar/OmniAvatar` |
| LatentSync | 唇同步后处理（可选） | `bytedance/LatentSync` |
| SAM2 | 上半身 mask | `facebookresearch/sam2` |
| ProPainter | 背景修复 + 边界修复 | `sczhou/ProPainter` |
| FunASR | 中文 ASR | `modelscope/FunASR` |
| CosyVoice | TTS / 声音克隆 | `FunAudioLLM/CosyVoice` |
| SyncNet | 唇同步 QA | `joonson/syncnet_python` |
| Wan2.1 | DiT 基础模型 | `Wan-Video/Wan2.1` |

---

## 引用

```bibtex
@article{wang2025fantasytalking,
  title={FantasyTalking: Realistic Talking Portrait Generation via Coherent Motion Synthesis},
  author={Wang, Mengchao and Wang, Qiang and others},
  journal={arXiv:2504.04842}, year={2025}
}
@article{jiang2025vace,
  title={VACE: All-in-One Video Creation and Editing},
  author={Jiang, Zeyinzi and others},
  journal={arXiv:2503.07598}, year={2025}
}
@article{gan2025omniavatar,
  title={OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation},
  author={Gan, Qijun and Yang, Ruizi and Zhu, Jianke and others},
  journal={arXiv:2506.18866}, year={2025}
}
```
