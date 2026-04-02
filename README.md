# Video Translation Pipeline

> 主播视频多语种翻译：替换语音 + 音频驱动上半身重建。
> 核心路线：**Wan2.1-VACE-14B masked V2V + FantasyTalking 音频 adapter**。

---

## 一句话描述

给定一段中文主播视频，输出日语（或其他语种）翻译版本：
- 语音由 CosyVoice 克隆声音重新合成
- 上半身（面部 + 唇形 + 头部动作）由新音频驱动 VACE 扩散重建
- 背景、身体、手臂直接复用原始视频，零计算量

---

## 架构总览

```
原始视频 (.mp4)
│
├─ [离线，每位主播做一次] ──────────────────────────────────────────
│   SAM2 → face_mask + upper_body_mask (T, H, W)
│   ProPainter → clean_bg (T, H, W, 3)
│   InsightFace → ref_frame.png（最优正面帧）
│   CosyVoice → voice_prompt.pt
│
└─ [在线，每个新视频] ──────────────────────────────────────────────
    │
    ├─ 音频管线
    │   FunASR → 转写 → LLM 翻译 → CosyVoice TTS → new_audio.wav
    │
    ├─ 视觉管线
    │   Wav2Vec2(new_audio) → audio_feat (T, 768)
    │   split_audio_sequence() → audio_windows (21, L, 768)
    │   AudioProjModel → audio_cond (21, L, 2048)
    │
    │   build_strength_map():
    │     唇部 0.55 │ 面部 0.35 │ 上半身 0.15 │ 背景 0.00
    │
    │   VACE-14B masked V2V（25步 DDIM）
    │     + WanAudioCrossAttentionProcessor（40 blocks × audio_attn）
    │   → edited_frames (T, H, W, 3)
    │
    └─ 合成管线
        Layer 0: clean_bg（ProPainter 离线输出）
        Layer 1: orig_body（mask 外原始帧）
        Layer 2: edited_face（VACE 输出）
        → Gaussian soft blend → Poisson seamlessClone
        → Kalman landmark 平滑 → LAB 色调归一化
        → SyncNet QA → FFmpeg mux → output.mp4
```

---

## 目录结构

```
video_trans/
│
├── module_B_audio/                  音频管线
│   ├── B1_asr/                      FunASR 转写（stub）
│   ├── B2_translate/                LLM 翻译（stub）
│   └── B3_tts/                      CosyVoice TTS（stub）
│
├── module_C_visual/                 视觉预处理
│   ├── sam2_tracker.py              ★ SAM2 face + upper_body 双 mask 追踪
│   ├── mask_utils.py                ★ dilate / soft_mask / strength_map
│   └── propainter_wrapper.py        ★ ProPainter 背景修复封装
│
├── module_D_diffusion/              核心扩散模块
│   ├── diff_strength_map.py         ★ Differential Diffusion 区域加噪
│   └── vace_audio_pipeline.py       ★ VACE + FantasyTalking adapter 集成
│       ├── AudioProjModel           Linear(768→2048) + LayerNorm
│       ├── WanAudioCrossAttentionProcessor  零初始化 cross-attn × 40 blocks
│       ├── install_audio_adapter()  注入 + 加载 FantasyTalking 权重
│       ├── VACEAudioPipeline.run_chunk()    单 81 帧推理
│       └── VACEAudioPipeline.run_long_video()  Hamming 窗口长视频分块
│
├── module_F_blending/               合成与后处理
│   ├── F1_composite.py              ★ 三层合成（Mode 1 / Mode 2）
│   ├── F2_poisson.py                ★ seamlessClone 边界融合
│   ├── F3_kalman.py                 ★ MediaPipe landmark + Kalman 平滑
│   └── F4_color_harmonize.py        ★ LAB 直方图匹配色调归一化
│
├── module_G_postprocess/            QA 与输出
│   ├── G1_syncnet_qa.py             ★ SyncNet 唇同步打分
│   └── G2_mux.py                    ★ FFmpeg audio/video mux
│
├── scripts/
│   ├── run_translate_video.py       ★ 一键入口（Mode 1 / Mode 2）
│   ├── preprocess_presenter.py      离线预处理入口
│   └── setup_envs.sh                conda 环境安装
│
└── data/                            本地数据（git ignore）
    ├── models/
    │   ├── Wan2.1-VACE-14B/
    │   ├── wav2vec2-base-960h/
    │   ├── fantasytalking_audio_adapter.ckpt
    │   └── sam2_hiera_large.pt
    └── presenters/
        └── bi/
            ├── raw/                 原始视频
            ├── masks/               SAM2 输出（npz）
            ├── clean_bg/            ProPainter 背景帧
            ├── ref_frame.png        最优参考帧
            └── voice_prompt.pt      CosyVoice 声音 prompt
```

---

## 环境

| 环境名 | 用途 | 状态 |
|--------|------|------|
| `wan_audio` | VACE + SAM2 + 音频管线 + 合成 | ✅ 就绪 |
| `propainter` | 离线背景修复（依赖版本老）| ✅ 就绪 |
| `syncnet` | SyncNet QA | ✅ 就绪 |
| `cosyvoice` | TTS | ⚠️ 部分 |
| `funasr` | ASR | ✅ 就绪 |

```
wan_audio: PyTorch 2.11.0 + cu128, diffusers, transformers
GPU: RTX 5090 (32GB, sm_120, CUDA 12.8)
```

### 关键依赖版本

```
torch==2.11.0+cu128
diffusers>=0.31.0         # Wan2.1 VACE 支持
transformers>=4.45.0      # Wav2Vec2
sam2                      # facebook/sam2
mediapipe>=0.10.0         # Kalman landmark
opencv-python
scipy
insightface
```

---

## 模型下载

```bash
conda activate wan_audio
pip install "huggingface_hub[cli]"

# Wan2.1-VACE-14B (~28GB)
huggingface-cli download Wan-AI/Wan2.1-VACE-14B \
    --local-dir data/models/Wan2.1-VACE-14B

# Wav2Vec2
huggingface-cli download facebook/wav2vec2-base-960h \
    --local-dir data/models/wav2vec2-base-960h

# SAM2
huggingface-cli download facebook/sam2-hiera-large \
    --local-dir data/models/

# FantasyTalking audio adapter（作为 adapter 初始化权重）
huggingface-cli download acvlab/FantasyTalking fantasytalking_model.ckpt \
    --local-dir data/models/
```

---

## 快速开始

### Step 0：离线预处理（每位主播做一次）

```bash
conda activate wan_audio
python scripts/preprocess_presenter.py \
    --presenter   bi \
    --video_dir   data/presenters/bi/raw/ \
    --output_dir  data/presenters/bi/

# 输出：
#   data/presenters/bi/masks/        SAM2 mask (npz)
#   data/presenters/bi/clean_bg/     ProPainter 背景帧
#   data/presenters/bi/ref_frame.png 参考帧
```

ProPainter 部分切换环境：

```bash
conda activate propainter
python module_C_visual/propainter_wrapper.py \
    --video_dir  data/presenters/bi/raw/ \
    --mask_dir   data/presenters/bi/masks/ \
    --output_dir data/presenters/bi/clean_bg/
```

### Step 1：翻译视频（在线，一键）

```bash
conda activate wan_audio
python scripts/run_translate_video.py \
    --input         data/presenters/bi/raw/BI_001.mp4 \
    --presenter     bi \
    --target_lang   ja \
    --output        runs/BI_001_ja.mp4 \
    --mode          2 \
    --num_steps     25 \
    --audio_cfg     2.0 \
    --lip_strength  0.55 \
    --face_strength 0.35 \
    --body_strength 0.15
```

`--mode 1`：仅唇部区域扩散（更快，适合语速接近原始的翻译）
`--mode 2`：上半身完整重建（默认，适合语言差异大的翻译）

---

## 合成两种模式详解

### Mode 1：唇部局部替换

```
扩散区域：face_mask（仅人脸）
strength_map：唇 0.55 / 脸其余 0.20 / 其他 0.00
合成：clean_bg + orig_body + edited_face（仅换唇区）
适用：中→英、中→粤，语速节奏接近
```

### Mode 2：上半身完整重建（默认）

```
扩散区域：upper_body_mask（头+肩+上臂）
strength_map：唇 0.55 / 脸 0.35 / 上半身 0.15 / 背景 0.00
合成：clean_bg + orig_body + edited_upper_body
适用：中→日，节奏差异较大，需要自然头部动作配合
```

---

## Audio Adapter 结构

### 音频特征流向

```
new_audio.wav (16kHz)
  → Wav2Vec2 encoder (frozen)
  → audio_feat (T_audio, 768)                    ~50 vec/sec
  → split_audio_sequence(n_latent=21)
  → audio_windows (21, L_win, 768)               每 latent 帧一个窗口
  → AudioProjModel: Linear(768→2048) + LN
  → audio_cond (21, L_win, 2048)
  → WanAudioCrossAttentionProcessor（注入全部 40 个 DiT block）
      query: video latent tokens
      key/value: audio_cond（零初始化，gated by audio_scale.tanh()）
```

### 可训练参数

```
AudioProjModel:
  Linear(768, 2048)              1,572,864
  LayerNorm(2048)                    4,096

WanAudioCrossAttentionProcessor × 40 blocks:
  k_audio Linear(2048, 5120)   × 40   419,430,400
  v_audio Linear(2048, 5120)   × 40   419,430,400  ← 零初始化
  audio_scale (scalar)         × 40            40

合计可训练：~212M 参数（VACE 全部 14B 参数冻结）
```

---

## 微调方案（可选，提升唇同步质量）

FantasyTalking 权重作为初始化（I2V→VACE 维度兼容），在 BI 数据上精调：

```
Stage 1 — 全局音视频对齐
  数据：53 个 BI 视频（13.4h）
  Loss：全上半身区域 MSE
  步数：20K steps
  LR：1e-4
  时间：~12 小时（RTX 5090）

Stage 2 — 唇形精细对齐
  数据：同上
  Loss：lip region MSE + SyncNet perceptual × 0.1
  步数：10K steps
  LR：5e-5
  时间：~6 小时（RTX 5090）
```

```bash
# Stage 1
python scripts/train_audio_adapter.py \
    --stage 1 --steps 20000 --lr 1e-4 \
    --data_dir data/training/ \
    --init_ckpt data/models/fantasytalking_audio_adapter.ckpt \
    --output_dir runs/adapter_stage1/

# Stage 2
python scripts/train_audio_adapter.py \
    --stage 2 --steps 10000 --lr 5e-5 \
    --data_dir data/training/ \
    --init_ckpt runs/adapter_stage1/final.ckpt \
    --output_dir runs/adapter_stage2/
```

---

## 显存需求（重要！！）

### 实测数据（2025-04-02 RTX 5090 32GB 实机验证）

**VACE-14B 实际参数量为 17.3B（不是 14B）**，包含：
- 40 个 main blocks: 56.22 GB (BF16)
- 8 个 vace_blocks: 12.19 GB (BF16)
- 非 block 参数: 1.47 GB (BF16)
- **总计 BF16: 34.68 GB > 32GB VRAM → 无法全量加载**

| 阶段 | 实测显存 | 状态 |
|------|---------|------|
| 离线 SAM2 hiera_large | ~3 GB, 7.8fps@1080p | ✅ 跑通 |
| 离线 ProPainter | ~8 GB (需 0.5x 降分辨率到 540p, 1080p OOM) | ✅ 跑通 |
| 在线 Wav2Vec2 | ~2 GB | ✅ 跑通 |
| VACE-14B 加载（T5 CPU + DiT CPU + VAE GPU） | 0.53 GB | ✅ 加载成功 |
| VACE-14B 推理 model.to(cuda) | 34.68 GB → **OOM** | ❌ 无法全量 |
| 合成 + Poisson | CPU | ✅ 跑通 |

### 新机器最低显存要求

| 方案 | 最低 VRAM | 推荐 GPU |
|------|----------|---------|
| VACE-14B BF16 全量 | **48 GB** | A6000, L40S, A100-40G勉强 |
| VACE-14B BF16 + block offload | **40 GB**（需 flash_attn） | A100-40G, L40S |
| VACE-14B FP8 全量 | **~20 GB** | RTX 5090（如果 FP8 推理可用）|
| VACE-14B 4bit (bitsandbytes) | **~12 GB** | RTX 4090/5090 |
| VACE-1.3B BF16 全量 | **~4 GB** | 任何 GPU（先跑通管线）|

### RTX 5090 (32GB) 遇到的具体问题

1. **VACE DiT 17.3B BF16 = 34.68GB > 32GB**：`model.to('cuda')` 直接 OOM
2. **flash_attn 编译失败**：RTX 5090 是 Blackwell 架构 sm_120，flash_attn pip 包不支持，需要从源码编译（需要 CUDA 12.8 + 正确的 nvcc）
3. **手动 block offload**（每个 block ~1.4GB 逐个搬到 GPU）：因为 flash_attn 不可用，写了 SDPA fallback 替代，但 varlen attention 格式转换有 bug，进程被 kill (exit 144)
4. **accelerate dispatch_model**：产生 meta tensor 问题，`Cannot copy out of meta tensor`
5. **accelerate cpu_offload**：同样 meta tensor 问题
6. **ProPainter 1080p OOM**：RAFT 光流计算在 1080p 时需要 ~8GB 额外显存，需要 `--resize_ratio 0.5` 降到 540p

### 推荐方案（新机器）

**方案 A（推荐）：48GB+ GPU**
- A6000 (48GB) 或 A100 (40/80GB) 或 L40S (48GB)
- BF16 全量加载，无需量化，无需 offload
- 预计推理速度：~30s/step (480p, 81帧)

**方案 B：32GB GPU (RTX 5090/4090)**
1. 安装 flash_attn（从源码编译，确保 sm_120 支持）
2. 手动 block offload：非 block 参数在 GPU (1.47GB)，每个 block 推理时搬到 GPU (~1.4GB)
3. 或者用 4-bit 量化：`bitsandbytes` 把 17.3B 量化到 ~8.7GB

**方案 C：先用 1.3B 小模型跑通**
- `Wan-AI/Wan2.1-VACE-1.3B` BF16 ≈ 2.6GB
- 任何 GPU 都能跑
- 先验证管线逻辑，再换 14B

### 速度优化选项（等 VRAM 问题解决后可叠加）

```bash
--num_steps 15         # 质量略降，速度 1.7×
--teacache 0.15        # TeaCache，速度 1.3–1.5×，官方支持
--t5_cpu               # T5 放 CPU，省 ~11GB 显存
--offload_model True   # block offload，速度降但省显存
```

---

## 合成流程顺序（不可乱）

```
① ProPainter 背景修复（离线）
     输入：orig_frames + upper_body_mask（膨胀 12px）
     输出：clean_bg（无人区域）

② VACE 扩散生成（在线）
     输入：orig_frames + strength_map + audio_cond + ref_frame
     输出：edited_frames（mask 内区域已重建）

③ 三层 alpha 合成
     Layer 0: clean_bg
     Layer 1: orig_frame × (1 - soft_mask)   身体区域
     Layer 2: edited_frame × soft_mask        新脸区域

④ Poisson seamlessClone（边界色调融合）

⑤ Kalman landmark 平滑（抑制帧间抖动）

⑥ LAB 色调归一化（消除 VACE 输出色偏）

⑦ SyncNet QA（sync_score < 3.0 报警）

⑧ FFmpeg mux → output.mp4
```

---

## 当前进度（2025-04-02）

### 已验证通过

| 步骤 | 状态 | 备注 |
|------|------|------|
| 项目结构 + 所有模块代码 | ✅ | 完整实现 |
| wan_audio conda 环境 | ✅ | PyTorch 2.11.0+cu128, diffusers, transformers, sam2, insightface, mediapipe, opencv |
| 模型权重下载 | ✅ | VACE-14B (70GB), wav2vec2 (1.1GB), sam2 (1.7GB), FantasyTalking (3.2GB) |
| SAM2 mask tracking | ✅ | 250帧@1080p, 7.8fps, 双mask(face+upper_body), 输出npz |
| InsightFace 人脸检测 | ✅ | buffalo_l 模型自动下载到 ~/.insightface/ |
| ref_frame 自动选取 | ✅ | 按人脸面积选最优正面帧 |
| ProPainter 背景修复 | ✅ | 需要 `--resize_ratio 0.5`（1080p OOM），结果在 clean_bg/frames/ |
| VACE 模型加载 | ✅ | wan.WanVace + t5_cpu=True，GPU仅占0.53GB(VAE) |
| VACE 音频适配器安装 | ✅ | 40个block全部安装 WanAudioCrossAttentionProcessor |
| 音频管线 B1/B2/B3 | ⏳ stub | FunASR/翻译/CosyVoice 接口已定义，实现待接入 |
| **VACE 推理** | **❌ 阻塞** | **17.3B BF16=34.7GB > 32GB VRAM** |
| 合成管线 F1-F4 | ✅ 代码 | 已实现，未实际跑通（依赖 VACE 输出） |
| SyncNet QA / FFmpeg mux | ✅ 代码 | 已实现 |

### 下一步（新机器上）

1. **解决 VRAM 问题**（见上方显存需求章节）
2. 跑通 VACE text-only V2V 推理（不加音频，纯验证 masked editing）
3. 接入音频 cross-attention（让 WanAudioCrossAttentionProcessor 在 DiT forward 中生效）
4. 端到端跑通 run_translate_video.py
5. 接入 CosyVoice 语音克隆

### 第三方代码修改记录

以下文件已被修改以适配本项目，换机器后需要重新 apply：

**third_party/Wan2.1/wan/vace.py**（核心修改）:

- `__init__` 第 121-136 行：替换 `self.model.to(self.device)` 为手动 block offload 逻辑（非 block 参数上 GPU，blocks 留 CPU）
- `generate` 第 389-398 行：VAE encode 后释放 VAE 显存 + z 移到 CPU
- `generate` 第 447-449 行：去噪循环前把 context 搬回 GPU
- `generate` 第 459-473 行：每步把 z 搬到 GPU，推理完释放
- `generate` 第 487-488 行：去噪完把 VAE 搬回 GPU 做 decode

**third_party/Wan2.1/wan/modules/vace_model.py**（block offload + device alignment）:

- `forward` 第 193-200 行：所有输入 tensor 对齐到 patch_embedding 设备
- `forward` 第 252-255 行：main blocks 循环改为逐 block 搬 GPU→推理→搬回 CPU
- `forward_vace` 第 137-158 行：vace_blocks 同上处理

**third_party/Wan2.1/wan/modules/attention.py**（flash_attn fallback）:

- 第 111-139 行：当 flash_attn 2/3 都不可用时，fallback 到 `F.scaled_dot_product_attention`
- ⚠️ **此 fallback 有 bug**：varlen attention 的 unflatten 格式转换不正确，是进程 crash 的直接原因
- 新机器上如果能装 flash_attn 就不需要这个 fallback

### SAM2 使用注意事项

- SAM2 pip 包 (`sam2==1.1.0`) 要求帧为 **JPEG 格式**（不支持 PNG）
- 文件名格式：`%06d.jpg`（如 000001.jpg）
- Hydra config 名：`sam2_hiera_l.yaml`（不是 `sam2.1_hiera_l.yaml`，不是 `sam2_hiera_large.yaml`）
- `init_state()` 不是 context manager，直接返回 state 对象
- `add_new_points_or_box()` 是正确的 API（不是 `add_new_prompts`）
- `propagate_in_video()` 返回的 logits 是 3D `(1, H, W)`，需要 `.squeeze()` 再用

### ProPainter 使用注意事项

- 1080p 全分辨率在 32GB GPU 上 OOM（RAFT 光流需要 ~8GB 额外显存）
- 必须用 `--resize_ratio 0.5` 降到 540p
- 权重自动从 GitHub releases 下载到 `third_party/ProPainter/weights/`
- 可以复用 `wan_audio` 环境（装了 einops, timm, decord 后就行）

### Wan2.1 VACE 加载方式

```python
# 正确的加载方式（使用 wan 原生 API，不是 diffusers）
import wan
from wan.configs import WAN_CONFIGS

model = wan.WanVace(
    config=WAN_CONFIGS['vace-14B'],
    checkpoint_dir='/abs/path/to/Wan2.1-VACE-14B',  # 必须绝对路径
    device_id=0, rank=0,
    t5_fsdp=False, dit_fsdp=False, use_usp=False,
    t5_cpu=True,  # T5 放 CPU 省 ~11GB
)

# 模型目录结构（非 diffusers 格式，没有 model_index.json）:
# Wan2.1-VACE-14B/
#   config.json                              # VaceWanModel config
#   diffusion_pytorch_model-0000X-of-00007.safetensors  # 7 个 shard
#   models_t5_umt5-xxl-enc-bf16.pth         # T5 text encoder
#   Wan2.1_VAE.pth                           # VAE
#   google/umt5-xxl/                         # T5 tokenizer

# 推理（offload_model=True 在每步搬整个 model 到 GPU）
result = model.generate(
    input_prompt='...',
    input_frames=src_video,     # prepare_source() 的输出
    input_masks=src_mask,
    input_ref_images=src_ref_images,
    size=(832, 480),
    frame_num=81,               # 必须满足 (N-1) % 4 == 0
    sampling_steps=25,
    guide_scale=5.0,
    offload_model=True,         # 32GB 必须开
    seed=42,
)
```

---

## 已知限制与处理

| 问题 | 原因 | 当前处理 |
|------|------|---------|
| 转头 >30° 时脸部畸变 | ref_frame 固定正面 | 按 chunk 动态选 ref_frame（TODO）|
| chunk 边界轻微闪烁 | 相邻 chunk 噪声不同 | Hamming 窗口融合（已实现）|
| 眼神轻微漂移 | face_strength=0.35 | 眼部单独设 0.10（可在 strength_map 调）|
| 快语速唇形不精确 | audio window 不够长 | 调 `expand=8`（默认 4）|
| 色调偏暖/偏冷 | VACE 生成色域偏移 | LAB 直方图匹配（F4 已实现）|

---

## 参考项目

| 项目 | 用途 | 我们使用的部分 |
|------|------|--------------|
| [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | VACE backbone | `generate.py --task vace-14B`，VAE，DiT |
| [Fantasy-AMAP/fantasy-talking](https://github.com/Fantasy-AMAP/fantasy-talking) | 音频 adapter | `AudioProjModel`，`WanCrossAttentionProcessor`，`split_audio_sequence` |
| [Omni-Avatar/OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar) | Audio Pack 参考 | 多层分级注入思路 |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | mask 追踪 | `sam2_video_predictor.propagate_in_video` |
| [sczhou/ProPainter](https://github.com/sczhou/ProPainter) | 背景修复 | `inference_propainter.py` |
| [exx8/differential-diffusion](https://github.com/exx8/differential-diffusion) | 区域加噪 | per-pixel strength map |
| [yangqy1110/NC-SDEdit](https://github.com/yangqy1110/NC-SDEdit) | chunk 融合 | Hamming window blending |
| [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | TTS | zero-shot 声音克隆 |
| [modelscope/FunASR](https://github.com/modelscope/FunASR) | ASR | 中文转写 |
| [joonson/syncnet_python](https://github.com/joonson/syncnet_python) | QA | Sync-C / Sync-D 指标 |

---

## 参考文献

```bibtex
@article{vace2025,
  title={VACE: All-in-One Video Creation and Editing},
  journal={arXiv:2503.07598}, year={2025}
}
@article{wang2025fantasytalking,
  title={FantasyTalking: Realistic Talking Portrait Generation
         via Coherent Motion Synthesis},
  journal={arXiv:2504.04842}, year={2025}
}
@misc{gan2025omniavatar,
  title={OmniAvatar: Efficient Audio-Driven Avatar Video Generation
         with Adaptive Body Animation},
  eprint={2506.18866}, year={2025}
}
@article{levin2023differential,
  title={Differential Diffusion: Giving Each Pixel Its Strength},
  journal={arXiv:2306.00950}, year={2023}
}
@inproceedings{yang2024ncsdedit,
  title={Noise Calibration: Plug-and-play Content-Preserving Video Enhancement},
  booktitle={ECCV}, year={2024}
}
@inproceedings{zhou2023propainter,
  title={ProPainter: Improving Propagation and Transformer for Video Inpainting},
  booktitle={ICCV}, year={2023}
}
```
