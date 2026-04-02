"""run_translate_video.py — One-click video translation (Mode 1 / Mode 2).

Usage:
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

Mode 1: lip-only local replacement (faster, for similar speech rhythm)
Mode 2: upper-body complete rebuild (default, for cross-language)

Pipeline order (§合成流程顺序):
  ① ProPainter clean_bg (offline, pre-computed)
  ② VACE diffusion + audio conditioning
  ③ Three-layer alpha compositing
  ④ Poisson seamlessClone
  ⑤ Kalman landmark smoothing
  ⑥ LAB color harmonization
  ⑦ SyncNet QA
  ⑧ FFmpeg mux
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def extract_frames(video: str, out_dir: str, fps: int = 25):
    os.makedirs(out_dir, exist_ok=True)
    ret = os.system(
        f'ffmpeg -i "{video}" -vf fps={fps} -q:v 2 '
        f'"{out_dir}/%06d.png" -y -hide_banner -loglevel error'
    )
    if ret != 0:
        raise RuntimeError("ffmpeg frame extraction failed")


def load_frames(d: str) -> list[np.ndarray]:
    files = sorted(f for f in os.listdir(d) if f.endswith('.png'))
    return [cv2.imread(os.path.join(d, f)) for f in files]


def save_frames(frames: list[np.ndarray], d: str):
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(d, f'{i:06d}.png'), f)


def load_masks_npz(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser(description='Video translation pipeline')
    parser.add_argument('--input', required=True, help='Input video')
    parser.add_argument('--presenter', default='bi')
    parser.add_argument('--target_lang', default='ja')
    parser.add_argument('--output', default='runs/translated.mp4')
    parser.add_argument('--mode', type=int, choices=[1, 2], default=2)
    parser.add_argument('--work_dir', default=None)
    parser.add_argument('--fps', type=int, default=25)
    # Diffusion params
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--audio_cfg', type=float, default=2.0)
    parser.add_argument('--lip_strength', type=float, default=0.55)
    parser.add_argument('--face_strength', type=float, default=0.35)
    parser.add_argument('--body_strength', type=float, default=0.15)
    parser.add_argument('--chunk_size', type=int, default=81)
    parser.add_argument('--overlap', type=int, default=13)
    parser.add_argument('--expand', type=int, default=4,
                        help='Audio window expansion (increase for fast speech)')
    # Optimization
    parser.add_argument('--t5_cpu', action='store_true')
    parser.add_argument('--offload_model', action='store_true')
    parser.add_argument('--teacache', type=float, default=0.0,
                        help='TeaCache threshold (0=off, 0.15=recommended)')
    args = parser.parse_args()

    work = args.work_dir or tempfile.mkdtemp(prefix='vtrans_')
    os.makedirs(work, exist_ok=True)
    print(f"Work dir: {work}")

    presenter_dir = os.path.join('data', 'presenters', args.presenter)

    # ── 1. Extract frames ────────────────────────────────────
    print("\n[1/8] Extracting frames...")
    frames_dir = os.path.join(work, 'frames')
    extract_frames(args.input, frames_dir, args.fps)
    orig_frames = load_frames(frames_dir)
    print(f"  {len(orig_frames)} frames @ {args.fps}fps")

    # ── 2. Audio pipeline ────────────────────────────────────
    print("\n[2/8] Audio pipeline (ASR → translate → TTS)...")
    audio_dir = os.path.join(work, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    new_audio = os.path.join(audio_dir, 'new_audio.wav')

    try:
        from module_B_audio.B1_asr.transcribe import transcribe
        from module_B_audio.B2_translate.translate import translate
        from module_B_audio.B3_tts.synthesize import synthesize

        asr_out = os.path.join(audio_dir, 'asr.json')
        trans_out = os.path.join(audio_dir, 'translation.json')
        transcribe(args.input, asr_out)
        translate(asr_out, trans_out, target_lang=args.target_lang)
        voice_prompt = os.path.join(presenter_dir, 'voice_prompt.pt')
        synthesize(trans_out, new_audio, voice_prompt=voice_prompt)
    except (ImportError, NotImplementedError) as e:
        print(f"  [skip] {e}")
        if not os.path.isfile(new_audio):
            print(f"  Extracting original audio as fallback...")
            os.system(f'ffmpeg -i "{args.input}" -vn -ar 16000 -ac 1 '
                      f'"{new_audio}" -y -hide_banner -loglevel error')

    # ── 3. Load pre-computed masks ───────────────────────────
    print("\n[3/8] Loading masks...")
    npz_path = os.path.join(presenter_dir, 'masks', 'masks.npz')
    if os.path.isfile(npz_path):
        mask_data = load_masks_npz(npz_path)
        face_masks = mask_data['face_mask']        # (T, H, W) uint8
        ub_masks = mask_data['upper_body_mask']     # (T, H, W) uint8
        print(f"  Loaded {face_masks.shape[0]} masks from npz")
    else:
        print(f"  [error] Masks not found at {npz_path}")
        print(f"  Run: python scripts/preprocess_presenter.py --presenter {args.presenter}")
        return

    # ── 4. Build strength map ────────────────────────────────
    print(f"\n[4/8] Building strength map (mode {args.mode})...")
    from module_C_visual.mask_utils import build_strength_map

    # Use first frame's mask as representative
    smap = build_strength_map(
        face_masks[0], ub_masks[0],
        mode=args.mode,
        lip_strength=args.lip_strength,
        face_strength=args.face_strength,
        body_strength=args.body_strength,
    )
    print(f"  Strength range: [{smap.min():.2f}, {smap.max():.2f}]")

    # ── 5. VACE + audio diffusion ────────────────────────────
    print(f"\n[5/8] VACE diffusion ({args.num_steps} steps)...")
    from module_D_diffusion.vace_audio_pipeline import VACEAudioPipeline

    ref_frame_path = os.path.join(presenter_dir, 'ref_frame.png')
    ref_frame = cv2.imread(ref_frame_path)
    if ref_frame is None:
        print(f"  [warn] ref_frame not found, using frame 0")
        ref_frame = orig_frames[0]

    pipeline = VACEAudioPipeline(
        vace_model_path='data/models/Wan2.1-VACE-14B',
        wav2vec2_path='data/models/wav2vec2-base-960h',
        ft_checkpoint='data/models/fantasytalking_audio_adapter.ckpt',
        t5_cpu=args.t5_cpu,
        offload_model=args.offload_model,
    )

    # Select masks based on mode
    edit_masks = [face_masks[i] for i in range(len(orig_frames))] if args.mode == 1 \
        else [ub_masks[i] for i in range(len(orig_frames))]

    edited_frames = pipeline.run_long_video(
        src_frames=orig_frames,
        src_masks=edit_masks,
        audio_path=new_audio,
        ref_frame=ref_frame,
        strength_map=smap,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        num_steps=args.num_steps,
        audio_cfg=args.audio_cfg,
    )

    # ── 6. Three-layer compositing ───────────────────────────
    print("\n[6/8] Compositing...")
    from module_F_blending.F1_composite import composite_mode1, composite_mode2

    composited = []
    for i in range(len(orig_frames)):
        mask = ub_masks[i] if args.mode == 2 else face_masks[i]
        if args.mode == 1:
            c = composite_mode1(orig_frames[i], edited_frames[i], mask)
        else:
            c = composite_mode2(edited_frames[i], orig_frames[i], mask)
        composited.append(c)

    # ── 6b. Poisson boundary blend ───────────────────────────
    from module_F_blending.F2_poisson import poisson_blend

    for i in range(len(composited)):
        mask = ub_masks[i] if args.mode == 2 else face_masks[i]
        composited[i] = poisson_blend(edited_frames[i], composited[i], mask)

    # ── 6c. Kalman smoothing (Mode 1 only) ───────────────────
    if args.mode == 1:
        print("  Kalman smoothing...")
        from module_F_blending.F3_kalman import kalman_smooth_frames
        composited = kalman_smooth_frames(
            composited, [face_masks[i] for i in range(len(composited))])

    # ── 6d. LAB color harmonization ──────────────────────────
    print("  Color harmonization...")
    from module_F_blending.F4_color_harmonize import histogram_match_region

    for i in range(len(composited)):
        mask = ub_masks[i] if args.mode == 2 else face_masks[i]
        composited[i] = histogram_match_region(composited[i], orig_frames[i], mask)

    composite_dir = os.path.join(work, 'composite')
    save_frames(composited, composite_dir)

    # ── 7. SyncNet QA ────────────────────────────────────────
    print("\n[7/8] SyncNet QA...")
    try:
        from module_G_postprocess.G1_syncnet_qa import syncnet_qa
        result = syncnet_qa(composite_dir, new_audio, threshold=3.0)
    except (ImportError, NotImplementedError):
        print("  [skip] SyncNet not available")

    # ── 8. FFmpeg mux ────────────────────────────────────────
    print("\n[8/8] Muxing output...")
    from module_G_postprocess.G2_mux import mux_output
    mux_output(composite_dir, new_audio, args.output, fps=args.fps, crf=18)

    print(f"\n✓ Done: {args.output}")
    print(f"  Work dir: {work}")


if __name__ == '__main__':
    main()
