"""run_translate_video.py — One-click video translation entry point.

Usage:
    python scripts/run_translate_video.py \\
        --input data/raw/bi/BI_0112.mp4 \\
        --presenter bi \\
        --target_lang ja \\
        --mode 1 \\
        --output output/BI_0112_ja.mp4

Modes:
    1 = FantasyTalking I2V (fast, ready now)
    2 = VACE masked V2V + audio (precise, requires adapter integration)
"""

import argparse
import os
import sys
import tempfile

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


def main():
    parser = argparse.ArgumentParser(description='Video translation pipeline')
    parser.add_argument('--input', required=True, help='Input video')
    parser.add_argument('--presenter', default='bi', help='Presenter ID')
    parser.add_argument('--target_lang', default='ja')
    parser.add_argument('--mode', type=int, choices=[1, 2], default=1)
    parser.add_argument('--output', default='output/translated.mp4')
    parser.add_argument('--work_dir', default=None)
    parser.add_argument('--fps', type=int, default=25)
    # Mode 1 params
    parser.add_argument('--audio_cfg', type=float, default=4.0)
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--chunk_size', type=int, default=81)
    parser.add_argument('--overlap', type=int, default=13)
    # Mode 2 params
    parser.add_argument('--strength_lips', type=float, default=0.50)
    parser.add_argument('--strength_face', type=float, default=0.35)
    parser.add_argument('--strength_shoulder', type=float, default=0.15)
    parser.add_argument('--num_steps', type=int, default=25)
    args = parser.parse_args()

    work = args.work_dir or tempfile.mkdtemp(prefix='vtrans_')
    os.makedirs(work, exist_ok=True)
    print(f"Work dir: {work}")

    frames_dir = os.path.join(work, 'frames')
    masks_dir = os.path.join(work, 'masks')
    audio_dir = os.path.join(work, 'audio')
    gen_dir = os.path.join(work, 'gen')
    composite_dir = os.path.join(work, 'composite')

    ref_frame = f'data/presenters/bi_ref_frames/{args.presenter.upper()}_ref.png'

    # ── 1. Extract frames ────────────────────────────────────
    print("\n[1/6] Extracting frames...")
    extract_frames(args.input, frames_dir, args.fps)
    n_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    print(f"  {n_frames} frames @ {args.fps}fps")

    # ── 2. Audio pipeline (ASR → translate → TTS) ────────────
    print("\n[2/6] Audio pipeline...")
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
        synthesize(trans_out, new_audio,
                   voice_prompt=f'data/presenters/{args.presenter}/voice_prompt.pt')
    except (ImportError, NotImplementedError) as e:
        print(f"  [skip] Audio pipeline not ready: {e}")
        print(f"  Provide audio manually: {new_audio}")
        if not os.path.isfile(new_audio):
            # Extract original audio as fallback
            os.system(f'ffmpeg -i "{args.input}" -vn -ar 16000 -ac 1 '
                      f'"{new_audio}" -y -hide_banner -loglevel error')

    # ── 3. SAM2 mask tracking ────────────────────────────────
    print("\n[3/6] SAM2 mask tracking...")
    try:
        from module_C_visual.C2_segment.sam2_tracker import FaceTracker
        tracker = FaceTracker()
        masks = tracker.track(frames_dir, dilate_px=10)
        for key in ('face_mask', 'upper_body_mask'):
            out = os.path.join(masks_dir, key)
            os.makedirs(out, exist_ok=True)
            import cv2
            for i, m in enumerate(masks[key]):
                cv2.imwrite(os.path.join(out, f'{i:06d}.png'), m)
    except (ImportError, RuntimeError) as e:
        print(f"  [skip] SAM2 not available: {e}")
        print(f"  Provide masks at: {masks_dir}/upper_body_mask/")

    ub_mask_dir = os.path.join(masks_dir, 'upper_body_mask')
    face_mask_dir = os.path.join(masks_dir, 'face_mask')

    # ── 4. Upper body re-generation ──────────────────────────
    print(f"\n[4/6] Mode {args.mode} generation...")

    if args.mode == 1:
        # FantasyTalking I2V
        ft_dir = os.path.join(ROOT, 'module_D_diffusion', 'D1_fantasytalking',
                              'fantasy-talking')
        infer_script = os.path.join(ft_dir, 'infer.py')

        if os.path.isfile(infer_script):
            os.makedirs(gen_dir, exist_ok=True)
            cmd = (
                f'python "{infer_script}"'
                f' --image_path "{ref_frame}"'
                f' --audio_path "{new_audio}"'
                f' --prompt "a presenter talking, upper body shot, natural gestures"'
                f' --num_inference_steps {args.num_inference_steps}'
                f' --audio_cfg {args.audio_cfg}'
                f' --output_path "{gen_dir}"'
            )
            print(f"  Running: {cmd}")
            ret = os.system(cmd)
            if ret != 0:
                print("  [error] FantasyTalking inference failed")
                return
        else:
            print(f"  [error] FantasyTalking not found at {ft_dir}")
            print(f"  Clone: git clone https://github.com/Fantasy-AMAP/fantasy-talking.git {ft_dir}")
            return

    elif args.mode == 2:
        # VACE + audio
        try:
            from module_D_diffusion.D2_vace.vace_audio_pipeline import (
                build_vace_audio_model, run_vace_audio,
            )
            from module_C_visual.C2_segment.mask_utils import build_region_strength_map
            import cv2
            import numpy as np

            # Build strength map from first frame's masks
            face_m = cv2.imread(os.path.join(face_mask_dir, '000001.png'),
                                cv2.IMREAD_GRAYSCALE)
            ub_m = cv2.imread(os.path.join(ub_mask_dir, '000001.png'),
                              cv2.IMREAD_GRAYSCALE)
            smap = build_region_strength_map(
                face_m, ub_m,
                strength_lips=args.strength_lips,
                strength_face=args.strength_face,
                strength_shoulder=args.strength_shoulder,
            )

            vace_ckpt = os.path.join(ROOT, 'module_D_diffusion', 'D2_vace',
                                     'VACE-Wan2.1-14B')
            ft_ckpt = os.path.join(ft_dir, 'models', 'fantasytalking_model.ckpt')

            pipe, audio_proj = build_vace_audio_model(vace_ckpt, ft_ckpt)
            frames = run_vace_audio(
                pipe, audio_proj,
                args.input, ub_mask_dir, new_audio, ref_frame, smap,
                num_steps=args.num_steps,
            )
            os.makedirs(gen_dir, exist_ok=True)
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(gen_dir, f'{i:06d}.png'), f)
        except (ImportError, FileNotFoundError) as e:
            print(f"  [error] VACE pipeline not ready: {e}")
            return

    # ── 5. Compositing ───────────────────────────────────────
    print("\n[5/6] Compositing...")
    os.makedirs(composite_dir, exist_ok=True)

    try:
        from module_F_blending.F1_composite import composite_mode1, composite_mode2
        from module_F_blending.F2_poisson import poisson_blend
        from module_F_blending.F4_color_harmonize import histogram_match_region
        import cv2

        gen_files = sorted(os.listdir(gen_dir))
        orig_files = sorted(os.listdir(frames_dir))
        mask_files = sorted(os.listdir(ub_mask_dir)) if os.path.isdir(ub_mask_dir) else []

        n = min(len(gen_files), len(orig_files), len(mask_files)) if mask_files else len(gen_files)

        for i in range(n):
            gen = cv2.imread(os.path.join(gen_dir, gen_files[i]))
            orig = cv2.imread(os.path.join(frames_dir, orig_files[i]))
            mask = cv2.imread(os.path.join(ub_mask_dir, mask_files[i]),
                              cv2.IMREAD_GRAYSCALE) if mask_files else None

            if args.mode == 1 and mask is not None:
                out = composite_mode1(orig, gen, mask)
                out = histogram_match_region(out, orig, mask)
                out = poisson_blend(gen, out, mask)
            elif args.mode == 2 and mask is not None:
                out = composite_mode2(gen, orig, mask)
            else:
                out = gen

            cv2.imwrite(os.path.join(composite_dir, f'{i:06d}.png'), out)

        # Kalman smoothing (Mode 1 only)
        if args.mode == 1 and os.path.isdir(face_mask_dir):
            print("  Kalman smoothing...")
            from module_F_blending.F3_kalman import kalman_smooth_frames
            comp_frames = [cv2.imread(os.path.join(composite_dir, f))
                           for f in sorted(os.listdir(composite_dir))]
            face_masks = [cv2.imread(os.path.join(face_mask_dir, f), cv2.IMREAD_GRAYSCALE)
                          for f in sorted(os.listdir(face_mask_dir))[:len(comp_frames)]]
            smoothed = kalman_smooth_frames(comp_frames, face_masks)
            for i, f in enumerate(smoothed):
                cv2.imwrite(os.path.join(composite_dir, f'{i:06d}.png'), f)

        # ProPainter boundary seam repair
        if os.path.isdir(ub_mask_dir):
            print("  Boundary seam repair...")
            try:
                from module_C_visual.C2_segment.mask_utils import boundary_seam_mask
                from module_C_visual.C3_inpaint.propainter_wrapper import inpaint_boundary_seam

                seam_dir = os.path.join(work, 'seam_masks')
                os.makedirs(seam_dir, exist_ok=True)
                for f in mask_files[:n]:
                    m = cv2.imread(os.path.join(ub_mask_dir, f), cv2.IMREAD_GRAYSCALE)
                    seam = boundary_seam_mask(m, width=20)
                    cv2.imwrite(os.path.join(seam_dir, f), seam)

                repaired_dir = os.path.join(work, 'repaired')
                inpaint_boundary_seam(composite_dir, seam_dir, repaired_dir)
                composite_dir = repaired_dir
            except (ImportError, FileNotFoundError) as e:
                print(f"  [skip] ProPainter boundary repair: {e}")

    except ImportError as e:
        print(f"  [skip] Compositing modules not available: {e}")

    # ── 6. QA + Mux ──────────────────────────────────────────
    print("\n[6/6] QA + Mux...")
    try:
        from module_G_postprocess.G1_syncnet_qa import syncnet_qa
        syncnet_qa(composite_dir, new_audio, threshold=3.0)
    except (ImportError, NotImplementedError):
        print("  [skip] SyncNet QA")

    from module_G_postprocess.G2_mux import mux_output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    mux_output(composite_dir, new_audio, args.output, fps=args.fps)

    print(f"\n✓ Done: {args.output}")
    print(f"  Work dir: {work}")


if __name__ == '__main__':
    main()
