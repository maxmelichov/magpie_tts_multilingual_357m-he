#!/usr/bin/env python3
"""Synthesize Hebrew speech with a fine-tuned MagpieTTS checkpoint.

Builds a one-off manifest + evalset config from IPA input and delegates to
NeMo's examples/tts/magpietts_inference.py.

Examples:
  # single IPA sentence, cloning the voice of a context wav
  venv/bin/python scripts/infer_hebrew.py \
      --checkpoint experiments/Magpie-TTS/<run>/checkpoints/last.ckpt \
      --hparams experiments/Magpie-TTS/<run>/hparams.yaml \
      --context-audio /home/maxm/AE_training_data_all/generated_audio/voice1_high_quality/voice1_knesset_012062.wav \
      --text "ʔanˈaχnu beʔˈad sifʁijˈa leʔumˈit." \
      --out-dir outputs/test1

  # batch: an existing NeMo-format manifest (audio_filepath ignored for synthesis)
  venv/bin/python scripts/infer_hebrew.py --checkpoint ... --hparams ... \
      --manifest data/manifests/voice1_val.json --audio-dir <dir> --out-dir outputs/val
"""

import argparse
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = "hebrew_chartokenizer"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help=".ckpt from fine-tuning (or a .nemo file)")
    ap.add_argument("--hparams", help="hparams.yaml from the training run (required with a .ckpt)")
    ap.add_argument("--text", action="append", default=[], help="IPA phoneme string to synthesize (repeatable)")
    ap.add_argument("--manifest", help="alternatively, a NeMo manifest of lines with 'text' (IPA)")
    ap.add_argument("--audio-dir", help="audio_dir for --manifest (context wavs resolved against it)")
    ap.add_argument("--context-audio", help="reference wav for voice cloning (used with --text)")
    ap.add_argument("--context-text", default="", help="IPA transcript of the context wav (optional)")
    ap.add_argument("--out-dir", default="outputs/infer")
    ap.add_argument("--tokenizer", default=TOKENIZER,
                    help="tokenizer name from the checkpoint (default hebrew_chartokenizer)")
    ap.add_argument("--codec", default="nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps")
    ap.add_argument("--gpu", default=os.environ.get("GPU", "1"), help="CUDA device id (default 1; cuda:0 busy)")
    # Quality knobs. The model is trained with a local transformer (multi-codebook
    # refinement) and with CFG dropout, but NeMo's inference script leaves both off
    # by default -- enabling them audibly improves naturalness.
    ap.add_argument("--use-cfg", action=argparse.BooleanOptionalAction, default=True,
                    help="classifier-free guidance (default on)")
    ap.add_argument("--use-local-transformer", action=argparse.BooleanOptionalAction, default=True,
                    help="multi-codebook refinement (default on)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="remaining args passed through to magpietts_inference.py")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        manifest_path = Path(args.manifest).absolute()
        audio_dir = Path(args.audio_dir or Path(args.manifest).parent).absolute()
    else:
        if not args.text or not args.context_audio:
            sys.exit("need --text and --context-audio (or --manifest)")
        ctx = Path(args.context_audio).absolute()
        with wave.open(str(ctx), "rb") as w:
            ctx_dur = w.getnframes() / w.getframerate()
        audio_dir = ctx.parent
        manifest_path = out_dir / "input_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for text in args.text:
                f.write(json.dumps({
                    "audio_filepath": ctx.name,      # placeholder target (required field)
                    "text": text,
                    "duration": round(ctx_dur, 3),
                    "context_audio_filepath": ctx.name,
                    "context_text": args.context_text or text,
                    "context_audio_duration": round(ctx_dur, 3),
                }, ensure_ascii=False) + "\n")

    evalset = {
        "hebrew_infer": {
            "manifest_path": str(manifest_path),
            "audio_dir": str(audio_dir),
            "feature_dir": str(audio_dir),
            "tokenizer_names": [args.tokenizer],
        }
    }
    evalset_path = out_dir / "evalset_config.json"
    evalset_path.write_text(json.dumps(evalset, ensure_ascii=False, indent=2))

    cmd = [
        str(REPO_ROOT / "venv/bin/python"), "examples/tts/magpietts_inference.py",
        "--model_type", "magpie",
        "--datasets_json_path", str(evalset_path),
        "--out_dir", str(out_dir),
        "--codecmodel_path", args.codec,
    ]
    if args.use_cfg:
        cmd.append("--use_cfg")
    if args.use_local_transformer:
        cmd.append("--use_local_transformer")
    cmd += args.extra

    if args.checkpoint.endswith(".nemo"):
        cmd += ["--nemo_files", str(Path(args.checkpoint).absolute())]
    else:
        if not args.hparams:
            sys.exit("--hparams is required when --checkpoint is a .ckpt")
        cmd += ["--checkpoint_files", str(Path(args.checkpoint).absolute()),
                "--hparams_files", str(Path(args.hparams).absolute())]

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    print("+", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT / "NeMo", env=env))


if __name__ == "__main__":
    main()
