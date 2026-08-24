#!/usr/bin/env python3
"""Fold LoRA weights from a magpietts_lora.py checkpoint into the base weights.

Produces a checkpoint loadable by the stock MagpieTTS architecture (for
examples/tts/magpietts_inference.py with the run's hparams.yaml).

Usage:
  venv/bin/python scripts/merge_lora.py \
      --ckpt experiments/Magpie-TTS/<run>/checkpoints/<best>.ckpt \
      --out  experiments/Magpie-TTS/<run>/checkpoints/merged.ckpt
"""

import argparse

from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-weights", default="checkpoints/extracted/model_weights.ckpt",
                    help="release weights, used to restore the 5 original baked speakers exactly")
    ap.add_argument("--keep-base-speakers", type=int, default=5,
                    help="restore this many leading baked speaker rows from --base-weights (0 to skip)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    merged = 0
    for a_key in [k for k in list(sd) if k.endswith(".lora_A")]:
        prefix = a_key[: -len(".lora_A")]
        A, B = sd.pop(a_key), sd.pop(prefix + ".lora_B")
        w_key = prefix + ".weight"
        # scaling = alpha / r; recover from stored hyperparams if present, else default 32/16
        hp = ckpt.get("hyper_parameters", {}) or {}
        lora_hp = hp.get("lora", {}) if isinstance(hp, dict) else {}
        scaling = float(lora_hp.get("alpha", 32)) / float(lora_hp.get("r", A.shape[0]))
        sd[w_key] = sd[w_key] + (B @ A) * scaling
        merged += 1

    if merged == 0:
        raise SystemExit("no lora_A keys found — is this a LoRA checkpoint?")

    # Save a minimal plain-tensor checkpoint: NeMo's inference loader uses
    # torch.load(weights_only=True), which rejects pickled OmegaConf objects.
    # Restore the released speaker voices bit-for-bit. Their gradients are masked
    # during training, but AdamW's decoupled weight decay still shrinks every
    # parameter it owns -- measured at a uniform 0.998 norm ratio per 2k steps,
    # direction untouched (cosine 1.000000). Harmless, but there is no reason to
    # ship Aria/Jason/John/Leo/Sofia even slightly rescaled.
    k = "baked_context_embedding.weight"
    n = args.keep_base_speakers
    if n and k in sd and Path(args.base_weights).exists():
        base = torch.load(args.base_weights, map_location="cpu", weights_only=False)
        base = base.get("state_dict", base)
        if k in base:
            sd[k][:n] = base[k][:n].to(sd[k].dtype)
            print(f"restored baked speakers 0..{n - 1} from {args.base_weights}")

    torch.save({"state_dict": sd}, args.out)
    print(f"merged {merged} LoRA adapters -> {args.out}")


if __name__ == "__main__":
    main()
