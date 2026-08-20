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

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
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
    torch.save({"state_dict": sd}, args.out)
    print(f"merged {merged} LoRA adapters -> {args.out}")


if __name__ == "__main__":
    main()
