#!/usr/bin/env python3
"""Check whether the voice-cloning context clip actually influences output.

A speaker-similarity number is meaningless without knowing the scale of the
embedding space, so this reports the two anchors alongside it:

    real speaker vs itself        ~0.7-0.8   "same speaker" looks like this
    real speaker vs other speaker ~0.0       "different speaker" looks like this

Then the question that matters: if you synthesize the same text twice with two
*different* reference voices, do the outputs differ? If cross-reference
similarity equals self-similarity, the model is ignoring the reference and any
"speaker similarity" score is measuring nothing.

Usage:
  venv/bin/python scripts/diagnose_speaker_conditioning.py \
      --gen-a outputs/ilspeech_eval/<run>/audio/repeat_0 \
      --gen-b outputs/ilspeech_seenvoice/<run>/audio/repeat_0 \
      --real-a data/ilspeech/ilspeech-v2/wav/speaker1_*.wav \
      --real-b /home/maxm/AE_training_data_all/generated_audio/voice1_high_quality
"""

import argparse
import glob
import logging
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def load_model(device):
    logging.disable(logging.INFO)
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    return EncDecSpeakerLabelModel.from_pretrained("titanet_large", map_location=device).eval()


def embed(model, paths, device, limit=None):
    import librosa
    import soundfile as sf
    import torch

    out = []
    for p in sorted(paths)[:limit]:
        a, sr = sf.read(p, dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if sr != 16000:
            a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        t = torch.tensor(a, device=device).unsqueeze(0)
        with torch.no_grad():
            _, e = model.forward(input_signal=t,
                                 input_signal_length=torch.tensor([t.shape[1]], device=device))
        out.append(torch.nn.functional.normalize(e, dim=-1).squeeze(0).cpu().numpy())
    return out


def mean_sim(A, B, exclude_diagonal=False):
    vals = [float(a @ b) for i, a in enumerate(A) for j, b in enumerate(B)
            if not (exclude_diagonal and i == j)]
    return float(np.mean(vals)) if vals else float("nan")


def expand(spec):
    """Resolve a dir or glob to wav paths.

    NeMo writes predicted_audio_*.wav next to target_audio_*.wav and
    context_audio_*.wav in the same directory. A bare *.wav glob therefore
    silently mixes real reference audio into the "generated" set and inflates
    every similarity score, so an output directory resolves to predictions only.
    """
    hits = glob.glob(spec)
    if len(hits) == 1 and Path(hits[0]).is_dir():
        d = Path(hits[0])
        hits = glob.glob(str(d / "predicted_audio_*.wav")) or glob.glob(str(d / "*.wav"))
    return [h for h in hits if h.endswith(".wav")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-a", required=True, help="generated wavs, reference voice A (dir or glob)")
    ap.add_argument("--gen-b", required=True, help="generated wavs, reference voice B")
    ap.add_argument("--real-a", required=True, help="real recordings of voice A")
    ap.add_argument("--real-b", required=True, help="real recordings of voice B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    model = load_model(args.device)
    sets = {}
    for name in ["gen_a", "gen_b", "real_a", "real_b"]:
        paths = expand(getattr(args, name))
        if not paths:
            raise SystemExit(f"no wavs matched for --{name.replace('_', '-')}")
        sets[name] = embed(model, paths, args.device, args.limit)
        print(f"{name}: {len(sets[name])} files")

    print("\n== embedding-space scale (real recordings) ==")
    print(f"  real A vs real A        {mean_sim(sets['real_a'], sets['real_a'], True):.3f}   same speaker")
    print(f"  real B vs real B        {mean_sim(sets['real_b'], sets['real_b'], True):.3f}   same speaker")
    print(f"  real A vs real B        {mean_sim(sets['real_a'], sets['real_b']):.3f}   different speakers")

    print("\n== did cloning work? ==")
    print(f"  gen A vs real A         {mean_sim(sets['gen_a'], sets['real_a']):.3f}")
    print(f"  gen B vs real B         {mean_sim(sets['gen_b'], sets['real_b']):.3f}")

    print("\n== does the reference clip change anything? ==")
    self_a = mean_sim(sets["gen_a"], sets["gen_a"], True)
    cross = mean_sim(sets["gen_a"], sets["gen_b"])
    print(f"  gen A vs gen A          {self_a:.3f}   (self-similarity)")
    print(f"  gen A vs gen B          {cross:.3f}   (different reference)")
    if abs(cross - self_a) < 0.05:
        print("\n  VERDICT: cross-reference similarity matches self-similarity —")
        print("  the model produces the same voice regardless of the reference clip.")
    else:
        print("\n  VERDICT: the reference clip measurably changes the output voice.")


if __name__ == "__main__":
    main()
