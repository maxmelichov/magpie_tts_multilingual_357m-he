#!/usr/bin/env python3
"""Build a NeMo eval manifest from the ILSpeech v2 held-out test set.

ILSpeech (https://huggingface.co/datasets/Phonikud/ILSpeech) is real recorded
Hebrew from two speakers, neither of which appears in our training data, and it
ships gold IPA. That makes it a zero-shot voice-cloning benchmark: the model has
to clone an unseen voice and read IPA it was never trained on.

Two dataset facts need handling:

  * No test wav reaches the 10 s context length the model requires (frame
    stacking factor 2). We therefore build ONE context clip per speaker by
    concatenating utterances from the *train* split until >= 10 s, so nothing
    from the test set leaks into the voice reference.
  * A couple of rows carry IPA symbols outside the model's 27-symbol Hebrew
    vocabulary (`w` in loanwords). These are mapped to their nearest in-vocab
    Hebrew realization and counted, rather than silently dropped.

Writes: <out-dir>/eval_manifest.json, <out-dir>/context/<speaker>_context.wav
"""

import argparse
import json
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The model's Hebrew IPA vocabulary (scripts/magpietts_lora.py HEBREW_IPA_CHARS).
ALLOWED = set("abdefhijklmnoprstuvzɡʁʃʔχˈ" + " ,.?!")
# Nearest in-vocab Hebrew realization for symbols the tokenizer never saw.
SUBSTITUTIONS = {"w": "v", "g": "ɡ", "x": "χ", "ʒ": "ʃ"}
CONTEXT_MIN_SEC = 10.0
CONTEXT_GAP_SEC = 0.15


def read_metadata(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        uid, ipa, text = line.split("|", 2)
        rows.append({"id": uid, "ipa": ipa, "text": text, "speaker": uid.rsplit("_", 1)[0]})
    return rows


def normalize_ipa(ipa: str):
    """Map out-of-vocab symbols; return (clean_ipa, n_substitutions)."""
    out, n = [], 0
    for ch in ipa:
        if ch in ALLOWED:
            out.append(ch)
        elif ch in SUBSTITUTIONS:
            out.append(SUBSTITUTIONS[ch])
            n += 1
        else:
            n += 1  # dropped
    return "".join(out), n


def wav_duration(path: Path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def build_context(speaker, train_rows, wav_dir: Path, out_path: Path):
    """Concatenate train utterances of one speaker into a >=10 s voice reference."""
    picked, total = [], 0.0
    for r in train_rows:
        p = wav_dir / f"{r['id']}.wav"
        if not p.exists():
            continue
        picked.append((p, r))
        total += wav_duration(p) + CONTEXT_GAP_SEC
        if total >= CONTEXT_MIN_SEC:
            break
    if total < CONTEXT_MIN_SEC:
        raise RuntimeError(f"{speaker}: only {total:.1f}s of context available")

    with wave.open(str(picked[0][0]), "rb") as w:
        params = w.getparams()
    gap = b"\x00" * int(CONTEXT_GAP_SEC * params.framerate) * params.sampwidth * params.nchannels

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        for i, (p, _) in enumerate(picked):
            if i:
                out.writeframes(gap)
            with wave.open(str(p), "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))

    ipa = " ".join(normalize_ipa(r["ipa"])[0] for _, r in picked)
    return {"path": out_path, "ipa": ipa, "duration": wav_duration(out_path),
            "n_utts": len(picked), "source_ids": [r["id"] for _, r in picked]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ilspeech-dir", type=Path,
                    default=REPO_ROOT / "data/ilspeech/ilspeech-v2")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/ilspeech/eval")
    ap.add_argument("--limit", type=int, help="only the first N test utterances (smoke test)")
    args = ap.parse_args()

    wav_dir = args.ilspeech_dir / "wav"
    test = read_metadata(args.ilspeech_dir / "metadata_test.csv")
    train = read_metadata(args.ilspeech_dir / "metadata_train.csv")
    if args.limit:
        test = test[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    contexts = {}
    for spk in sorted({r["speaker"] for r in test}):
        rows = [r for r in train if r["speaker"] == spk]
        ctx = build_context(spk, rows, wav_dir, args.out_dir / "context" / f"{spk}_context.wav")
        contexts[spk] = ctx
        print(f"[{spk}] context {ctx['duration']:.1f}s from {ctx['n_utts']} train utts "
              f"({', '.join(ctx['source_ids'])})")

    manifest_path = args.out_dir / "eval_manifest.json"
    n_subs = n_rows_subbed = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in test:
            gt = wav_dir / f"{r['id']}.wav"
            if not gt.exists():
                print(f"  missing wav for {r['id']}, skipped")
                continue
            ipa, subs = normalize_ipa(r["ipa"])
            n_subs += subs
            n_rows_subbed += bool(subs)
            ctx = contexts[r["speaker"]]
            f.write(json.dumps({
                "audio_filepath": str(gt),             # ground truth, for reference metrics
                "text": ipa,
                "duration": round(wav_duration(gt), 3),
                "context_audio_filepath": str(ctx["path"]),
                "context_text": ctx["ipa"],
                "context_audio_duration": round(ctx["duration"], 3),
                # carried through for scoring, ignored by NeMo
                "utt_id": r["id"],
                "speaker": r["speaker"],
                "ref_text": r["text"],
                "ref_ipa": ipa,
            }, ensure_ascii=False) + "\n")

    print(f"\n{len(test)} utterances -> {manifest_path}")
    print(f"out-of-vocab IPA symbols substituted: {n_subs} in {n_rows_subbed} utterances")


if __name__ == "__main__":
    main()
