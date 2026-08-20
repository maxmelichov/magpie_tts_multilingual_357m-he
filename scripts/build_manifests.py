#!/usr/bin/env python3
"""Build NeMo MagpieTTS training manifests from the Hebrew IPA datasets.

Sources (under --data-root, default /home/maxm/AE_training_data_all):
  - generated_audio/voice1_high_quality_phonemes.csv   (filename,phonemes)
  - generated_audio/voice2_improved_phonemes.csv       (filename,original_phonemes,whisper_phonemes,wer_score,passed_filter)
  - generated_audio/voice3_improved_phonemes.csv       (same schema as voice2)
  - optionally slow_44K/{female1,male1}_hebrew_slow_filtered.csv  (--include-slow44k)

Output: JSONL manifests in NeMo MagpieTTS format, one train + one val per voice.
Each line:
  {"audio_filepath": "<relative to audio_dir>", "text": "<IPA phonemes>",
   "duration": <sec>, "context_audio_filepath": "<same-speaker ref wav>",
   "context_text": "<its IPA phonemes>"}

The context audio is a randomly chosen *other* utterance of the same voice with
duration in [--context-min-dur, --max-dur] (voice-cloning reference; docs
recommend >= 3s).

Durations are read from WAV headers (fast, no decode). Rows whose wav is
missing, unparsable, or outside [--min-dur, --max-dur] are dropped, as are
rows failing the whisper filter (passed_filter != True or wer_score > --max-wer).
"""

import argparse
import csv
import json
import random
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASETS = {
    # name: (audio_dir relative to data root, csv relative to data root, text column)
    "voice1": ("generated_audio/voice1_high_quality", "generated_audio/voice1_high_quality_phonemes.csv", "phonemes"),
    "voice2": ("generated_audio/voice2_high_quality", "generated_audio/voice2_improved_phonemes.csv", "original_phonemes"),
    "voice3": ("generated_audio/voice3_high_quality", "generated_audio/voice3_improved_phonemes.csv", "original_phonemes"),
}

# All 12 slow_44K speakers are Hebrew IPA with the same filtered-CSV schema.
SLOW44K_SPEAKERS = [
    "female1_hebrew", "female1", "female2", "female3", "female4", "female5",
    "male1_hebrew", "male1", "male2", "male3", "male4", "male5",
]
SLOW44K_DATASETS = {
    spk: (f"slow_44K/data/{spk}_slow", f"slow_44K/{spk}_slow_filtered.csv", "original_phonemes")
    for spk in SLOW44K_SPEAKERS
}


def wav_duration(path: Path):
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


# Symbols the Hebrew IPA tokenizer knows. Rows containing anything else are
# dropped: they are mojibake, stray orthography, or foreign-script leakage.
ALLOWED_CHARS = set("abdefhijklmnoprstuvzɡʁʃʔχˈ" + " ,.?!")


def load_rows(csv_path: Path, text_col: str, max_wer: float):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = (r.get(text_col) or "").strip()
            if not text:
                continue
            if not set(text) <= ALLOWED_CHARS:
                continue
            if "passed_filter" in r and r["passed_filter"].strip().lower() != "true":
                continue
            if "wer_score" in r:
                try:
                    if float(r["wer_score"]) > max_wer:
                        continue
                except ValueError:
                    continue
            rows.append((r["filename"].strip(), text))
    return rows


def build_dataset(name, audio_dir: Path, csv_path: Path, text_col: str, args, out_dir: Path):
    rows = load_rows(csv_path, text_col, args.max_wer)
    print(f"[{name}] {len(rows)} rows after CSV filters", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        durations = list(ex.map(lambda r: wav_duration(audio_dir / r[0]), rows))

    entries = []
    missing = dropped_dur = 0
    for (fname, text), dur in zip(rows, durations):
        if dur is None:
            missing += 1
            continue
        if not (args.min_dur <= dur <= args.max_dur):
            dropped_dur += 1
            continue
        entries.append({"audio_filepath": fname, "text": text, "duration": round(dur, 3)})
    print(f"[{name}] kept {len(entries)} (missing/unreadable wav: {missing}, out-of-range duration: {dropped_dur})", flush=True)
    if len(entries) < 2:
        print(f"[{name}] not enough usable entries, skipping", file=sys.stderr)
        return None

    # Same-speaker context (voice-cloning reference): random other utterance, long enough.
    rng = random.Random(args.seed)
    ctx_pool = [e for e in entries if e["duration"] >= args.context_min_dur] or entries
    for e in entries:
        ctx = rng.choice(ctx_pool)
        while ctx is e and len(ctx_pool) > 1:
            ctx = rng.choice(ctx_pool)
        e["context_audio_filepath"] = ctx["audio_filepath"]
        e["context_text"] = ctx["text"]
        e["context_audio_duration"] = ctx["duration"]

    rng.shuffle(entries)
    n_val = min(args.val_size, max(1, len(entries) // 20))
    splits = {"val": entries[:n_val], "train": entries[n_val:]}
    paths = {}
    for split, items in splits.items():
        p = out_dir / f"{name}_{split}.json"
        with open(p, "w", encoding="utf-8") as f:
            for e in items:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        paths[split] = p
        print(f"[{name}] wrote {len(items):>6} -> {p}", flush=True)
    return {"name": name, "audio_dir": str(audio_dir), **{f"{s}_manifest": str(p) for s, p in paths.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("/home/maxm/AE_training_data_all"))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "manifests")
    ap.add_argument("--max-wer", type=float, default=0.2, help="max whisper WER for voice2/voice3 rows")
    ap.add_argument("--min-dur", type=float, default=0.5)
    ap.add_argument("--max-dur", type=float, default=20.0, help="matches MagpieTTS dataset max_duration")
    ap.add_argument("--context-min-dur", type=float, default=5.0, help="min duration of the voice-cloning context wav")
    ap.add_argument("--val-size", type=int, default=100, help="val utterances per voice (capped at 5%% of data)")
    ap.add_argument("--include-slow44k", action=argparse.BooleanOptionalAction, default=True,
                    help="also build manifests for the 12 slow_44K Hebrew speakers")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = dict(DATASETS)
    if args.include_slow44k:
        datasets.update(SLOW44K_DATASETS)

    results = []
    for name, (audio_rel, csv_rel, text_col) in datasets.items():
        audio_dir = args.data_root / audio_rel
        csv_path = args.data_root / csv_rel
        if not csv_path.exists() or not audio_dir.is_dir():
            print(f"[{name}] SKIP: missing {csv_path if not csv_path.exists() else audio_dir}", file=sys.stderr)
            continue
        r = build_dataset(name, audio_dir, csv_path, text_col, args, args.out_dir)
        if r:
            results.append(r)

    index = args.out_dir / "datasets.json"
    with open(index, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote dataset index -> {index}")


if __name__ == "__main__":
    main()
