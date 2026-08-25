#!/usr/bin/env python3
"""Build IPA manifests for English/German/Italian/Spanish (run 5).

Goal of run 5: show that feeding IPA to a LoRA fine-tune beats the base model's
own g2p pipeline on its OWN languages -- the same trick that worked for Hebrew,
applied where the base model already has a tokenizer to compare against.

Source: /mnt/data/lightblue/.../combined_dataset_cleaned_real_data.csv
  filename,whisper_phonemes,speaker_id,wer_score,lang,phonemized
`whisper_phonemes` is already IPA (the `phonemized` column is a bool flag, not
text), and every row is already wer_score == 0, so no G2P and no WER filtering
is needed here.

Orthographic text -- needed as the WER reference and as the BASE model's input
for the comparison -- is not in that CSV. It is recovered from the LJSpeech-style
metadata.csv (`id|text|normalized`) sitting next to each wavs/ directory.
"""

import argparse
import csv
import json
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

COMBINED = Path("/mnt/data/lightblue/generated_audio/combined_dataset_cleaned_real_data.csv")
LANGS = ["en", "de", "it", "es"]

# Some English rows were double-phonemized: an IPA string was fed back through an
# English G2P, which pronounced each IPA character by its Unicode name. The result
# reads "stress I lengthened turned-V small-cap-I ..." instead of speech, and does
# not match its audio at all. 100% of the michael-he/en subset, 1.5% of LibriTTS,
# 0.6% of hifiTTS. Training on these would teach the model to say "stress".
CORRUPT_MARKERS = ("stɹˈɛs", "lˈɛŋθənd", "smˌɔːlkˌæp", "lˌɛɾɚtˈuː", "ʃwˈɑː", "ˌoʊpənˈ")


def looks_double_phonemized(ipa: str) -> bool:
    return any(m in ipa for m in CORRUPT_MARKERS)


@lru_cache(maxsize=8192)
def load_metadata(wavdir: str):
    """id -> orthographic text, from the metadata.csv beside a wavs/ dir.

    M-AILABS (de/it/es) uses LJSpeech format `id|text|normalized`. Note the
    English michael-he/en/metadata.csv holds IPA rather than orthography, so it
    is rejected here -- using it would make ref_text identical to the model
    input and the WER comparison meaningless.
    """
    meta = Path(wavdir).parent / "metadata.csv"
    out = {}
    if meta.exists():
        for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("|")
            if len(parts) >= 2:
                txt = (parts[-1] or parts[1]).strip()
                # IPA-only file: skip it, these are not orthographic references.
                if any(ch in txt for ch in "ˈˌɪʊɹɐəɜː"):
                    return {}
                out[parts[0].strip().replace(".wav", "")] = txt
    return out


@lru_cache(maxsize=1)
def load_hifitts():
    """hifiTTS_44k ships one metadata.csv for the whole flat directory."""
    p = Path("/home/maxm/AE_training_data_all/datasets_4AE_extracted/hifiTTS_44k/metadata.csv")
    out = {}
    if p.exists():
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                fn = (r.get("filename") or "").strip()
                if fn:
                    out[Path(fn).stem] = (r.get("text") or "").strip()
    return out


def orthographic_text(p: Path) -> str:
    """Resolve the orthographic transcript for one wav, per corpus layout."""
    # LibriTTS: one .normalized.txt sitting beside each wav
    nt = p.with_suffix("").with_suffix(".normalized.txt")
    if not nt.exists():
        nt = p.parent / (p.stem + ".normalized.txt")
    if nt.exists():
        return nt.read_text(encoding="utf-8", errors="replace").strip()
    if "hifiTTS_44k" in str(p):
        return load_hifitts().get(p.stem, "")
    return load_metadata(str(p.parent)).get(p.stem, "")


def wav_duration(p: Path):
    try:
        with wave.open(str(p), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("data/manifests_multiling"))
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--max-per-lang", type=int, default=60000)
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--min-dur", type=float, default=0.5)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = {l: [] for l in args.langs}
    with open(COMBINED, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            l = (r.get("lang") or "").strip()
            if l in rows and len(rows[l]) < args.max_per_lang:
                ipa = (r.get("whisper_phonemes") or "").strip()
                if ipa and not looks_double_phonemized(ipa):
                    rows[l].append((r["filename"], ipa, (r.get("speaker_id") or l).strip()))

    import random
    index = []
    for lang in args.langs:
        items = rows[lang]
        print(f"[{lang}] {len(items)} rows from CSV", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            durs = list(ex.map(lambda it: wav_duration(Path(it[0])), items))

        entries, no_text, bad = [], 0, 0
        for (fp, ipa, spk), d in zip(items, durs):
            if d is None or not (args.min_dur <= d <= args.max_dur):
                bad += 1
                continue
            p = Path(fp)
            text = orthographic_text(p)
            if not text:
                no_text += 1
            entries.append({"audio_filepath": str(p), "text": ipa, "duration": round(d, 3),
                            "ref_text": text, "speaker_id": spk, "lang": lang})
        print(f"[{lang}] kept {len(entries)} (bad/missing wav {bad}, no orthographic text {no_text})", flush=True)
        if len(entries) < 10:
            continue

        rng = random.Random(args.seed)
        # Voice-cloning context: another utterance from the same speaker.
        by_spk = {}
        for e in entries:
            by_spk.setdefault(e["speaker_id"], []).append(e)
        for e in entries:
            pool = by_spk[e["speaker_id"]]
            ctx = rng.choice(pool)
            for _ in range(5):
                if ctx is not e:
                    break
                ctx = rng.choice(pool)
            e["context_audio_filepath"] = ctx["audio_filepath"]
            e["context_audio_duration"] = ctx["duration"]

        rng.shuffle(entries)
        n_val = min(args.val_size, max(1, len(entries) // 20))
        for split, items2 in [("val", entries[:n_val]), ("train", entries[n_val:])]:
            p = args.out_dir / f"{lang}_{split}.json"
            with open(p, "w", encoding="utf-8") as f:
                for e in items2:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            print(f"[{lang}] wrote {len(items2):>6} -> {p}", flush=True)
        index.append({"name": lang, "audio_dir": "/",
                      "train_manifest": str(args.out_dir / f"{lang}_train.json"),
                      "val_manifest": str(args.out_dir / f"{lang}_val.json")})

    with open(args.out_dir / "datasets.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"\nWrote index -> {args.out_dir/'datasets.json'}")


if __name__ == "__main__":
    main()
