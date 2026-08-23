#!/usr/bin/env python3
"""Re-phonemize the Hebrew corpus with RenikudPlus.

The stored `original_phonemes` predate the current phonemizer and effectively
never contain `w`, `ʒ`, `tʃ` or `dʒ` (7 `w` and 1 `ʒ` across 1.3M utterances), so
the model cannot learn the loanword phonemes that `ג׳ ז׳ צ׳` and foreign `w`
require. RenikudPlus (notmax123/RenikudPlus) produces them from the Hebrew text.

The `hebrew_text` lives in slow_44K/metadata_<spk>.csv while the quality columns
live in slow_44K/<spk>_slow_filtered.csv; they join on `filename`.

Re-phonemizing invalidates the stored `wer_score`, which was computed against the
old phonemes. This recomputes it as the word-level distance between the NEW IPA
and the existing `whisper_phonemes` ASR readback, so a `--max-wer 0.0` filter
still means "the ASR heard exactly these phonemes" rather than silently passing
unverified labels through.

Output: slow_44K/<spk>_slow_renikud.csv, same schema as the *_filtered.csv files.
"""

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RENIKUD_DIR = REPO_ROOT / "tools/renikud"
DATA = Path("/home/maxm/AE_training_data_all/slow_44K")

_g2p = None
# The stored hebrew_text is already vocalized, but RenikudPlus predicts the
# diacritics itself and mangles pre-pointed input -- nikud leaks straight into
# the IPA. Strip it and the output matches the reference exactly.
NIKUD = re.compile(r"[\u0591-\u05c7]")


def _init():
    """One ONNX session per worker process."""
    global _g2p
    sys.path.insert(0, str(RENIKUD_DIR))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from renikud_onnx import G2P
    _g2p = G2P(str(RENIKUD_DIR / "renikud_cons5_point8_int8.onnx"))


def _edit(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _work(item):
    fname, hebrew, whisper = item
    try:
        ipa = _g2p.phonemize(NIKUD.sub('', hebrew)).strip()
    except Exception:
        return None
    ref, hyp = ipa.split(), whisper.split()
    wer = _edit(ref, hyp) / max(len(ref), 1)
    return fname, ipa, whisper, round(wer, 6)


def speaker_rows(spk):
    meta_p, filt_p = DATA / f"metadata_{spk}.csv", DATA / f"{spk}_slow_filtered.csv"
    if not (meta_p.exists() and filt_p.exists()):
        return None
    meta = {r["filename"]: (r.get("hebrew_text") or "").strip()
            for r in csv.DictReader(open(meta_p, newline="", encoding="utf-8"))}
    out = []
    for r in csv.DictReader(open(filt_p, newline="", encoding="utf-8")):
        h = meta.get(r["filename"], "")
        if h:
            out.append((r["filename"], h, (r.get("whisper_phonemes") or "").strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speakers", nargs="*", help="default: every speaker with a metadata_<spk>.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N rows per speaker (smoke test)")
    args = ap.parse_args()

    speakers = args.speakers or sorted(p.stem.replace("metadata_", "") for p in DATA.glob("metadata_*.csv"))
    print(f"speakers: {', '.join(speakers)}\nworkers: {args.workers}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for spk in speakers:
            rows = speaker_rows(spk)
            if not rows:
                print(f"[{spk}] no joinable rows, skipping", flush=True)
                continue
            if args.limit:
                rows = rows[: args.limit]
            out_p = DATA / f"{spk}_slow_renikud.csv"
            done = failed = 0
            with open(out_p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["filename", "original_phonemes", "whisper_phonemes", "wer_score", "passed_filter"])
                for res in ex.map(_work, rows, chunksize=64):
                    if res is None:
                        failed += 1
                        continue
                    fname, ipa, whisper, wer = res
                    w.writerow([fname, ipa, whisper, wer, "True"])
                    done += 1
                    if done % 20000 == 0:
                        print(f"[{spk}] {done:,}/{len(rows):,}", flush=True)
            print(f"[{spk}] wrote {done:,} rows ({failed} failed) -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
