#!/usr/bin/env python3
"""Run 5 head-to-head: base model (orthographic + its own g2p) vs IPA LoRA.

Both systems synthesize the SAME held-out sentences; only the input
representation differs. Scored with whisper-large-v3 against the orthographic
reference, plus the real recording as the measurement floor.
"""
import argparse, glob, json, re, sys, unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_ilspeech import edit_distance

import soundfile as sf, torch, librosa


def norm(t):
    t = unicodedata.normalize("NFC", t).lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return " ".join(t.split())


def transcribe(model_name, wavs, lang, device, bs=8):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name)
    m = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=torch.float16,
                                                  low_cpu_mem_usage=True).to(device).eval()
    out = []
    for i in range(0, len(wavs), bs):
        au = []
        for p in wavs[i:i + bs]:
            a, sr = sf.read(p, dtype="float32")
            if a.ndim > 1: a = a.mean(1)
            if sr != 16000: a = librosa.resample(a, orig_sr=sr, target_sr=16000)
            au.append(a)
        f = proc(au, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = m.generate(f.input_features.to(device, torch.float16),
                             language=lang, task="transcribe", max_new_tokens=200)
        out += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
    del m; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=["en", "de", "it", "es"])
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/run5_scores.json")
    args = ap.parse_args()

    results = {}
    for lang in args.langs:
        rows = [json.loads(l) for l in open(f"data/manifests_multiling/eval/{lang}_ipa.json", encoding="utf-8")]
        refs = [r["ref_text"] for r in rows]
        gts = [r["audio_filepath"] for r in rows]
        res = {}
        for tag in ["ipa", "base"]:
            d = glob.glob(f"outputs/r5_{lang}_{tag}/*/audio/repeat_0")[0]
            wavs = [f"{d}/predicted_audio_{i}.wav" for i in range(len(rows))]
            wavs = [w for w in wavs if Path(w).exists()]
            hyp = transcribe(args.asr, wavs, lang, args.device)
            E = N = 0
            for r, h in zip(refs, hyp):
                a, b = norm(r).split(), norm(h).split()
                E += edit_distance(a, b); N += len(a)
            res[tag] = E / max(N, 1)
            print(f"[{lang}] {tag:5s} WER {res[tag]:.2%}", flush=True)
        hyp = transcribe(args.asr, gts, lang, args.device)
        E = N = 0
        for r, h in zip(refs, hyp):
            a, b = norm(r).split(), norm(h).split()
            E += edit_distance(a, b); N += len(a)
        res["real"] = E / max(N, 1)
        print(f"[{lang}] real  WER {res['real']:.2%}", flush=True)
        results[lang] = res

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + "=" * 56)
    print(f"{'lang':<6}{'base(g2p)':>12}{'IPA LoRA':>11}{'real rec':>11}{'delta':>10}")
    for l, r in results.items():
        print(f"{l:<6}{r['base']:>11.1%}{r['ipa']:>11.1%}{r['real']:>11.1%}{r['ipa']-r['base']:>+10.1%}")


if __name__ == "__main__":
    main()
