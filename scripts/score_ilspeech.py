#!/usr/bin/env python3
"""Score MagpieTTS output on the ILSpeech held-out test set.

Runs two ASR passes over both the synthesized audio and the ground-truth
recordings, so every number comes with the recording-quality ceiling next to it:

  Hebrew WER/CER  ivrit-ai/whisper-large-v3-turbo, orthographic text
  IPA PER         notmax123/whisper-he-ipa, compared against the gold IPA
                  (no G2P round-trip, so it scores what the model was actually
                  asked to pronounce)
  speaker sim     cosine similarity of TitaNet embeddings, pred vs ground truth
                  and pred vs the context clip the voice was cloned from
  duration ratio  generated length / ground-truth length

The ground-truth row is the topline: it is the same metric on real recordings,
so it separates synthesis errors from ASR errors.

Usage:
  venv/bin/python scripts/score_ilspeech.py \
      --pred-dir outputs/ilspeech_eval/<run>/audio/repeat_0 \
      --manifest data/ilspeech/eval/eval_manifest.json
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
HEBREW_ASR = "ivrit-ai/whisper-large-v3-turbo"
IPA_ASR = "notmax123/whisper-he-ipa"
NIQQUD = re.compile(r"[֑-ׇ]")
PUNCT = re.compile(r"[^\w\s֐-׿]", re.UNICODE)


def edit_distance(a, b):
    """Levenshtein distance over two sequences (rolling row)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_heb(text):
    """Strip niqqud and punctuation; ASR output has neither consistently."""
    text = unicodedata.normalize("NFC", text)
    text = NIQQUD.sub("", text)
    text = PUNCT.sub(" ", text)
    return " ".join(text.split())


# notmax123/whisper-he-ipa emits an ASCII transliteration, not literal IPA.
# Without this map every ʔ/ʃ/χ/ɡ/ʁ counts as a substitution and the PER floor
# sits near 37% even on real recordings.
ASR_IPA_MAP = str.maketrans({"q": "ʔ", "S": "ʃ", "x": "χ", "g": "ɡ", "r": "ʁ",
                             "Z": "ʒ", "'": "ˈ", "R": "ʁ", "X": "χ"})


def norm_asr_ipa(text):
    return norm_ipa(text.translate(ASR_IPA_MAP))


def norm_ipa(text):
    """Strip punctuation but KEEP stress.

    Stress was excluded originally on the assumption that ASR could not recover
    it. Measured on 15,271 words of real recordings it is recovered 99.5% of the
    time, so dropping it just discarded a real quality signal: on the ILSpeech
    full set the stress-blind IPA WER cannot separate two models (delta -0.21%,
    CI [-0.57, +0.18]) while the stress-inclusive one can (-0.65%, CI
    [-1.02, -0.27]). Stress placement is meaningful in Hebrew; score it."""
    return " ".join(re.sub(r"[,.?!]", "", text).split())


def rate(ref_units, hyp_units):
    """(errors, ref_length) for corpus-level accumulation."""
    return edit_distance(ref_units, hyp_units), len(ref_units)


def transcribe_all(model_name, wavs, device, batch_size=8, language="he"):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()

    out = []
    for i in range(0, len(wavs), batch_size):
        chunk = wavs[i: i + batch_size]
        audio = []
        for p in chunk:
            a, sr = sf.read(p, dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            if sr != 16000:
                import librosa
                a = librosa.resample(a, orig_sr=sr, target_sr=16000)
            audio.append(a)
        feats = proc(audio, sampling_rate=16000, return_tensors="pt",
                     return_attention_mask=True)
        with torch.no_grad():
            ids = model.generate(
                feats.input_features.to(device, torch.float16),
                attention_mask=getattr(feats, "attention_mask", None).to(device)
                if getattr(feats, "attention_mask", None) is not None else None,
                language=language, task="transcribe", max_new_tokens=200,
            )
        out.extend(proc.batch_decode(ids, skip_special_tokens=True))
        print(f"  {model_name}: {min(i + batch_size, len(wavs))}/{len(wavs)}", flush=True)

    del model
    torch.cuda.empty_cache()
    return [t.strip() for t in out]


def speaker_embeddings(wavs, device):
    """TitaNet embeddings; returns None if the model can't be fetched."""
    try:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        sv = EncDecSpeakerLabelModel.from_pretrained("titanet_large", map_location=device).eval()
    except Exception as e:
        print(f"  speaker model unavailable ({type(e).__name__}: {e}); skipping similarity")
        return None
    embs = []
    with torch.no_grad():
        for p in wavs:
            a, sr = sf.read(p, dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            if sr != 16000:
                import librosa
                a = librosa.resample(a, orig_sr=sr, target_sr=16000)
            t = torch.tensor(a, device=device).unsqueeze(0)
            _, emb = sv.forward(input_signal=t, input_signal_length=torch.tensor([t.shape[1]], device=device))
            embs.append(torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu())
    del sv
    torch.cuda.empty_cache()
    return embs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True, help="dir with predicted_audio_<i>.wav")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/ilspeech/eval/eval_manifest.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/ilspeech_scores.json")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-speaker-sim", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    preds, kept = [], []
    for i, r in enumerate(rows):
        p = args.pred_dir / f"predicted_audio_{i}.wav"
        if p.exists():
            preds.append(p)
            kept.append(r)
        else:
            print(f"  missing prediction for row {i} ({r['utt_id']})")
    rows = kept
    gts = [Path(r["audio_filepath"]) for r in rows]
    print(f"scoring {len(rows)} utterances")

    print("\nHebrew ASR (predicted)...")
    heb_pred = transcribe_all(HEBREW_ASR, preds, args.device, args.batch_size)
    print("Hebrew ASR (ground truth)...")
    heb_gt = transcribe_all(HEBREW_ASR, gts, args.device, args.batch_size)
    print("IPA ASR (predicted)...")
    ipa_pred = transcribe_all(IPA_ASR, preds, args.device, args.batch_size)
    print("IPA ASR (ground truth)...")
    ipa_gt = transcribe_all(IPA_ASR, gts, args.device, args.batch_size)

    sims_gt = sims_ctx = sims_ctrl = None
    if not args.no_speaker_sim:
        print("\nSpeaker embeddings...")
        ctxs = [Path(r["context_audio_filepath"]) for r in rows]
        e_pred = speaker_embeddings(preds, args.device)
        if e_pred is not None:
            e_gt = speaker_embeddings(gts, args.device)
            e_ctx = speaker_embeddings(sorted(set(ctxs)), args.device)
            ctx_map = {p: e for p, e in zip(sorted(set(ctxs)), e_ctx)}
            sims_gt = [float(a @ b) for a, b in zip(e_pred, e_gt)]
            sims_ctx = [float(a @ ctx_map[c]) for a, c in zip(e_pred, ctxs)]
            # Control: ground-truth recording vs its own context clip. Same real
            # speaker, so this is the ceiling the synthesized voice is chasing.
            sims_ctrl = [float(a @ ctx_map[c]) for a, c in zip(e_gt, ctxs)]

    METRICS = [
        # key,            reference,        hypothesis list, unit
        ("heb_wer_pred",  "text", heb_pred, "word"),
        ("heb_wer_gt",    "text", heb_gt,   "word"),
        ("heb_cer_pred",  "text", heb_pred, "char"),
        ("heb_cer_gt",    "text", heb_gt,   "char"),
        ("ipa_per_pred",  "ipa",  ipa_pred, "char"),
        ("ipa_per_gt",    "ipa",  ipa_gt,   "char"),
        # Word-level over IPA: one wrong phoneme fails the whole word, so this is
        # far stricter than PER and shows errors that char-level distance averages away.
        ("ipa_wer_pred",  "ipa",  ipa_pred, "word"),
        ("ipa_wer_gt",    "ipa",  ipa_gt,   "word"),
        # baseline: how much of the PER is the ASR's own transliteration noise

    ]

    per_utt = []
    for i, r in enumerate(rows):
        refs = {"text": norm_heb(r["ref_text"]), "ipa": norm_ipa(r["ref_ipa"])}
        norms = {"text": norm_heb, "ipa": norm_asr_ipa}
        u = {"utt_id": r["utt_id"], "speaker": r["speaker"],
             "ref_text": r["ref_text"], "ref_ipa": r["ref_ipa"],
             "heb_asr_pred": heb_pred[i], "heb_asr_gt": heb_gt[i],
             "ipa_asr_pred": ipa_pred[i], "ipa_asr_gt": ipa_gt[i], "_err": {}}
        for key, ref_kind, hyps, unit in METRICS:
            ref, hyp = refs[ref_kind], norms[ref_kind](hyps[i])
            if unit == "word":
                rr, hh = ref.split(), hyp.split()
            else:
                rr, hh = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
            e, n = rate(rr, hh)
            u["_err"][key] = (e, n)
            u[key] = e / max(n, 1)
        u["dur_pred"] = sf.info(preds[i]).duration
        u["dur_gt"] = sf.info(gts[i]).duration
        u["dur_ratio"] = u["dur_pred"] / u["dur_gt"]
        if sims_gt is not None:
            u["spk_sim_gt"] = sims_gt[i]
            u["spk_sim_context"] = sims_ctx[i]
            u["spk_sim_gt_vs_context"] = sims_ctrl[i]
        per_utt.append(u)

    def agg(sub):
        """Corpus-level rates (total errors / total reference length) plus
        utterance-mean values for the non-rate metrics."""
        out = {"n": len(sub)}
        for key, *_ in METRICS:
            e = sum(u["_err"][key][0] for u in sub)
            n = sum(u["_err"][key][1] for u in sub)
            out[key] = e / max(n, 1)
        for key in ["dur_ratio", "spk_sim_gt", "spk_sim_context", "spk_sim_gt_vs_context"]:
            vals = [u[key] for u in sub if key in u]
            if vals:
                out[key] = float(np.mean(vals))
        return out

    summary = {"corpus": agg(per_utt),
               "by_speaker": {s: agg([u for u in per_utt if u["speaker"] == s])
                              for s in sorted({u["speaker"] for u in per_utt})}}
    for u in per_utt:
        del u["_err"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_utterance": per_utt},
                                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"ILSpeech held-out test set — {len(rows)} utterances")
    print("=" * 72)
    print(f"{'metric':<22}{'synthesized':>14}{'ground truth':>14}   (lower is better)")
    c = summary["corpus"]
    print(f"{'Hebrew WER':<22}{c['heb_wer_pred']:>13.1%}{c['heb_wer_gt']:>14.1%}")
    print(f"{'Hebrew CER':<22}{c['heb_cer_pred']:>13.1%}{c['heb_cer_gt']:>14.1%}")
    print(f"{'IPA PER':<22}{c['ipa_per_pred']:>13.1%}{c['ipa_per_gt']:>14.1%}")
    print(f"{'IPA WER (word-level)':<22}{c['ipa_wer_pred']:>13.1%}{c['ipa_wer_gt']:>14.1%}")
    m = summary["corpus"]
    if "spk_sim_gt" in m:
        print(f"\n{'speaker sim vs GT':<22}{m['spk_sim_gt']:>13.3f}   (higher is better)")
        print(f"{'speaker sim vs context':<22}{m['spk_sim_context']:>13.3f}"
              f"{m['spk_sim_gt_vs_context']:>14.3f}   <- real speaker ceiling")
    print(f"{'duration ratio':<22}{m['dur_ratio']:>13.3f}   (1.0 = matches GT pace)")
    print("\nby speaker:")
    for s, v in summary["by_speaker"].items():
        line = f"  {s:<12} n={v['n']:<4} WER {v['heb_wer_pred']:.1%} (gt {v['heb_wer_gt']:.1%})  PER {v['ipa_per_pred']:.1%} (gt {v['ipa_per_gt']:.1%})  ipaWER {v['ipa_wer_pred']:.1%} (gt {v['ipa_wer_gt']:.1%})"
        if "spk_sim_gt" in v:
            line += f"  sim {v['spk_sim_gt']:.3f}"
        print(line)
    print(f"\nfull results -> {args.out}")


if __name__ == "__main__":
    main()
