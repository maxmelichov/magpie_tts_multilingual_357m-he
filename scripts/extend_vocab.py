#!/usr/bin/env python3
"""Add IPA symbols to a trained Hebrew checkpoint without retraining.

Phonikud emits phonemes our 27-symbol Hebrew vocabulary does not cover -- `w`
(SPECIAL_PHONEMES) and the geresh set `ג׳ ז׳ צ׳` -> `dʒ ʒ tʃ`. Of those only `w`
and `ʒ` are new *characters*: the tokenizer is character-level, so the affricates
`tʃ`/`dʒ` are already expressible as two tokens, exactly like `ts` (צ) is today.

Retraining cannot teach these anyway -- the Hebrew corpus contains 7 instances of
`w` and 1 of `ʒ` in 1.3M utterances. But the base model was pretrained on six IPA
languages that all use them, so the embeddings already exist. This script grows
the Hebrew block, carries every trained row across by symbol, and seeds the new
rows from those donor languages.

Usage:
  venv/bin/python scripts/extend_vocab.py \
      --ckpt experiments/magpie_hebrew_run3.ckpt \
      --hparams experiments/magpie_hebrew_run3.hparams.yaml \
      --add "wʒ" --out-ckpt extended.ckpt --out-hparams extended.hparams.yaml
"""

import argparse
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
IPA_DONORS = ["spanish_phoneme", "portuguese_Brazilian_phoneme", "german_phoneme",
              "hindi_phoneme", "english_phoneme"]


def hebrew_tokens(chars, tok_cfg):
    cfg = OmegaConf.to_container(tok_cfg, resolve=True)
    cfg["chars"] = chars
    return [str(t) for t in instantiate(OmegaConf.create(cfg)).tokens]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hparams", required=True)
    ap.add_argument("--add", required=True, help="characters to append, e.g. 'wʒ'")
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--out-hparams", required=True)
    ap.add_argument("--hebrew-name", default="hebrew_chartokenizer")
    args = ap.parse_args()

    hp = OmegaConf.load(args.hparams)
    tcfgs = hp.cfg.text_tokenizers if "cfg" in hp else hp.text_tokenizers
    heb_cfg = tcfgs[args.hebrew_name]
    old_chars = str(heb_cfg.chars)
    new_chars = old_chars + "".join(c for c in args.add if c not in old_chars)
    if new_chars == old_chars:
        raise SystemExit("nothing to add -- all characters already present")
    print(f"chars: {len(old_chars)} -> {len(new_chars)}  (added {new_chars[len(old_chars):]!r})")

    old_heb = hebrew_tokens(old_chars, heb_cfg)
    new_heb = hebrew_tokens(new_chars, heb_cfg)
    old_idx = {t: i for i, t in enumerate(old_heb)}
    print(f"hebrew tokens: {len(old_heb)} -> {len(new_heb)}")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    emb = sd["text_embedding.weight"]
    n_old, dim = emb.shape
    heb_off = n_old - 2 - len(old_heb)          # [base | hebrew | BOS | EOS]
    n_new = heb_off + len(new_heb) + 2
    print(f"text_embedding: {n_old} -> {n_new} rows (hebrew block starts at {heb_off})")

    # Donor rows. Build the real aggregated tokenizer rather than accumulating
    # offsets by hand: text_ce_tokenizer is an HF tokenizer that will not
    # instantiate standalone, and one unbuildable entry invalidates every offset
    # after it.
    from nemo.collections.tts.models.magpietts import setup_tokenizers
    agg = setup_tokenizers(all_tokenizers_config=tcfgs, mode='train',
                           cfg_nemo_version=hp.get('nemo_version', None))
    assert agg.tokenizer_offsets[args.hebrew_name] == heb_off, (
        f"hebrew offset mismatch: tokenizer says {agg.tokenizer_offsets[args.hebrew_name]}, "
        f"checkpoint layout implies {heb_off}")
    donors = {}
    for name in IPA_DONORS:
        if name not in agg.tokenizers:
            continue
        tk = agg.tokenizers[name]
        toks = [str(t) for t in (tk.tokens if hasattr(tk, "tokens") else tk.get_vocab().keys())]
        donors[name] = {t: agg.tokenizer_offsets[name] + i for i, t in enumerate(toks)}

    new = torch.empty(n_new, dim, dtype=emb.dtype)
    torch.nn.init.normal_(new, 0.0, float(emb[:heb_off].std()))
    new[:heb_off] = emb[:heb_off]               # every other language, untouched
    new[-2] = emb[-2]                           # BOS moves with the block
    new[-1] = emb[-1]                           # EOS

    carried, seeded, missing = 0, [], []
    for i, tok in enumerate(new_heb):
        if tok in old_idx:
            new[heb_off + i] = emb[heb_off + old_idx[tok]]
            carried += 1
            continue
        for dname, rows in donors.items():
            if tok in rows:
                new[heb_off + i] = emb[rows[tok]]
                seeded.append(f"{tok}<-{dname.split('_')[0]}")
                break
        else:
            missing.append(tok)
    print(f"carried over {carried} trained rows")
    print(f"seeded from donor languages: {' '.join(seeded) if seeded else '(none)'}")
    if missing:
        print(f"no donor, random init: {missing}")

    sd["text_embedding.weight"] = new
    torch.save({"state_dict": sd}, args.out_ckpt)

    heb_cfg.chars = new_chars
    OmegaConf.save(hp, args.out_hparams)
    print(f"\nwrote {args.out_ckpt}\n      {args.out_hparams}")


if __name__ == "__main__":
    main()
