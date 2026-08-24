#!/usr/bin/env python3
"""Make a merged checkpoint's hparams self-contained and portable.

Two problems with hparams.yaml as produced by training:
  1. train_ds/validation_ds embed ~90 absolute paths into this machine's data
     directory. Dead weight for inference -- strip them.
  2. Six tokenizers (spanish/german/hindi/portuguese/arabic phoneme dicts) and
     the speaker_map point at `nemo:<hash>_<name>` artifacts that were extracted
     to checkpoints/extracted/ on THIS machine and nowhere else. Without them,
     loading the model on another machine fails at tokenizer construction --
     before any of our code even runs.

This copies every referenced artifact into <out_dir>/assets/ and rewrites the
tokenizer/speaker_map configs to relative paths, so the release dir is complete
and machine-independent.
"""
import argparse
import shutil
from pathlib import Path
from omegaconf import OmegaConf, open_dict


def resolve_artifacts(node, extracted: Path, assets: Path):
    """Walk the config; copy any local artifact reference and rewrite in place.

    Two forms show up: `nemo:<hash>_<name>` (unresolved, e.g. speaker_map) and an
    already-resolved absolute path under checkpoints/extracted/ (e.g. the
    phoneme_dict/heteronyms fields our tokenizer roster builder resolved eagerly
    at train time). Both point at the same machine-local directory.
    """
    if isinstance(node, str) and node.startswith("nemo:"):
        src = extracted / node[len("nemo:"):]
    elif isinstance(node, str) and str(extracted) in node:
        src = Path(node)
    else:
        src = None
    if src is not None:
        if not src.exists():
            raise FileNotFoundError(f"artifact referenced but missing: {src}")
        dst = assets / src.name
        if not dst.exists():
            shutil.copy(src, dst)
        return f"assets/{src.name}"
    if hasattr(node, "items"):
        for k in list(node.keys()):
            node[k] = resolve_artifacts(node[k], extracted, assets)
    elif isinstance(node, list):
        for i in range(len(node)):
            node[i] = resolve_artifacts(node[i], extracted, assets)
    return node


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hparams", required=True)
    ap.add_argument("--extracted", default="checkpoints/extracted", help="dir the base .nemo was unpacked to")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    hp = OmegaConf.load(args.hparams)
    c = hp.cfg if "cfg" in hp else hp

    with open_dict(c):
        for k in ["train_ds", "validation_ds"]:
            c.pop(k, None)

    resolve_artifacts(c.text_tokenizers, Path(args.extracted), assets)
    with open_dict(c):
        c.speaker_map = resolve_artifacts(c.get("speaker_map"), Path(args.extracted), assets)

    out_path = out / "hparams.yaml"
    OmegaConf.save(hp, out_path)
    remaining = [l for l in out_path.read_text().splitlines() if "/home/" in l or "/mnt/" in l]
    print(f"wrote {out_path}")
    print(f"assets copied: {len(list(assets.iterdir()))}")
    if remaining:
        print(f"WARNING: {len(remaining)} absolute local paths remain:")
        for l in remaining[:10]:
            print(" ", l.strip()[:100])
    else:
        print("no absolute local paths remain")


if __name__ == "__main__":
    main()
