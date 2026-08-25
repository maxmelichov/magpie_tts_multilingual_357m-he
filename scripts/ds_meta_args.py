#!/usr/bin/env python3
"""Emit hydra train_ds_meta/val_ds_meta overrides from data/manifests/datasets.json."""
import json
import sys
from pathlib import Path

index = Path(sys.argv[1] if len(sys.argv) > 1 else "data/manifests/datasets.json")
# Second arg is either one tokenizer for every dataset, or "per-language" to use
# "<name>_ipa_chartokenizer" (run 5 trains four languages at once, each with its
# own IPA tokenizer).
tokenizer = sys.argv[2] if len(sys.argv) > 2 else "hebrew_chartokenizer"

args = []
for d in json.loads(index.read_text()):
    name, audio_dir = d["name"], d["audio_dir"]
    for split, meta in (("train", "train_ds_meta"), ("val", "val_ds_meta")):
        key = f"{meta}.{name}" if split == "train" else f"{meta}.{name}_val"
        args += [
            f"+{key}.manifest_path={d[f'{split}_manifest']}",
            f"+{key}.audio_dir={audio_dir}",
            f"+{key}.feature_dir={audio_dir}",
            f"+{key}.sample_weight=1.0",
            f"+{key}.tokenizer_names=[{name + '_ipa_chartokenizer' if tokenizer == 'per-language' else tokenizer}]",
        ]
print("\n".join(args))
