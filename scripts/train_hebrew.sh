#!/usr/bin/env bash
# Fine-tune nvidia/magpie_tts_multilingual_357m on Hebrew (IPA input).
#
# Hebrew is a NEW language for this model, so we follow the NeMo
# "adding a new language" recipe: a byte-level byt5 tokenizer
# (`hebrew_chartokenizer`) consumes the IPA phoneme strings from the
# manifests directly.
#
# Datasets are read from data/manifests/datasets.json (built by
# scripts/build_manifests.py) — one train+val ds_meta entry per voice.
#
# Prereqs:  bash scripts/setup.sh  &&  venv/bin/python scripts/build_manifests.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_JSON="$REPO_ROOT/data/manifests/datasets.json"
CKPT="$REPO_ROOT/checkpoints/magpie_tts_multilingual_357m.nemo"

GPU="${GPU:-0,1}"                     # comma-separated GPU ids
NUM_DEVICES=$(awk -F, '{print NF}' <<< "$GPU")
BATCH_SIZE="${BATCH_SIZE:-8}"         # per device; 32 GB RTX 5090, fp32
LR="${LR:-1e-4}"                      # LoRA fine-tune LR
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-2000}"   # weighted sampling across voices

for f in "$CKPT" "$DATASETS_JSON"; do
  [ -e "$f" ] || { echo "missing $f — run setup.sh / build_manifests.py first" >&2; exit 1; }
done

mapfile -t DS_ARGS < <("$REPO_ROOT/venv/bin/python" "$REPO_ROOT/scripts/ds_meta_args.py" "$DATASETS_JSON")
echo "datasets: $(( ${#DS_ARGS[@]} / 10 ))"

cd "$REPO_ROOT/NeMo"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"  # for scripts.hebrew_ipa_tokenizer

exec "$REPO_ROOT/venv/bin/python" "$REPO_ROOT/scripts/magpietts_lora.py" \
    --config-path="$REPO_ROOT/NeMo/examples/tts/conf/magpietts" \
    --config-name=magpietts \
    +init_from_nemo_model="$CKPT" \
    +lora.r="$LORA_R" \
    +lora.alpha="$LORA_ALPHA" \
    exp_manager.exp_dir="$REPO_ROOT/experiments" \
    ++exp_manager.checkpoint_callback_params.save_top_k=3 \
    \
    \
    "${DS_ARGS[@]}" \
    \
    model.codecmodel_path=nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
    model.context_duration_min=10.0 \
    model.context_duration_max=10.0 \
    model.alignment_loss_scale=0.0 \
    model.prior_scaling_factor=null \
    model.optim.lr="$LR" \
    ~model.optim.sched \
    model.load_cached_codes_if_available=true \
    trainer.precision=32 \
    trainer.devices="$NUM_DEVICES" \
    trainer.num_nodes=1 \
    weighted_sampling_steps_per_epoch="$STEPS_PER_EPOCH" \
    batch_size="$BATCH_SIZE" \
    max_epochs="$MAX_EPOCHS" \
    "$@"
