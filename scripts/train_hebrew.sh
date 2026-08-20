#!/usr/bin/env bash
# Fine-tune nvidia/magpie_tts_multilingual_357m on Hebrew (IPA input).
#
# Hebrew is a NEW language for this model, so we follow the NeMo
# "adding a new language" recipe: a byte-level byt5 tokenizer
# (`hebrew_chartokenizer`) consumes the IPA phoneme strings from the
# manifests directly.
#
# Prereqs:  bash scripts/setup.sh  &&  venv/bin/python scripts/build_manifests.py
#
# NOTE: defaults to GPU 1 (override with GPU=0,1 for both GPUs once cuda:0 is free).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFESTS="$REPO_ROOT/data/manifests"
DATA_ROOT="${DATA_ROOT:-/home/maxm/AE_training_data_all}"
CKPT="$REPO_ROOT/checkpoints/magpie_tts_multilingual_357m.nemo"

GPU="${GPU:-1}"                       # comma-separated GPU ids; cuda:0 is busy -> default 1
NUM_DEVICES=$(awk -F, '{print NF}' <<< "$GPU")
BATCH_SIZE="${BATCH_SIZE:-8}"         # 32 GB RTX 5090, fp32
LR="${LR:-1e-5}"                      # new-language fine-tune LR per NeMo docs
MAX_EPOCHS="${MAX_EPOCHS:-100}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"   # weighted sampling across the 3 voices

for f in "$CKPT" "$MANIFESTS/voice1_train.json"; do
  [ -f "$f" ] || { echo "missing $f — run setup.sh / build_manifests.py first" >&2; exit 1; }
done

cd "$REPO_ROOT/NeMo"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

exec "$REPO_ROOT/venv/bin/python" examples/tts/magpietts.py \
    --config-path=conf/magpietts \
    --config-name=magpietts \
    +init_from_nemo_model="$CKPT" \
    exp_manager.exp_dir="$REPO_ROOT/experiments" \
    +exp_manager.checkpoint_callback_params.save_top_k=3 \
    \
    +model.text_tokenizers.hebrew_chartokenizer._target_=AutoTokenizer \
    +model.text_tokenizers.hebrew_chartokenizer.pretrained_model="google/byt5-small" \
    \
    +train_ds_meta.voice1.manifest_path="$MANIFESTS/voice1_train.json" \
    +train_ds_meta.voice1.audio_dir="$DATA_ROOT/generated_audio/voice1_high_quality" \
    +train_ds_meta.voice1.feature_dir="$DATA_ROOT/generated_audio/voice1_high_quality" \
    +train_ds_meta.voice1.sample_weight=1.0 \
    "+train_ds_meta.voice1.tokenizer_names=[hebrew_chartokenizer]" \
    +train_ds_meta.voice2.manifest_path="$MANIFESTS/voice2_train.json" \
    +train_ds_meta.voice2.audio_dir="$DATA_ROOT/generated_audio/voice2_high_quality" \
    +train_ds_meta.voice2.feature_dir="$DATA_ROOT/generated_audio/voice2_high_quality" \
    +train_ds_meta.voice2.sample_weight=1.0 \
    "+train_ds_meta.voice2.tokenizer_names=[hebrew_chartokenizer]" \
    +train_ds_meta.voice3.manifest_path="$MANIFESTS/voice3_train.json" \
    +train_ds_meta.voice3.audio_dir="$DATA_ROOT/generated_audio/voice3_high_quality" \
    +train_ds_meta.voice3.feature_dir="$DATA_ROOT/generated_audio/voice3_high_quality" \
    +train_ds_meta.voice3.sample_weight=1.0 \
    "+train_ds_meta.voice3.tokenizer_names=[hebrew_chartokenizer]" \
    \
    +val_ds_meta.voice1_val.manifest_path="$MANIFESTS/voice1_val.json" \
    +val_ds_meta.voice1_val.audio_dir="$DATA_ROOT/generated_audio/voice1_high_quality" \
    +val_ds_meta.voice1_val.feature_dir="$DATA_ROOT/generated_audio/voice1_high_quality" \
    +val_ds_meta.voice1_val.sample_weight=1.0 \
    "+val_ds_meta.voice1_val.tokenizer_names=[hebrew_chartokenizer]" \
    +val_ds_meta.voice2_val.manifest_path="$MANIFESTS/voice2_val.json" \
    +val_ds_meta.voice2_val.audio_dir="$DATA_ROOT/generated_audio/voice2_high_quality" \
    +val_ds_meta.voice2_val.feature_dir="$DATA_ROOT/generated_audio/voice2_high_quality" \
    +val_ds_meta.voice2_val.sample_weight=1.0 \
    "+val_ds_meta.voice2_val.tokenizer_names=[hebrew_chartokenizer]" \
    +val_ds_meta.voice3_val.manifest_path="$MANIFESTS/voice3_val.json" \
    +val_ds_meta.voice3_val.audio_dir="$DATA_ROOT/generated_audio/voice3_high_quality" \
    +val_ds_meta.voice3_val.feature_dir="$DATA_ROOT/generated_audio/voice3_high_quality" \
    +val_ds_meta.voice3_val.sample_weight=1.0 \
    "+val_ds_meta.voice3_val.tokenizer_names=[hebrew_chartokenizer]" \
    \
    model.codecmodel_path=nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
    model.context_duration_min=5.0 \
    model.context_duration_max=5.0 \
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
