# MagpieTTS Multilingual 357M — Hebrew Fine-tuning

Fine-tunes [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)
on Hebrew using **IPA phoneme input**, following NeMo's
["adding a new language" recipe](https://docs.nvidia.com/nemo/speech/nightly/tts/magpietts-finetuning.html).

## Approach

- **Model**: 357M encoder-decoder transformer (NeMo MagpieTTS), NanoCodec @ 22.05 kHz,
  voice cloning via a ~5 s context audio.
- **Hebrew input = IPA**: the training CSVs already contain stress-marked IPA
  (e.g. `ʔanˈaχnu beʔˈad sifʁijˈa leʔumˈit.`). Since Hebrew is a new language for the
  model, we register a byte-level tokenizer `hebrew_chartokenizer` (`google/byt5-small`)
  and feed the IPA strings directly as `text`. No G2P runs at train or inference time —
  you always pass IPA.
- **Data** (from `/home/maxm/AE_training_data_all`):
  | dataset | audio | rows | filter |
  |---|---|---|---|
  | voice1 | `generated_audio/voice1_high_quality` (44.1 kHz) | ~60k | pre-filtered |
  | voice2 | `generated_audio/voice2_high_quality` | ~88k CSV rows | `passed_filter` + WER ≤ 0.2 |
  | voice3 | `generated_audio/voice3_high_quality` | ~52k wavs | `passed_filter` + WER ≤ 0.2 |

  `slow_44K` Hebrew speakers (time-stretched Chatterbox audio) are available via
  `--include-slow44k` but off by default — the slowed speaking rate would bias prosody.
  NeMo resamples 44.1 kHz → 22.05 kHz on the fly.
- **Context pairing**: each utterance gets a random same-voice utterance (≥ 5 s) as the
  voice-cloning reference (`context_audio_filepath` / `context_text`).

## Usage

```bash
# 1. env + NeMo + checkpoint (downloads ~1.5 GB)
bash scripts/setup.sh

# 2. build train/val manifests (CPU only, a few minutes for ~170k wav headers)
venv/bin/python scripts/build_manifests.py

# 3. train — defaults to GPU 1 only (cuda:0 busy). DO NOT start while other jobs need the GPU.
bash scripts/train_hebrew.sh
#   both GPUs:  GPU=0,1 bash scripts/train_hebrew.sh
#   overrides:  BATCH_SIZE=8 LR=1e-5 MAX_EPOCHS=100 STEPS_PER_EPOCH=1000
#   extra hydra overrides pass through: bash scripts/train_hebrew.sh trainer.log_every_n_steps=10

# 4. synthesize (IPA in, wav out)
venv/bin/python scripts/infer_hebrew.py \
  --checkpoint experiments/Magpie-TTS/<run>/checkpoints/<best>.ckpt \
  --hparams   experiments/Magpie-TTS/<run>/hparams.yaml \
  --context-audio /home/maxm/AE_training_data_all/generated_audio/voice1_high_quality/voice1_knesset_012062.wav \
  --text "ʔanˈaχnu beʔˈad sifʁijˈa leʔumˈit." \
  --out-dir outputs/test1
```

## LoRA

NeMo MagpieTTS has no built-in PEFT, so `scripts/magpietts_lora.py` implements it:
LoRA adapters (default r=16, α=32) are injected into the attention projections
(`qkv_net`, `o_net`, `q_net`, `kv_net`) of the encoder/decoder/local transformer,
and everything else is frozen **except the text embeddings** (Hebrew byte tokens are
new to the model, so `text_embedding` must train). Checkpoints contain base + LoRA
weights; fold them for stock-architecture inference with:

```bash
venv/bin/python scripts/merge_lora.py --ckpt <run>/checkpoints/<best>.ckpt --out merged.ckpt
```

Tune via env (`LORA_R`, `LORA_ALPHA`) or hydra (`+lora.dropout=0.05`,
`"+lora.targets=[...]"`). `scripts/ds_meta_args.py` generates the per-dataset hydra
overrides from `data/manifests/datasets.json` (15 Hebrew datasets: 3 Knesset voices +
12 slow_44K speakers).

## Key hyperparameters

| setting | value | why |
|---|---|---|
| `model.optim.lr` | `1e-4`, no schedule | standard LoRA LR (use `1e-5` for full fine-tune) |
| `model.alignment_loss_scale` / `prior_scaling_factor` | `0.0` / `null` | alignment prior over-constrains fine-tuning |
| `trainer.precision` | `32` | fine-tuning stability |
| `model.context_duration_{min,max}` | `5.0` | fixed 5 s cloning context |
| `weighted_sampling_steps_per_epoch` | `1000` | balanced sampling across the 3 voices |
| `batch_size` | `8` | fp32 on a 32 GB RTX 5090; raise if memory allows |
| init | `+init_from_nemo_model=checkpoints/magpie_tts_multilingual_357m.nemo` | pretrained weights |

## Notes

- Monitor `experiments/` with TensorBoard; watch val loss — tens of epochs is often enough.
- To also preserve the model's original languages, mix in some of the original-language data
  (e.g. HiFiTTS under `datasets_4AE_extracted/`) as extra `train_ds_meta` entries with the
  built-in tokenizers; pure-Hebrew fine-tuning will degrade the other languages.
- Codec codes are computed on the fly. For faster epochs, pre-extract codes and add
  `target_audio_codes_path` / `context_audio_codes_path` to the manifests
  (`model.load_cached_codes_if_available=true` is already set).
