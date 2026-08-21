# MagpieTTS Multilingual 357M — Hebrew Fine-tuning

Fine-tunes [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)
on Hebrew using **IPA phoneme input**, following NeMo's
["adding a new language" recipe](https://docs.nvidia.com/nemo/speech/nightly/tts/magpietts-finetuning.html).

Trained model: LoRA fine-tune, **110,000 steps, val_loss 9.345**, on 2× RTX 5090.

## Approach

- **Model**: 357M encoder-decoder transformer (NeMo MagpieTTS), NanoCodec @ 22.05 kHz,
  voice cloning from a 10 s context clip.
- **Hebrew input = IPA**. The training CSVs contain stress-marked IPA
  (e.g. `ʔanˈaχnu beʔˈad sifʁijˈa leʔumˈit.`), which is fed directly as `text` —
  no G2P runs at train or inference time, you always pass IPA.
  Hebrew is a new language for this model, so a 16th tokenizer
  (`hebrew_chartokenizer`) is appended to the checkpoint's existing 15, and the
  text-embedding table is padded with 33 fresh rows. Appending last preserves
  every pretrained token-ID offset.
- **Tokenizer: one token per IPA symbol** (33-symbol vocabulary), not byte-level.
  This matches how the base model tokenizes its 8 other IPA languages; byt5 is what
  NVIDIA uses for orthographic text (French, Italian, Korean). Implemented as NeMo's
  `BaseCharsTokenizer` parameterized with the Hebrew IPA charset — NeMo blocks
  instantiation of targets outside its own namespace, so a custom subclass will not load.
- **Data**: 15 Hebrew datasets, ~778k utterances / ~1,079 hours, from
  `/home/maxm/AE_training_data_all` — 3 Knesset voices (`generated_audio/voice{1,2,3}_high_quality`)
  plus 12 `slow_44K` speakers. Filtered by `passed_filter` + whisper WER ≤ 0.2, and by
  charset (283 utterances dropped for mojibake / foreign-script leakage).
  NeMo resamples 44.1 kHz → 22.05 kHz on the fly.
- **Context pairing**: each utterance gets a random same-voice utterance (≥ 5 s) as its
  voice-cloning reference (`context_audio_filepath` / `context_text` / `context_audio_duration`).

## Usage

```bash
# 1. env + NeMo + checkpoint (downloads ~1.5 GB)
bash scripts/setup.sh

# 2. build train/val manifests (CPU only; reads ~1.1M wav headers)
venv/bin/python scripts/build_manifests.py

# 3. train (both GPUs by default)
bash scripts/train_hebrew.sh
#   single GPU:  GPU=1 bash scripts/train_hebrew.sh
#   overrides:   BATCH_SIZE=8 LR=1e-4 MAX_EPOCHS=100 STEPS_PER_EPOCH=2000 LORA_R=16
#   extra hydra overrides pass through: bash scripts/train_hebrew.sh trainer.log_every_n_steps=10

# 4. fold LoRA into the base weights for inference
venv/bin/python scripts/merge_lora.py \
  --ckpt experiments/Magpie-TTS/checkpoints/<best>.ckpt --out merged.ckpt

# 5. synthesize (IPA in, wav out)
venv/bin/python scripts/infer_hebrew.py \
  --checkpoint merged.ckpt \
  --hparams   experiments/Magpie-TTS/version_0/hparams.yaml \
  --context-audio /home/maxm/AE_training_data_all/generated_audio/voice1_high_quality/voice1_knesset_007634.wav \
  --context-text "<IPA of the context clip>" \
  --text "ʔanˈaχnu beʔˈad sifʁijˈa leʔumˈit." \
  --out-dir outputs/test1
```

**Inference quality flags matter.** NeMo's inference script defaults classifier-free
guidance and the local transformer (multi-codebook refinement) to *off*, even though
both are trained. `infer_hebrew.py` turns both **on by default** — A/B testing showed
each audibly reduces robotic artifacts. Disable with `--no-use-cfg` /
`--no-use-local-transformer`.

## LoRA

NeMo MagpieTTS has no built-in PEFT, so `scripts/magpietts_lora.py` implements it:
adapters (default r=16, α=32) are injected into the attention projections
(`qkv_net`, `o_net`, `q_net`, `kv_net`) of the encoder/decoder/local transformer —
78 adapters, ~5.5M of 375M params trainable (1.5%). Everything else is frozen
**except the text embeddings**, which must train because the Hebrew tokens are new.

It also rebuilds the model config from the checkpoint's own `model_config.yaml` rather
than the stock YAML — the released checkpoint uses frame stacking (factor 2) and a
different codebook layout, and the stock config will not load into it. Frame stacking
is also why `context_duration_{min,max}` must be ≥ 10 s.

Tune via env (`LORA_R`, `LORA_ALPHA`) or hydra (`+lora.dropout=0.05`,
`"+lora.targets=[...]"`). `scripts/ds_meta_args.py` generates per-dataset hydra
overrides from `data/manifests/datasets.json`.

## Key hyperparameters

| setting | value | why |
|---|---|---|
| `model.optim.lr` | `1e-4`, no schedule | LoRA LR (use `1e-5` for a full fine-tune) |
| `model.alignment_loss_scale` / `prior_scaling_factor` | `0.0` / `null` | alignment prior over-constrains fine-tuning |
| `trainer.precision` | `32` | fine-tuning stability |
| `model.context_duration_{min,max}` | `10.0` | required by frame stacking factor 2 |
| `weighted_sampling_steps_per_epoch` | `2000` | balanced sampling across 15 uneven datasets |
| `batch_size` | `8` per GPU | fp32 on a 32 GB RTX 5090 |

## Results and limitations

Validation loss converged well before the configured 200k steps:

| step | val_loss |
|---|---|
| 2,000 | 10.054 |
| 8,000 | 9.676 |
| 40,000 | 9.436 |
| 90,000 | 9.361 |
| 110,000 | 9.345 |

Gains per 10k steps fell from −0.20 early to −0.003 by step 90k. **More steps are not
the lever** past roughly 50k.

The remaining naturalness ceiling comes from the training data, which is synthetic:
noise floors measure 0.0001–0.0004 with near-zero variance across files (digital
silence, not recorded rooms), the audio lives under `generated_audio/`, and the CSVs
carry an `original_phonemes → whisper_phonemes → wer_score` synthesize-then-ASR-verify
structure. Training on TTS output caps naturalness near the teacher system's.
`slow_44K` is additionally time-stretched (`atempo=0.85` per `resample_and_slow.py`),
though measured speaking rates end up within 4% of the Knesset voices (15.13 vs
15.76 phonemes/sec), so the stretch matters less than it first appears.

**To improve further, add real recorded Hebrew** and weight it heavily via
`sample_weight`. Re-weighting between existing sources will not help — they share the
same synthetic ceiling.

## Notes

- Monitor `experiments/` with TensorBoard.
- Pure-Hebrew fine-tuning degrades the model's original 12 languages. To preserve them,
  mix in original-language data (e.g. HiFiTTS under `datasets_4AE_extracted/`) as extra
  `train_ds_meta` entries using the built-in tokenizers.
- Codec codes are computed on the fly. To speed up epochs, pre-extract codes and add
  `target_audio_codes_path` / `context_audio_codes_path` to the manifests
  (`model.load_cached_codes_if_available=true` is already set).
