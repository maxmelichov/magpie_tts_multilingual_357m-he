# MagpieTTS Multilingual 357M — Hebrew

<img src="assets/banner.jpg" alt="MagpieTTS Hebrew" width="100%">

Hebrew fine-tune of [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m).
**Input is IPA, not Hebrew letters** — there is no G2P in the model, you pass phonemes.

Model: [notmax123/magpie_tts_multilingual_357m-he](https://huggingface.co/notmax123/magpie_tts_multilingual_357m-he)

## Results

1,525 utterances of [Phonikud/ILSpeech](https://huggingface.co/datasets/Phonikud/ILSpeech) — real
recorded Hebrew, not in our training data. Every metric sits next to the same metric on the **real
recordings**, because the ASR has its own error rate.

| metric | synthesized | real recordings |
|---|---|---|
| Hebrew WER | 10.4% | 10.2% |
| Hebrew CER | 4.7% | 4.8% |
| IPA WER | 5.0% | 3.3% |
| IPA PER | 0.9% | 0.7% |
| pace vs ground truth | 0.97× | 1.00 |

Reading Hebrew works: WER lands within 0.2 points of human recordings and CER is slightly better.
ASR models used: `ivrit-ai/whisper-large-v3-turbo` (Hebrew), `notmax123/whisper-he-ipa` (IPA).

## Limitations

**No voice cloning, and no voice selection.** The released base checkpoint has the `context_encoder`
weights removed entirely (0 tensors in the file), replaced by 5 baked speaker embeddings — NVIDIA
dropped zero-shot cloning for security reasons. So `context_audio` and `context_text` are both
ignored: **the reference clip has no effect on the output voice.** All 14 Hebrew speakers trained
against the same baked embedding, so the output is an inconsistent average rather than one stable
voice (speaker self-similarity 0.36, where same-speaker is ~0.80). Fixing this means extending the
baked speaker embedding table and passing `speaker_indices` — not more training.

**Training data is synthetic** (TTS output verified by ASR, not studio recordings), which caps
naturalness near the teacher systems'.

**Other languages are intact.** English WER measured identical to the base model (4.2%, same
transcripts) — training was masked to the 33 new Hebrew embedding rows only.

## Usage

```bash
bash scripts/setup.sh                          # venv + NeMo + base checkpoint
venv/bin/python scripts/build_manifests.py     # manifests from the Hebrew CSVs
bash scripts/train_hebrew.sh                   # train (both GPUs)

venv/bin/python scripts/merge_lora.py \
  --ckpt experiments/Magpie-TTS/checkpoints/<best>.ckpt --out merged.ckpt

venv/bin/python scripts/infer_hebrew.py \
  --checkpoint merged.ckpt --hparams experiments/Magpie-TTS/version_0/hparams.yaml \
  --text "ʃalˈom, mˈa ʃlomχˈa?" --context-audio <any 10s+ wav> --out-dir outputs/
```

Phonemize with [Phonikud](https://github.com/thewh1teagle/phonikud). Supported symbols (27):

```
a b d e f h i j k l m n o p r s t u v z ɡ ʁ ʃ ʔ χ ˈ   plus  , . ? !
```

`--context-audio` is required by the plumbing but does not change the voice (see Limitations).

## Evaluation

```bash
venv/bin/python scripts/build_ilspeech_eval.py --split all --out-name eval_full.json
venv/bin/python scripts/infer_hebrew.py --manifest data/ilspeech/eval/eval_full.json ...
venv/bin/python scripts/score_ilspeech.py --pred-dir outputs/<run>/audio/repeat_0
```

`scripts/diagnose_speaker_conditioning.py` checks whether the reference clip changes the output at
all, printing same-speaker and different-speaker anchors so the similarity number has a scale.

## How it works

LoRA (r=16, α=32) on attention projections — 78 adapters, 4.7M of 375M params trainable, merged into
the base weights for release. 80k steps on 2× RTX 5090.

Hebrew is a 16th tokenizer appended to the checkpoint's 15, so pretrained token IDs keep their
offsets. Four things that were not obvious and cost real debugging time:

- **BOS/EOS move** when you append tokens. The table is `[vocab | new | BOS | EOS]`, so copying the
  checkpoint as one contiguous prefix drops the pretrained BOS/EOS onto the first two new tokens and
  leaves the real ones random. Copy vocabulary and special rows separately.
- **26 of 27 Hebrew IPA symbols already exist** in the base model's Spanish/Portuguese/Hindi phoneme
  vocabularies. Seed the new rows from those instead of from noise; only `ʔ` is genuinely new.
- **The shared embedding table spans all 16 languages.** A gradient mask keeps training to the new
  rows, otherwise Hebrew drags the other languages with it.
- **The released checkpoint uses frame stacking (factor 2)**, so the stock config will not load into
  it — rebuild the model config from the checkpoint's own `model_config.yaml` — and context clips
  must be >= 10 s.

Data: ~285k utterances / 292 hours across 14 Hebrew speakers, filtered to `wer_score == 0.0`
(exact ASR agreement). That strict filter beat `<= 0.2` by 0.36% WER (95% CI [-0.65, -0.06]) — real,
but small, and it costs 63% of the data.

## License

Derivative of `nvidia/magpie_tts_multilingual_357m` under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
ILSpeech is non-commercial and was used for evaluation only — none of it is in the weights.
