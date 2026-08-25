#!/usr/bin/env python3
"""LoRA fine-tuning for NeMo MagpieTTS.

NeMo's MagpieTTS has no built-in PEFT support, so this script wraps the stock
training entrypoint (examples/tts/magpietts.py) with LoRA injection:

  1. Build MagpieTTSModel and load the pretrained weights
     (+init_from_nemo_model=...) exactly like the stock script.
  2. Replace the attention projection Linears (qkv_net, o_net, q_net, kv_net)
     in the encoder/decoder (and local transformer) with LoRA-augmented
     versions.
  3. Freeze everything except the LoRA matrices and the text embeddings
     (the new Hebrew tokenizer needs trainable text embeddings).

Configure via hydra overrides (all optional):
  +lora.r=16 +lora.alpha=32 +lora.dropout=0.0
  "+lora.targets=[qkv_net,o_net,q_net,kv_net]"
  "+lora.extra_trainable=[text_embedding]"

Checkpoints saved by exp_manager contain base + LoRA weights. Use
scripts/merge_lora.py to fold LoRA into the base weights for inference with
the stock architecture.
"""

import math
import tarfile
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch import nn

from nemo.collections.tts.models import MagpieTTSModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

# Hebrew IPA symbol set, exactly what appears in the training data.
HEBREW_IPA_CHARS = "abdefhijklmnoprstuvzɡʁʃʔχˈ"
HEBREW_IPA_PUNCT = [',', '.', '?', '!']

DEFAULT_TARGETS = ["qkv_net", "o_net", "q_net", "kv_net"]
# Modules kept fully trainable (substring match on parameter name). The text
# embedding must train: Hebrew byte tokens are new to the model.
DEFAULT_EXTRA_TRAINABLE = ["text_embedding", "baked_context_embedding"]


class LoRALinear(nn.Linear):
    """nn.Linear with an additive low-rank branch. Keeps the original
    parameter names ('weight'/'bias'), so base checkpoints stay compatible."""

    def init_lora(self, r: int, alpha: int, dropout: float):
        self.lora_r = r
        self.lora_scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev, dtype = self.weight.device, self.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        return out + F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.lora_scaling


def inject_lora(model: nn.Module, targets, r: int, alpha: int, dropout: float) -> int:
    replaced = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                lora = LoRALinear(child.in_features, child.out_features, bias=child.bias is not None)
                lora.weight = child.weight
                if child.bias is not None:
                    lora.bias = child.bias
                lora.init_lora(r, alpha, dropout)
                setattr(module, child_name, lora)
                replaced += 1
    return replaced


def freeze_non_lora(model: nn.Module, extra_trainable) -> None:
    # Match on the parameter's own module path, not a bare substring: "text_embedding"
    # as a substring also matches "context_text_embedding", which silently unfroze
    # the speaker-conditioning table.
    def matches(name):
        parts = name.split(".")
        return any(t in parts for t in extra_trainable)

    for name, p in model.named_parameters():
        trainable = "lora_" in name or matches(name)
        p.requires_grad = trainable
    # The frozen codec must stay frozen regardless.
    if hasattr(model, "_codec_model"):
        for p in model._codec_model.parameters():
            p.requires_grad = False


def extract_nemo(nemo_path: str, out_dir: Path) -> Path:
    """Unpack a .nemo archive (model_config.yaml, model_weights.ckpt, artifacts)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "model_weights.ckpt").exists():
        with tarfile.open(nemo_path) as tar:
            tar.extractall(out_dir, filter="data")
    return out_dir


def load_extra_tokenizers(spec_path):
    """Extra per-language IPA char tokenizers, from a JSON spec.

    {"<name>": {"chars": "...", "punct": [","...], "donor": "<base tokenizer>"}}

    Used by run 5, which adds IPA tokenizers for languages the base model ALREADY
    supports (en/de/it/es) so an IPA LoRA can be compared head-to-head against the
    base model's own g2p pipeline on the same audio.
    """
    import json as _json
    if not spec_path:
        return {}
    return _json.loads(Path(spec_path).read_text(encoding="utf-8"))


def build_tokenizer_roster(extracted: Path, hebrew_name: str = "hebrew_chartokenizer",
                           extra_spec=None):
    """The checkpoint's exact text_tokenizers (artifact paths resolved), with a
    new byte-level Hebrew tokenizer appended LAST so all pretrained token-ID
    offsets are preserved."""
    ckpt_cfg = OmegaConf.load(extracted / "model_config.yaml")
    tokenizers = OmegaConf.to_container(ckpt_cfg.text_tokenizers, resolve=True)

    def resolve(v):
        if isinstance(v, str) and v.startswith("nemo:"):
            return str(extracted / v[len("nemo:"):])
        if isinstance(v, dict):
            return {k: resolve(x) for k, x in v.items()}
        return v

    tokenizers = {name: resolve(t) for name, t in tokenizers.items()}
    assert hebrew_name not in tokenizers
    # One token per IPA symbol, matching how the base model tokenizes its other
    # IPA languages (byt5 is used only for orthographic text). NeMo only
    # instantiates targets inside its own namespace, so this is BaseCharsTokenizer
    # parameterized with the Hebrew IPA symbol set rather than a custom subclass.
    tokenizers[hebrew_name] = {
        "_target_": "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.BaseCharsTokenizer",
        "chars": HEBREW_IPA_CHARS,
        "punct": True,
        "apostrophe": False,
        "pad_with_space": False,
        "non_default_punct_list": list(HEBREW_IPA_PUNCT),
    }
    # Extra IPA tokenizers append AFTER Hebrew, so every previously assigned
    # token-ID offset (base 15 + Hebrew) is preserved.
    for name, spec in (extra_spec or {}).items():
        assert name not in tokenizers, f"tokenizer name collision: {name}"
        tokenizers[name] = {
            "_target_": "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.BaseCharsTokenizer",
            "chars": spec["chars"],
            "punct": True,
            "apostrophe": False,
            "pad_with_space": False,
            "non_default_punct_list": list(spec.get("punct", [",", ".", "?", "!"])),
        }
    return OmegaConf.create(tokenizers)


# Donor tokenizers for initializing the new Hebrew rows, best match first.
# Spanish first: a 5-vowel system like Hebrew. Portuguese supplies the uvular
# /ʁ/, Hindi the uvular fricative /χ/. 26 of the 27 Hebrew IPA symbols already
# exist in one of these, with embeddings the base model has already trained.
IPA_DONORS = [
    "spanish_phoneme",
    "portuguese_Brazilian_phoneme",
    "german_phoneme",
    "hindi_phoneme",
    "english_phoneme",
]
# /ʔ/ appears in no donor vocabulary. /h/ is the closest available: both glottal,
# differing in manner, which beats a random vector.
IPA_FALLBACK = {"ʔ": "h"}


def _donor_token_rows(agg, name):
    """Map token string -> global text_embedding row for one sub-tokenizer."""
    tok = agg.tokenizers[name]
    toks = list(tok.tokens) if hasattr(tok, "tokens") else list(tok.get_vocab().keys())
    off = agg.tokenizer_offsets[name]
    return {str(t): off + i for i, t in enumerate(toks)}


def init_extra_token_embeddings(model, extra_spec):
    """Seed each extra IPA tokenizer from the base model's matching language.

    en_ipa seeds from english_phoneme, de_ipa from german_phoneme, and so on, so
    a symbol the model already pronounces starts from the embedding it already
    learned rather than from noise.
    """
    if not extra_spec:
        return
    agg = model.tokenizer
    emb = model.text_embedding.weight.data
    for name, spec in extra_spec.items():
        if name not in agg.tokenizer_offsets:
            continue
        off = agg.tokenizer_offsets[name]
        toks = [str(t) for t in agg.tokenizers[name].tokens]
        order = [spec.get("donor")] + IPA_DONORS
        donors = {}
        for d in order:
            if d and d in agg.tokenizers and d not in donors:
                donors[d] = _donor_token_rows(agg, d)
        hit, miss = 0, []
        for i, tok in enumerate(toks):
            for rows in donors.values():
                if tok in rows:
                    emb[off + i] = emb[rows[tok]].clone()
                    hit += 1
                    break
            else:
                miss.append(tok)
        if miss:
            target = emb[:off].norm(dim=1).mean()
            for i, tok in enumerate(toks):
                if tok in miss:
                    row = emb[off + i]
                    emb[off + i] = row * (target / row.norm().clamp_min(1e-6))
        logging.info(f"{name}: seeded {hit}/{len(toks)} from {spec.get('donor')} (+fallbacks); "
                     f"no donor: {''.join(miss) if miss else 'none'}")


def init_new_token_embeddings(model, hebrew_name="hebrew_chartokenizer"):
    """Seed the new Hebrew rows from the same IPA symbol in a language the model
    already speaks, instead of leaving them randomly initialized.

    The base checkpoint carries six IPATokenizer vocabularies, so nearly every
    Hebrew phone is a symbol it has already learned to pronounce; only the
    tokenizer that owns the row is new. Copying those rows starts training from
    a working pronunciation rather than from noise.
    """
    agg = model.tokenizer
    emb = model.text_embedding.weight.data
    heb_off = agg.tokenizer_offsets[hebrew_name]
    heb_tokens = [str(t) for t in agg.tokenizers[hebrew_name].tokens]

    donors = {d: _donor_token_rows(agg, d) for d in IPA_DONORS if d in agg.tokenizers}
    copied, missing = [], []
    for local_i, tok in enumerate(heb_tokens):
        want = IPA_FALLBACK.get(tok, tok)
        for dname, rows in donors.items():
            if want in rows:
                emb[heb_off + local_i] = emb[rows[want]].clone()
                copied.append(f"{tok}<-{dname.split('_')[0]}"
                              + (f":{want}" if want != tok else ""))
                break
        else:
            missing.append(tok)

    # Anything with no donor keeps its fresh init, but rescaled to the norm of the
    # pretrained table -- fresh init is several times too large otherwise.
    if missing:
        target = emb[:heb_off].norm(dim=1).mean()
        for local_i, tok in enumerate(heb_tokens):
            if tok in missing:
                row = emb[heb_off + local_i]
                emb[heb_off + local_i] = row * (target / row.norm().clamp_min(1e-6))

    logging.info(f"Hebrew embeddings seeded from existing IPA rows ({len(copied)}/{len(heb_tokens)}): "
                 f"{' '.join(copied)}")
    if missing:
        logging.info(f"  no donor, rescaled fresh init: {missing}")


def extend_baked_speakers(model, n_new: int, seed_noise: float = 0.15):
    """Give each Hebrew speaker its own baked context embedding.

    The released checkpoint has no context_encoder -- NeMo strips it when baked
    embeddings are present -- so `context_audio` and `context_text` are both
    ignored and voice is chosen solely by indexing this table. It ships 5 rows
    (Aria, Jason, John, Leo, Sofia). Training every Hebrew speaker against the
    single default row is what left the output voice an unconditioned average.

    New rows are appended after the 5 originals (so existing indices keep their
    voices) and seeded from a *different* original each, plus noise: identical
    seeds give identical gradients and the voices would never separate.
    """
    emb = model.baked_context_embedding
    n_old, dim = emb.weight.shape
    new = nn.Embedding(n_old + n_new, dim, device=emb.weight.device, dtype=emb.weight.dtype)
    with torch.no_grad():
        new.weight[:n_old] = emb.weight
        for i in range(n_new):
            src = i % n_old                      # rotate donors to break symmetry
            new.weight[n_old + i] = emb.weight[src]
            new.weight[n_old + i] += torch.randn_like(emb.weight[src]) * seed_noise * emb.weight[src].std()
    model.baked_context_embedding = new

    lens = model.baked_context_embedding_len
    model.baked_context_embedding_len = torch.cat(
        [lens, torch.stack([lens[i % n_old] for i in range(n_new)])]
    ).to(lens.device)
    # Freeze the 5 released voices: measured without this, Hebrew training drifts
    # Aria, Jason, John, Leo and Sofia away from what NVIDIA shipped.
    mask = torch.zeros(n_old + n_new, 1)
    mask[n_old:] = 1.0
    new.weight.register_hook(lambda g, m=mask: g * m.to(g.device, g.dtype))
    logging.info(f"baked speakers: {n_old} -> {n_old + n_new} "
                 f"(hebrew voices occupy indices {n_old}..{n_old + n_new - 1}; "
                 f"original {n_old} frozen)")


def freeze_pretrained_embedding_rows(model, new_tokenizers=("hebrew_chartokenizer",)):
    """Let gradients reach only the NEW tokenizers' rows of text_embedding.

    text_embedding is shared across every language. Leaving the whole table
    trainable lets this run drag the base model's phoneme embeddings around, and
    BOS/EOS with them. A gradient mask keeps the new rows learnable while every
    pretrained row stays exactly as released.

    Every added tokenizer must be listed: masking only Hebrew while also adding
    en/de/it/es IPA tokenizers silently froze those 231 rows, so they could never
    learn anything.
    """
    agg = model.tokenizer
    w = model.text_embedding.weight
    mask = torch.zeros(w.shape[0], 1)
    spans = []
    for name in new_tokenizers:
        if name not in agg.tokenizer_offsets:
            continue
        start = agg.tokenizer_offsets[name]
        n = agg.num_tokens_per_tokenizer[name]
        mask[start: start + n] = 1.0          # BOS/EOS stay frozen: they are pretrained
        spans.append(f"{name}[{start}, {start + n})")
    trainable = int(mask.sum().item())
    w.register_hook(lambda grad, m=mask: grad * m.to(grad.device, grad.dtype))
    logging.info(f"text_embedding: training {trainable} rows -- {', '.join(spans)}; "
                 f"{w.shape[0] - trainable} pretrained rows frozen (incl. BOS/EOS)")


def load_pretrained_weights(model: MagpieTTSModel, extracted: Path, seed_from_ipa: bool = True):
    """Load checkpoint weights into the Hebrew-extended text embedding table.

    The table is laid out [base vocab | new Hebrew block | BOS | EOS], so BOS and
    EOS MOVE when the Hebrew block is appended. Copying the checkpoint rows as one
    contiguous prefix would drop the pretrained BOS/EOS onto the first two Hebrew
    tokens and leave the real BOS/EOS randomly initialized -- the model would have
    to relearn sequence start/end from scratch. Copy the vocabulary and the two
    special rows separately.
    """
    sd = torch.load(extracted / "model_weights.ckpt", map_location="cpu", weights_only=False)
    if not isinstance(sd, dict) or "text_embedding.weight" not in sd:
        sd = sd.get("state_dict", sd)
    ckpt_emb = sd["text_embedding.weight"]
    model_emb = model.text_embedding.weight.data
    if model_emb.shape[0] < ckpt_emb.shape[0]:
        raise RuntimeError(f"model text_embedding {model_emb.shape} smaller than checkpoint {ckpt_emb.shape}")

    base_vocab = ckpt_emb.shape[0] - 2          # checkpoint rows minus its BOS/EOS
    n_new = model_emb.shape[0] - ckpt_emb.shape[0]
    padded = model_emb.clone()
    padded[:base_vocab] = ckpt_emb[:base_vocab]  # shared vocabulary
    padded[-2] = ckpt_emb[-2]                    # BOS, at its new position
    padded[-1] = ckpt_emb[-1]                    # EOS
    sd["text_embedding.weight"] = padded
    logging.info(f"text_embedding: {base_vocab} pretrained rows + {n_new} new Hebrew rows "
                 f"+ pretrained BOS/EOS remapped to rows {model_emb.shape[0] - 2}/{model_emb.shape[0] - 1}")
    model.load_state_dict(sd, strict=True)

    if seed_from_ipa:
        init_new_token_embeddings(model)


@hydra_runner(config_path="conf/magpietts", config_name="magpietts")
def main(cfg):
    mp.set_start_method("spawn", force=True)

    nemo_path = cfg.get("init_from_nemo_model")
    if nemo_path is None:
        raise ValueError("pass +init_from_nemo_model=/path/to/model.nemo")
    extracted = extract_nemo(nemo_path, Path(nemo_path).parent / "extracted")

    # The stock yaml's architecture defaults don't match the released checkpoint
    # (frame stacking, codebook count, ...). Use the checkpoint's own model config
    # as the base and overlay only training-specific keys from our yaml/CLI.
    ours = OmegaConf.to_container(cfg.model, resolve=True)
    ckpt_model = OmegaConf.to_container(OmegaConf.load(extracted / "model_config.yaml"), resolve=True)
    for key in [
        "train_ds", "validation_ds", "optim", "codecmodel_path",
        "context_duration_min", "context_duration_max", "alignment_loss_scale",
        "prior_scaling_factor", "load_cached_codes_if_available",
        "max_epochs", "steps_per_epoch", "cfg_unconditional_prob",
    ]:
        if key in ours:
            ckpt_model[key] = ours[key]
        else:
            ckpt_model.pop(key, None)
    with open_dict(cfg):
        cfg.model = OmegaConf.create(ckpt_model)
        # The checkpoint's full tokenizer roster + Hebrew appended last.
        extra_spec = load_extra_tokenizers(cfg.get("extra_tokenizers_json"))
        cfg.model.text_tokenizers = build_tokenizer_roster(extracted, extra_spec=extra_spec)
        del cfg["init_from_nemo_model"]  # weights are loaded manually below

    logging.info('\nConfig Params:\n%s', OmegaConf.to_yaml(cfg, resolve=True))
    trainer = pl.Trainer(**cfg.trainer)
    trainer.callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step', log_weight_decay=True))
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = MagpieTTSModel(cfg=cfg.model, trainer=trainer)
    load_pretrained_weights(model, extracted,
                            seed_from_ipa=bool(cfg.get("seed_hebrew_from_ipa", True)))
    init_extra_token_embeddings(model, extra_spec)

    lora_cfg = cfg.get("lora", OmegaConf.create({}))
    r = int(lora_cfg.get("r", 16))
    alpha = int(lora_cfg.get("alpha", 32))
    dropout = float(lora_cfg.get("dropout", 0.0))
    targets = list(lora_cfg.get("targets", DEFAULT_TARGETS))
    extra_trainable = list(lora_cfg.get("extra_trainable", DEFAULT_EXTRA_TRAINABLE))

    n = inject_lora(model, targets, r, alpha, dropout)
    freeze_non_lora(model, extra_trainable)
    n_new_spk = int(cfg.get("num_new_speakers", 0))
    if n_new_spk:
        extend_baked_speakers(model, n_new_spk)
        freeze_non_lora(model, extra_trainable)   # re-apply: the table was replaced
    if bool(cfg.get("freeze_pretrained_embeddings", True)):
        freeze_pretrained_embedding_rows(
            model, ["hebrew_chartokenizer"] + list((extra_spec or {}).keys()))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(
        f"LoRA: injected {n} adapters (r={r}, alpha={alpha}, targets={targets}); "
        f"trainable params {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
        f"({100 * trainable / total:.2f}%)"
    )
    if n == 0:
        raise RuntimeError("No LoRA adapters injected — target module names did not match")

    logging.info("Starting LoRA training...")
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
