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

DEFAULT_TARGETS = ["qkv_net", "o_net", "q_net", "kv_net"]
# Modules kept fully trainable (substring match on parameter name). The text
# embedding must train: Hebrew byte tokens are new to the model.
DEFAULT_EXTRA_TRAINABLE = ["text_embedding"]


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
    for name, p in model.named_parameters():
        trainable = "lora_" in name or any(s in name for s in extra_trainable)
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


def build_tokenizer_roster(extracted: Path, hebrew_name: str = "hebrew_chartokenizer"):
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
    tokenizers[hebrew_name] = {"_target_": "AutoTokenizer", "pretrained_model": "google/byt5-small"}
    return OmegaConf.create(tokenizers)


def load_pretrained_weights(model: MagpieTTSModel, extracted: Path):
    """Load checkpoint weights, padding text_embedding for the appended Hebrew
    tokenizer block (new rows keep the model's fresh initialization)."""
    sd = torch.load(extracted / "model_weights.ckpt", map_location="cpu", weights_only=False)
    if not isinstance(sd, dict) or "text_embedding.weight" not in sd:
        sd = sd.get("state_dict", sd)
    ckpt_emb = sd["text_embedding.weight"]
    model_emb = model.text_embedding.weight.data
    if model_emb.shape[0] < ckpt_emb.shape[0]:
        raise RuntimeError(f"model text_embedding {model_emb.shape} smaller than checkpoint {ckpt_emb.shape}")
    padded = model_emb.clone()
    padded[: ckpt_emb.shape[0]] = ckpt_emb
    sd["text_embedding.weight"] = padded
    logging.info(
        f"text_embedding: {ckpt_emb.shape[0]} pretrained rows + "
        f"{model_emb.shape[0] - ckpt_emb.shape[0]} new Hebrew rows"
    )
    model.load_state_dict(sd, strict=True)


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
        cfg.model.text_tokenizers = build_tokenizer_roster(extracted)
        del cfg["init_from_nemo_model"]  # weights are loaded manually below

    logging.info('\nConfig Params:\n%s', OmegaConf.to_yaml(cfg, resolve=True))
    trainer = pl.Trainer(**cfg.trainer)
    trainer.callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step', log_weight_decay=True))
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = MagpieTTSModel(cfg=cfg.model, trainer=trainer)
    load_pretrained_weights(model, extracted)

    lora_cfg = cfg.get("lora", OmegaConf.create({}))
    r = int(lora_cfg.get("r", 16))
    alpha = int(lora_cfg.get("alpha", 32))
    dropout = float(lora_cfg.get("dropout", 0.0))
    targets = list(lora_cfg.get("targets", DEFAULT_TARGETS))
    extra_trainable = list(lora_cfg.get("extra_trainable", DEFAULT_EXTRA_TRAINABLE))

    n = inject_lora(model, targets, r, alpha, dropout)
    freeze_non_lora(model, extra_trainable)

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
