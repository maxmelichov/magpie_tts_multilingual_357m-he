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

import lightning.pytorch as pl
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import OmegaConf
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


@hydra_runner(config_path="conf/magpietts", config_name="magpietts")
def main(cfg):
    logging.info('\nConfig Params:\n%s', OmegaConf.to_yaml(cfg, resolve=True))
    mp.set_start_method("spawn", force=True)

    trainer = pl.Trainer(**cfg.trainer)
    trainer.callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step', log_weight_decay=True))
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = MagpieTTSModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

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
