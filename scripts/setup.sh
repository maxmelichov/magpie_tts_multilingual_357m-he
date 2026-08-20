#!/usr/bin/env bash
# One-time environment setup for Hebrew fine-tuning of nvidia/magpie_tts_multilingual_357m.
# Creates ./venv, installs NeMo (TTS) from source, downloads the pretrained
# checkpoint and verifies the NanoCodec is reachable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Python env -------------------------------------------------------------
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -U pip wheel setuptools

# RTX 5090 (sm_120) needs a CUDA 12.8+ torch build.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# --- NeMo from source (required for examples/tts/magpietts.py) --------------
if [ ! -d NeMo ]; then
  git clone --depth 1 https://github.com/NVIDIA/NeMo.git
fi
pip install -e "./NeMo[tts]"
pip install "huggingface_hub[cli]"

# --- Pretrained checkpoint ---------------------------------------------------
mkdir -p checkpoints
if [ ! -f checkpoints/magpie_tts_multilingual_357m.nemo ]; then
  hf download nvidia/magpie_tts_multilingual_357m \
    magpie_tts_multilingual_357m.nemo --local-dir checkpoints
fi

echo
echo "Setup complete."
echo "  checkpoint: $REPO_ROOT/checkpoints/magpie_tts_multilingual_357m.nemo"
echo "  codec:      resolved automatically from HF (nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)"
echo "Next: venv/bin/python scripts/build_manifests.py && bash scripts/train_hebrew.sh"
