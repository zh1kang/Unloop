#!/usr/bin/env bash
# CUDA environment for the full run. vLLM 0.33.0 pins torch 2.14.0 and transformers < 5.21, so the cluster uses
# its own virtual environment instead of the uv.lock environment (torch 2.15, transformers 5.22) used locally.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv-cluster --python 3.12
# shellcheck disable=SC1091
. .venv-cluster/bin/activate
uv pip install "vllm==0.33.0" "math-verify==0.9.0" "datasets>=4" matplotlib safetensors
uv pip install --no-deps -e .
# Optional: CUDA kernels for the Gated DeltaNet layers; without them the hf backend uses slow PyTorch fallbacks.
uv pip install flash-linear-attention || echo "flash-linear-attention not installed; hf backend will be slow"
uv pip install --no-build-isolation causal-conv1d || echo "causal-conv1d not installed; hf backend will be slow"
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
