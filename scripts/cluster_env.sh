# Source this before any unloop command on a cluster: `. scripts/cluster_env.sh`.
# It keeps every cache inside the repository's .cache directory, so nothing is written to $HOME,
# and activates .venv-cluster if it exists. With UNLOOP_OFFLINE=1 (compute nodes without internet),
# Hugging Face reads only from the cache that `unloop fetch` filled on a login node.
UNLOOP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
C=$UNLOOP_ROOT/.cache
export HF_HOME=$C/hf
export UV_CACHE_DIR=$C/uv UV_PYTHON_INSTALL_DIR=$C/uv-python PIP_CACHE_DIR=$C/pip
export XDG_CACHE_HOME=$C/xdg XDG_CONFIG_HOME=$C/config MPLCONFIGDIR=$C/matplotlib
export TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor CUDA_CACHE_PATH=$C/nv
export VLLM_CACHE_ROOT=$C/vllm VLLM_CONFIG_ROOT=$C/vllm-config FLASHINFER_WORKSPACE_BASE=$C
export TMPDIR=$C/tmp
mkdir -p "$TMPDIR"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
unset TRANSFORMERS_CACHE HF_HUB_CACHE HF_DATASETS_CACHE  # these would override HF_HOME
O=0
if [ -n "${UNLOOP_OFFLINE:-}" ]; then O=1; fi
export HF_HUB_OFFLINE=$O HF_DATASETS_OFFLINE=$O TRANSFORMERS_OFFLINE=$O
if [ -f "$UNLOOP_ROOT/.venv-cluster/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$UNLOOP_ROOT/.venv-cluster/bin/activate"
fi
