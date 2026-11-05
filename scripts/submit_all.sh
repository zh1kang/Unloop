#!/usr/bin/env bash
# Submit the study to Slurm: one analysis job, then one single-GPU job per condition.
# Conditions with an edit or a hook wait for the analysis job, because they read its directions and sweep.
# Usage: scripts/submit_all.sh [config] [condition ...]
#   SBATCH_ARGS="..."    extra sbatch options for every job.
#                        On Princeton Della, do not pass --partition=gpu (the wrapper rejects it):
#                        SBATCH_ARGS="--account=jordanat --constraint=gpu80"
#   SKIP_ANALYSIS=1      do not submit the analysis job (its results already exist)
#   VLLM_TIME, HF_TIME   time limits of the condition jobs (default 24:00:00 and 72:00:00)
# When the jobs finish: `unloop table --run <name>` (no GPU needed).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. scripts/cluster_env.sh
CONFIG=${1:-configs/full.toml}
shift || true
mkdir -p logs
read -r -a EXTRA <<< "${SBATCH_ARGS:-}"
sub() { sbatch --parsable ${EXTRA[@]+"${EXTRA[@]}"} "$@"; }
ANALYSIS=
if [ -z "${SKIP_ANALYSIS:-}" ]; then
  ANALYSIS=$(sub --job-name=unloop-analysis --time=08:00:00 scripts/job.sbatch bash -c \
    "unloop reproduce && unloop directions && unloop sweep && unloop explain && unloop origin")
  echo "analysis: job $ANALYSIS"
fi
python - "$CONFIG" "$@" <<'EOF' |
import sys, tomllib
from pathlib import Path

cfg = tomllib.load(open(sys.argv[1], "rb"))
rank1_keys = set()
path = Path("results/directions.safetensors")
if path.exists():
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        rank1_keys = set(f.keys())
for name in sys.argv[2:] or cfg["conditions"]:
    spec = cfg["conditions"][name]
    edit = spec.get("edit")
    if isinstance(edit, dict) and str(edit.get("direction", "")).startswith("rank1"):
        layer = edit["layers"][0]
        if f"{edit['direction']}.u.{layer}" not in rank1_keys:
            print(f"skip {name}: run `unloop fit-rank1` before this condition", file=sys.stderr)
            continue
    print(name, spec.get("backend", cfg["backend"]), int("edit" in spec or "hook" in spec))
EOF
while read -r cond backend needs_analysis; do
  dep=()
  if [ "$needs_analysis" = 1 ] && [ -n "$ANALYSIS" ]; then dep=(--dependency=afterok:"$ANALYSIS"); fi
  if [ "$backend" = hf ]; then t=${HF_TIME:-72:00:00}; else t=${VLLM_TIME:-24:00:00}; fi
  id=$(sub ${dep[@]+"${dep[@]}"} --job-name="unloop-$cond" --time="$t" scripts/job.sbatch \
    unloop eval --config "$CONFIG" --condition "$cond")
  echo "$cond ($backend): job $id"
done
