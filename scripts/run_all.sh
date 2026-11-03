#!/usr/bin/env bash
# The whole study. Usage: scripts/run_all.sh [config] [condition ...]
#   SMOKE=1            2 problems per benchmark and 256 tokens per condition (a few minutes; run this first)
#   SKIP_ANALYSIS=1    skip reproduce, directions, sweep, explain (they write results/ used by the conditions)
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/full.toml}
shift || true
toml() { python -c "import sys, tomllib; c = tomllib.load(open(sys.argv[1], 'rb')); print($1)" "$CONFIG"; }
NAME=$(toml "c['name']")
CONDITIONS=${*:-$(toml "' '.join(c['conditions'])")}
SMOKE_FLAG=${SMOKE:+--smoke}
if [ -z "${SKIP_ANALYSIS:-}" ]; then
  unloop reproduce
  unloop directions
  unloop sweep
  unloop explain
fi
for c in $CONDITIONS; do
  unloop eval --config "$CONFIG" --condition "$c" $SMOKE_FLAG
done
if [ -z "${SMOKE:-}" ]; then
  unloop origin
fi
unloop table --run "$NAME${SMOKE:+-smoke}"
