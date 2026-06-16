#!/usr/bin/env bash
# Run the full prefill/decode power-characterisation pipeline with LIVE progress.
#
#   ./run.sh                # model_info -> measure(both) -> analyze(all)
#   ./run.sh measure        # just the sweeps
#
# Output streams to the terminal in real time AND is teed to results/run.log,
# so you can also watch from another shell with:   tail -f results/run.log
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # single V100
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=code
export PYTHONUNBUFFERED=1                                  # real-time stdout

mkdir -p results
LOG=results/run.log
: > "$LOG"

step="${1:-all}"
run() { echo ">>> $*" | tee -a "$LOG"; stdbuf -oL -eL "$@" 2>&1 | tee -a "$LOG"; }

case "$step" in
  info)    run python3 code/model_info.py ;;
  measure) run python3 code/measure.py --phase both ;;
  analyze) run python3 code/analyze.py --step all ;;
  all)
    run python3 code/model_info.py
    run python3 code/measure.py --phase both
    run python3 code/analyze.py --step all ;;
  *) echo "usage: $0 [all|info|measure|analyze]"; exit 1 ;;
esac
echo "done -> figures/  results/  (log: $LOG)" | tee -a "$LOG"
