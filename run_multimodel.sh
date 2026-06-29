#!/usr/bin/env bash
# Cross-model validation of the analytical P(T) model.
# Per model: model_info (arch+roofline) + batch sweep (measure). DVFS only for a subset.
# Hardened + LIVE output: clears stale outputs before each model, copies ONLY freshly
# produced files, streams progress line-by-line in real time, and tees a full per-model log.
#
#   SUDO_PASS=... bash run_multimodel.sh           # do NOT prefix with sudo
set -uo pipefail
cd "$(dirname "$0")"

if [ "$(id -u)" -eq 0 ]; then
  echo "ERROR: do NOT run with sudo (root can't see ~/.local torch/numpy)."
  echo "       Run:  SUDO_PASS='<pw>' bash run_multimodel.sh"
  exit 1
fi
python3 -c "import numpy, torch" 2>/dev/null || { echo "ERROR: numpy/torch not importable as $(whoami)"; exit 1; }
[ -z "${SUDO_PASS:-}" ] && { echo "ERROR: SUDO_PASS required to lock the clock."; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=code PYTHONUNBUFFERED=1

# --- FORCE a fixed SM clock for ALL models (no DVFS, no auto-throttle while it holds) ---
# Note: the HARDWARE thermal slowdown (~85C) cannot be disabled in software; we pick a
# clock the V100 sustains so it never triggers -> the clock stays put across every model.
# Raise this only if you verify sm_clk_avg in the CSVs still equals it (else it throttled).
FIXED_CLOCK="${FIXED_CLOCK:-1005}"
sudo_nv() { printf '%s\n' "$SUDO_PASS" | sudo -S -p '' nvidia-smi -i 0 "$@"; }
echo ">>> locking SM clock to ${FIXED_CLOCK} MHz for the whole run"
sudo_nv -lgc "${FIXED_CLOCK}" 2>&1 | tail -1
trap 'echo ">>> unlocking SM clock"; sudo_nv -rgc 2>&1 | tail -1' EXIT
sleep 1
echo ">>> SM clock now: $(nvidia-smi -i 0 --query-gpu=clocks.sm,clocks_event_reasons.active --format=csv,noheader) (expect ~${FIXED_CLOCK})"

# stream(): run a command with LIVE line-by-line output + a full tee'd log.
#   $1 = log file ; $2 = grep regex (or "ALL" to show everything) ; rest = command
stream() {
  local log="$1" re="$2"; shift 2
  if [ "$re" = "ALL" ]; then
    stdbuf -oL "$@" 2>&1 | tee "$log"
  else
    stdbuf -oL "$@" 2>&1 | tee "$log" | stdbuf -oL grep -E --line-buffered "$re"
  fi
}

MEAS_RE='\] >|\] =|SWEEP|wrote |OOM|loaded |GPU |Error|Traceback|No module|RuntimeError|CUDA|assert'

MODELS=(
  "facebook/opt-1.3b"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "microsoft/Phi-3-mini-4k-instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

echo ">>> clearing stale mm_* outputs"
rm -f results/mm_*

for M in "${MODELS[@]}"; do
  slug="${M##*/}"
  echo "============================================================"
  echo "=== MODEL: $M  (slug=$slug)   $(date +%H:%M:%S) ==="
  echo "============================================================"
  export POWERCHAR_MODEL="$M"
  rm -f results/model_info.json results/prefill.csv results/decode.csv results/dvfs.csv

  echo ">>> [$slug] model_info (live)"
  stream "results/mm_${slug}_modelinfo.log" ALL python3 -u code/model_info.py
  if [ -f results/model_info.json ]; then cp results/model_info.json "results/mm_${slug}_info.json";
  else echo "!! [$slug] model_info FAILED (see results/mm_${slug}_modelinfo.log)"; fi

  echo ">>> [$slug] batch sweep prefill+decode (live)"
  stream "results/mm_${slug}_measure.log" "$MEAS_RE" python3 -u code/measure.py --phase both
  if [ -f results/prefill.csv ]; then cp results/prefill.csv "results/mm_${slug}_prefill.csv";
  else echo "!! [$slug] prefill FAILED (see results/mm_${slug}_measure.log)"; fi
  if [ -f results/decode.csv ]; then cp results/decode.csv "results/mm_${slug}_decode.csv";
  else echo "!! [$slug] decode FAILED (see results/mm_${slug}_measure.log)"; fi

  rm -rf "$HOME/.cache/huggingface/hub/models--${M//\//--}" 2>/dev/null
  echo ">>> [$slug] done; disk free: $(df -h / | awk 'NR==2{print $4}')"
done
echo "=== ALL MODELS DONE ===" ; ls -1 results/mm_*_info.json 2>/dev/null
