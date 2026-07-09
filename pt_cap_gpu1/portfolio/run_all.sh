#!/bin/bash
# ONE-CLICK measurement: preflight -> power-cap sweep of ALL portfolio workloads (prefill+decode,
# methodology v3) -> fits & figures. Designed for an OFFLINE host (H200 or any NVIDIA GPU):
# no network access is attempted; models must already be in the local HF cache (see preflight).
#
#   [SUDO_PASS=...] ./run_all.sh [--gpu N] [--outdir DIR] [--smoke] [--ids a,b] [--caps auto|LIST]
#
#   --gpu N       which GPU to use (default 0)
#   --outdir DIR  output directory (default: data_<gpu-name-slug>, e.g. data_h200)
#   --smoke       quick machine validation: ONE workload (chat-phi3), 4 caps, ~4 min
#   --ids a,b     restrict to specific workload ids
#   --caps ...    'auto' (default: 10 points spanning the device's own [min,max]) or '200,300,...'
#
# Privilege for nvidia-smi -pl: run as root, OR passwordless sudo, OR export SUDO_PASS.
# Resume-aware: re-running skips workloads whose CSVs already exist (use FORCE=1 to redo).
# The sweep script itself restores the pre-run power cap in its `finally`.
set -u
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"   # arrays/PIPESTATUS need bash, not sh
cd "$(dirname "$0")"

# Running via `sudo ./run_all.sh` resets HOME to /root -> the HF model cache "disappears".
# Point HF_HOME back at the invoking user's cache unless the caller already set it.
if [ "$(id -u)" = "0" ] && [ -n "${SUDO_USER:-}" ] && [ -z "${HF_HOME:-}" ]; then
  U_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
  [ -d "$U_HOME/.cache/huggingface" ] && export HF_HOME="$U_HOME/.cache/huggingface" \
    && echo "(sudo detected: HF_HOME -> $HF_HOME)"
fi

GPU=0; OUTDIR=""; SMOKE=0; IDS=""; CAPS="auto"
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)    GPU="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --smoke)  SMOKE=1; shift;;
    --ids)    IDS="$2"; shift 2;;
    --caps)   CAPS="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="../../code"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # offline host: never touch the network

# default outdir: tag by GPU name (e.g. data_h200, data_tesla-v100-dgxs-32gb)
if [ -z "$OUTDIR" ]; then
  TAG=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null \
        | head -1 | tr 'A-Z ' 'a-z-' | sed 's/nvidia-//; s/-sxm.*//; s/-pcie.*//; s/[^a-z0-9-]//g')
  OUTDIR="data_${TAG:-unknown}"
fi
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run_all.log"
echo "== run_all: GPU$GPU -> $OUTDIR (log: $LOG) =="

echo "== 1/3 preflight ==" | tee -a "$LOG"
python3 preflight.py 2>&1 | tee -a "$LOG"
PF=${PIPESTATUS[0]}
if [ "$PF" -ne 0 ]; then
  echo "preflight FAILED -- fix the items above, then re-run." | tee -a "$LOG"; exit 1
fi

echo "== 2/3 power-cap sweep (methodology v3) ==" | tee -a "$LOG"
ARGS=(--outdir "$OUTDIR" --caps "$CAPS")
[ -n "$IDS" ] && ARGS+=(--ids "$IDS")
[ "${FORCE:-0}" = "1" ] && ARGS+=(--force)
if [ "$SMOKE" = "1" ]; then
  ARGS=(--outdir "$OUTDIR" --ids chat-phi3 --grid-n 4 --caps "$CAPS" --force)
  echo "(smoke mode: chat-phi3 only, 4 caps)" | tee -a "$LOG"
fi
python3 run_portfolio.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
[ "$RC" -ne 0 ] && { echo "sweep exited rc=$RC -- re-run to resume from where it stopped." | tee -a "$LOG"; exit "$RC"; }

echo "== 3/3 fits & figures ==" | tee -a "$LOG"
if python3 -c "import matplotlib" 2>/dev/null; then
  PORTFOLIO_DATA="$OUTDIR" python3 plot_decode_models.py 2>&1 | tee -a "$LOG"
  PORTFOLIO_DATA="$OUTDIR" python3 plot_portfolio.py    2>&1 | tee -a "$LOG"
else
  echo "matplotlib missing -- skipped plots; copy $OUTDIR/ back and plot elsewhere:" | tee -a "$LOG"
  echo "  PORTFOLIO_DATA=$OUTDIR python3 plot_decode_models.py && PORTFOLIO_DATA=$OUTDIR python3 plot_portfolio.py" | tee -a "$LOG"
fi

echo "== DONE. Everything to bring back is inside: $OUTDIR/ (CSVs, meta.json, figures, log) ==" | tee -a "$LOG"
nvidia-smi -i "$GPU" --query-gpu=power.limit --format=csv,noheader | sed 's/^/final power limit: /' | tee -a "$LOG"
