#!/bin/bash
# Auto-run the portfolio power-cap sweep on the first GPU that becomes cleanly idle.
# Waits for a GPU with util==0 AND mem<1500 MiB (i.e. no other user's job) sustained 60s,
# then runs run_portfolio.py there (resume-aware: already-done workloads are skipped).
# Retries across free GPUs until all 8 workloads have data, or a 6h deadline passes.
set -u
cd "$(dirname "$0")"
: "${SUDO_PASS:?set SUDO_PASS in the environment before launching (never hardcode it)}"
export SUDO_PASS
export PYTHONPATH=../../code
LOG=auto_run.log
ALL_IDS="chat-phi3 rag-phi3 code-phi3 longform-phi3 summarize-qwen7b translate-qwen3b fastchat-qwen15b classify-qwen7b"

log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

done_count(){
  local c=0 id f
  for id in $ALL_IDS; do
    f="data/${id}_decode.csv"
    [ -f "$f" ] && [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -ge 2 ] && c=$((c+1))
  done
  echo $c
}

pick_free_gpu(){  # first GPU idle enough to be nobody-else's; empty if none
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"");
                  if (($2+0)==0 && ($3+0)<1500) {print $1; exit}}'
}

log "watcher start; $(done_count)/8 workloads already have data"
DEADLINE=$(( $(date +%s) + 6*3600 ))
while [ "$(done_count)" -lt 8 ]; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "TIMEOUT (6h) waiting for a free GPU; giving up"; exit 2; }
  g1=$(pick_free_gpu)
  if [ -z "$g1" ]; then
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | awk -F',' '{gsub(/ /,""); printf "gpu%s %s%% %sMiB  ",$1,$2,$3}' >> "$LOG"; echo >> "$LOG"
    sleep 45; continue
  fi
  log "GPU $g1 looks idle; confirming sustained-idle for 60s..."
  sleep 60
  g2=$(pick_free_gpu)
  if [ -z "$g2" ] || [ "$g1" != "$g2" ]; then log "GPU $g1 no longer clean (now '$g2'); re-scanning"; continue; fi
  log "GPU $g1 confirmed free -> launching portfolio ($(done_count)/8 done)"
  CUDA_VISIBLE_DEVICES="$g1" python3 run_portfolio.py >> "$LOG" 2>&1
  rc=$?
  log "run_portfolio exited rc=$rc; now $(done_count)/8 workloads done"
done
log "ALL 8 WORKLOADS DONE -> ready to plot"
