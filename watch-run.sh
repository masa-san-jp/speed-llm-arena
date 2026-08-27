#!/usr/bin/env bash
# Record what the arena and both ollama servers are doing, once a minute.
# The 2026-08-27 gx10 run went silent for 30 minutes and left nothing to read,
# because its only output was a results file written after the last match.
set -u

LOG=${1:-$HOME/.arena/arena.log}
OUT=${2:-$HOME/.arena/heartbeat.log}
MAX_MINUTES=${3:-240}

for ((i = 0; i < MAX_MINUTES; i++)); do
  ts=$(date -Is)
  pid=$(pgrep -f "^python3 -u speed_arena.py" | head -1)
  lines=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  a=$(OLLAMA_HOST=127.0.0.1:11434 timeout 15 ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | paste -sd, -)
  b=$(OLLAMA_HOST=127.0.0.1:11435 timeout 15 ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | paste -sd, -)
  free_gb=$(free -g | awk '/^Mem:/{print $7}')
  echo "$ts pid=${pid:-DEAD} log_lines=$lines 11434=[${a:--}] 11435=[${b:--}] avail_gb=$free_gb" >> "$OUT"
  [ -z "$pid" ] && { echo "$ts arena process gone" >> "$OUT"; break; }
  sleep 60
done
