#!/bin/bash
# On gene: serve each model with llama.cpp, run the three conditions, score.
# Usage: bash run.sh [model ...]   env: LIMIT=n CONDS="unaided probe agent"
set -uo pipefail
cd "$(dirname "$0")"
. ../.venv/bin/activate 2>/dev/null || . .venv/bin/activate
SERVER=${SERVER:-$HOME/src/llama.cpp/build/bin/llama-server}
declare -A GGUF=( [mimir]=DFM-Mimir-Q8_0.gguf [llama3b]=Llama-3.2-3B-Instruct-Q8_0.gguf [llama1b]=Llama-3.2-1B-Instruct-Q8_0.gguf [gemma4b]=google_gemma-3-4b-it-Q6_K.gguf [qwen3b]=Qwen2.5-3B-Instruct-Q8_0.gguf )
ORDER=(qwen3b llama1b llama3b gemma4b mimir)
[ $# -gt 0 ] && ORDER=("$@")
CONDS=${CONDS:-"unaided probe agent agent2"}
LIMIT=${LIMIT:-0}
mkdir -p logs
[ -s tasks.jsonl ] || python gen_tasks.py
for name in "${ORDER[@]}"; do
  f=$HOME/models/${GGUF[$name]}
  [ -s "$f" ] || { echo "skip $name (no gguf)"; continue; }
  echo "=== $(date +%H:%M) $name"
  "$SERVER" -m "$f" --port 8091 -ngl 99 -c 8192 -np 3 -ctk q8_0 -ctv q8_0 --jinja -fa on --log-disable > logs/server_$name.log 2>&1 &
  SPID=$!
  up=0
  for i in $(seq 1 90); do curl -s localhost:8091/health | grep -q '"ok"' && { up=1; break; }; sleep 2; done
  [ $up = 1 ] || { echo "server for $name did not come up"; kill $SPID 2>/dev/null; continue; }
  for cond in $CONDS; do
    python runner.py $cond --model $name --url http://localhost:8091/v1 \
      --out pred_${name}_${cond}.jsonl --limit $LIMIT --parallel 3 2>&1 | tail -2
  done
  kill $SPID; wait $SPID 2>/dev/null
done
python score.py pred_*.jsonl | tee SCORES.txt
echo "=== $(date +%H:%M) all done"
