#!/usr/bin/env bash
# run_batch.sh — run-B training-grid batch runner for one node.
#
# Runs a list of (arm rho seed) cells through train_run.py at a chosen
# concurrency, emitting per-run progress lines to stdout (the queue log) and
# touching a heartbeat file every run. Each train_run.py invocation is one cell;
# it self-re-execs the cuDNN LD_LIBRARY_PATH shim and writes its own feature +
# run-marker files to /vault/datasets/features/expB/ (shared across both nodes).
#
# Usage:
#   run_batch.sh <node-tag> <concurrency> <cells-file> <heartbeat-file>
# cells-file: one "arm rho seed" per line.
#
# Idempotent: a cell whose run-marker JSON already exists is SKIPPED (so a
# re-launch after a partial run resumes cleanly without retraining).
set -u
NODE="${1:?node-tag}"
CONC="${2:?concurrency}"
CELLS="${3:?cells-file}"
HB="${4:?heartbeat-file}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${OUTDIR:-/vault/datasets/features/expB}"
mkdir -p "$OUTDIR" "$(dirname "$HB")"

echo "[$NODE] run_batch start $(date -Is) concurrency=$CONC cells=$(wc -l < "$CELLS") outdir=$OUTDIR"
touch "$HB"

# background heartbeat: touch every 60s while this batch runs
( while true; do touch "$HB"; sleep 60; done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null' EXIT

run_one() {
  local arm="$1" rho="$2" seed="$3"
  local tag="${arm}_rho${rho}_seed${seed}"
  if [ -f "$OUTDIR/run_${tag}.json" ]; then
    echo "[$NODE] SKIP $tag (run-marker present)"; return 0
  fi
  local t0 t1
  t0=$(date +%s)
  echo "[$NODE] START $tag $(date -Is)"
  ( cd "$HERE" && LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-} \
      uv run train_run.py --arm "$arm" --rho "$rho" --seed "$seed" \
      --epochs 30 --outdir "$OUTDIR" ) >> "$OUTDIR/log_${tag}.txt" 2>&1
  local rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ] && [ -f "$OUTDIR/run_${tag}.json" ]; then
    local acc
    acc=$(grep -o '"std_test_acc": [0-9.]*' "$OUTDIR/run_${tag}.json" | head -1)
    echo "[$NODE] DONE  $tag rc=0 ${acc} wall=$((t1-t0))s"
  else
    echo "[$NODE] FAIL  $tag rc=$rc wall=$((t1-t0))s (see log_${tag}.txt)"
  fi
  touch "$HB"
}

# Concurrency control: keep at most $CONC run_one workers in flight. We track
# worker PIDs EXPLICITLY (not `jobs -rp`, which also counts the heartbeat
# background job and would deadlock at concurrency 1).
workers=()
prune() {  # drop finished PIDs from the workers array
  local live=()
  for p in "${workers[@]}"; do kill -0 "$p" 2>/dev/null && live+=("$p"); done
  workers=("${live[@]}")
}
while read -r arm rho seed; do
  [ -z "${arm:-}" ] && continue
  run_one "$arm" "$rho" "$seed" &
  workers+=($!)
  prune
  while [ "${#workers[@]}" -ge "$CONC" ]; do sleep 2; prune; done
done < "$CELLS"
for p in "${workers[@]}"; do wait "$p" 2>/dev/null; done

ndone=$(ls "$OUTDIR"/run_*.json 2>/dev/null | wc -l)
echo "[$NODE] run_batch END $(date -Is) | run-markers now present: $ndone"
