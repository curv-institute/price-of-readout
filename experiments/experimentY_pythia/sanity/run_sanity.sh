#!/usr/bin/env bash
# Seed-variance sanity check on the three far-OOD points (the ones driving the
# recovery signal). Seed 0 is already in results/; this adds seeds 1 and 2.
set -u
cd "$(dirname "$0")"
S=sanity; mkdir -p $S
TXT=/vault/datasets/text
OFF=2000000
log(){ echo "[$(date +%H:%M:%S)] $*"; }
cell(){ # tag, seed, args...
  local tag="$1" seed="$2"; shift 2
  log "RUN $tag seed=$seed"
  uv run y1_disjoint.py "$@" --refit-offset $OFF --refit-seed "$seed" \
    --out "$S/${tag}_s${seed}.json" >> "$S/log_${tag}_s${seed}.txt" 2>&1 \
    && log "OK  $tag s$seed $(python3 -c "import json;d=json.load(open('$S/${tag}_s${seed}.json'));print('rec=%.4f'%(d['frozen']-d['refit']))")" \
    || log "FAIL $tag s$seed"
}
for s in 1 2; do
  cell yoruba  $s --mode shift --split q1 --textfile $TXT/ood_yo_wikipedia/data.txt
  cell swahili $s --mode shift --split sw
  cell welsh   $s --mode shift --split q1 --textfile $TXT/ood_cy_wikipedia/data.txt
done
log "SANITY DONE"
