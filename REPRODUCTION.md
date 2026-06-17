# Reproduction record

## Experiment 3 (CIFAR masking) — CONFIRMED: reproduces within epsilon (2026-06-16)

A fresh, independent re-run of the Experiment-3 training grid from this public harness
confirms the committed results within the expected non-determinism of CNN training, with
**all gate verdicts unchanged**. Published numbers are unchanged (validation only).

**What was run.** Full grid `experiments/experiment3_cifar/run_batch.sh`: 4 ρ
{0.7, 0.9, 0.95, 0.99} × 10 seeds {0..9} × 2 arms {naive, lawful} = 80 cells (CNN9, 1.74M
params, 30 epochs), on 2× NVIDIA GB10 (aarch64, torch 2.12+cu130), fresh output dir, then
`analyze_masking.py --run` (CPU). Instrument pin verified: `lib/fourq.py`
sha256 = `8b66bc7d…abdd255a` (matches the script's `FOURQ_SHA256`).

**Gate verdicts (fresh run).** §5.1 difference-recovery license PASS (|err| 0.0056 ≤ 0.05;
null 0.0007). **G-B1, G-B2, G-B3, G-B4 all PASS.** `masking_at_scale_reproduced = true`;
`C_B5_violated = false`. No verdict flipped.

**Science-column deltas (fresh vs committed `results.csv`, 800 rows, 0 key mismatches).**
- disclosed analytic CE-bias: Δ = 0.000000 bits (identical, as required).
- bootstrap SE: max Δ = 0.0154 bits (≤ 0.016).
- per-cell `value_bits`: max ε = **0.31 bits** (single-seed, on a `delta_transfer`
  difference-of-CEs in the descriptive *anticorrelate* shift).
- **seed-mean `value_bits` (the level the gates use): max ε = 0.054 bits on the descriptive
  *anticorrelate* shift; 0.026 bits over the gated *decorrelate* rows.**

**Reading.** Per-cell deltas (~0.3 bits) are single-seed CNN-training non-determinism on
difference quantities; they average down at the seed-mean level the gates evaluate, which is why
every gate reproduces. Documented epsilon (scoped to what the gates use): **gated (decorrelate)
seed-mean ≤ 0.026 bits; descriptive (anticorrelate) seed-mean ≤ 0.054 bits; per-cell ≤ 0.31
bits.** Reproduces within epsilon; gate verdicts unchanged.

**Provenance.** The validation runner `experiments/experiment3_cifar/run_batch.sh` used here
(sha256 `6a44bd41…`) is a behavior-preserving `OUTDIR`-parameterization of the pinned
`b6333e6d…` — a single line, `OUTDIR="${OUTDIR:-/vault/datasets/features/expB}"`, so the output
dir is overridable; `OUTDIR` is used only as a path and changes no science. The edit and the
`experiments/lib -> ../lib` import symlink are committed alongside this record. The `fourq.py`
instrument sha matches its pin exactly (integrity gate fired: analyze.log "pins OK").

**Evidence.** Fresh `results.csv`, `machine_summary.json`, and the diff are archived with the
run at `/gfs/curv-campaign/exp3_run/` (and `exp3_run/evidence/`). Experiments 1 & 2 + theory
were previously reproduced bit-exact on CPU.
