# Reproduction record

## Pythia-410m external replication (Discussion §`sec:pythia`) — reproduces within epsilon

**Post-hoc, NOT pre-registered** (unlike Experiments 1–3). See
`experiments/experimentY_pythia/POSTHOC_NOTE.md`. This record certifies that the
package regenerates the two v17.6 Discussion tables from its committed raw data.

**What was run.** The packaged `experiments/experimentY_pythia/verify.py` reads the
17 raw per-cell JSONs in `experimentY_pythia/results/` (frozen/refit bits/byte from
`EleutherAI/pythia-410m`, head-only refit, ctx 512, eval 2 MB / refit 8 MB / 400k
tok), recomputes `tab:pythia-quant` (quant decomposition vs the 16-bit head-refit
reference 0.861), `tab:pythia-ood` (refit recovery vs OOD distance, n=8), the
Pearson/Spearman correlations, and the Y5 activation-quant footnote, and asserts every
cell against `experimentY_pythia/EXPECTED_NUMBERS.md`. **Result: `VERIFY: PASS`** —
all cells match (Pearson 0.9287→0.93, Spearman 0.8571→0.86; quant 4/3/2-bit irrecoverable
0.166/1.390/1.557; OOD recoveries +0.003…+0.274; Y5 mean ≈41%). Telugu
(`shift_te.json`) is correctly excluded from the n=8 correlation as a multi-byte-script
bits/byte artifact.

**Epsilon.** The runner is single-run, deterministic-by-seed, but head-refit uses bf16
autocast, so end-to-end regeneration reproduces to ~2–3 decimals on the same GPU class,
not bit-exactly across hardware/library versions (same posture as Experiment 3 below).
Documented tolerance: **|Δ bits/byte| ≤ 0.01 per cell, |Δ correlation| ≤ 0.01**;
`%interface` ±1.5 pts; Y5 mean ±5 pts. `verify.py --smoke` re-runs two cells
end-to-end on a GPU and checks them within the same per-cell epsilon.

**Provenance.** Runners ship as the working-tree (`--textfile`) version of
`mlr-proof-program/experiments/expD_quantization/pilot/y1_pretrained_por.py` (+ the
`y5` activation-quant runner), with `$DATA_ROOT`/`--idfile` added for path portability
(behavior-preserving). Monorepo *results* commits: Y1 `b78770f…`, Y3 `91244b0…`,
Y3b `ceb19bf…`, Y5 `fc545de…`; paper v17.6 fold-in `e18054a…`. No pre-registration
and no OpenTimestamps attestation exist for this leg (post-hoc, by design).

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
