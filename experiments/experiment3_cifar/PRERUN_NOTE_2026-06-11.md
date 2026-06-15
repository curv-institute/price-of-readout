# Pre-run note — expB analysis script committed before the grid (2026-06-11)

**Agent:** run-B. **Scope:** records the once-only analysis script's commit, its
selftest result, and the training-grid launch — committed BEFORE any full-grid
arm-comparison / shift quantity exists. run-B computed NO gated quantity (no
arm comparison, no CE_Q, no Δ, no naive-vs-lawful number, no gate).

## Committed scripts (sha256 at this commit)

| file | sha256 | role |
|---|---|---|
| `analyze_masking.py` | `8484e5265ba9cc5ab9afc92ec459ef215f8d2547ff198328c769b8f08d140af8` | the §6 once-only analysis; `--selftest` runs §5.1 alone (no trained features), `--run` is the analysis agent's single unblinding |
| `train_run.py` | `b911c4e1f1bcc571091eebe9919d593730c78213695943d6b6ca037779fc60c5` | one training run per invocation (CNN9 §3 recipe; builds the cell's image sets via the pinned spurious_cifar.py §2 streams; dumps the four §4 feature sets + run marker) |
| `run_batch.sh` | `b6333e6dd9508b836f59a66fd3f52fbe34a968f47a762777fcfe5c9de3d7913b` | per-node sequential batch runner (idempotent skip; heartbeat; per-run log lines) |

Pins verified at authorship: `../lib/fourq.py` sha256
`8b66bc7d…abdd255a` (match), `./spurious_cifar.py` sha256
`ab8f6e61…0c9050ba` (match). The analysis script re-checks both at startup and
aborts on mismatch.

## §5.1 proxy-validation selftest result — FAIL at the committed n,D,K; a §5.9 trigger (escalated)

`uv run analyze_masking.py --selftest` was RUN (touches no trained features). At
the committed `A_log` settings (C=1.0, standardize=True, **5,000 fit / 5,000 eval,
D=256, K=10**) the planted-posterior recovery is:

| planted case | CE\* (MC) | recovered CEhat | abs err | τ=0.05 |
|---|---|---|---|---|
| planted-sufficient | 1.52 | 2.07 | +0.54 | FAIL |
| planted-sufficient-mild | 2.31 | 2.73 | +0.42 | FAIL |
| planted-insufficient-control | 3.32 | 3.72 | +0.39 | FAIL |

This is **not green**, and it is not a script defect. The excess is the intrinsic
finite-sample held-out-CE excess of a K×D multinomial-logistic MLE,
`≈ K_readout/(2 n ln2) = 2570/(2·5000·0.693) ≈ 0.37 bits`, confirmed by a
fixed-n dimension sweep (D=256→+0.54, 64→+0.08, 16→+0.05, 8→+0.01) and corroborated
upstream: `lib/test_fourq.py` recovers planted CE\* to <0.001 bits — but only at
large n/D (its anchors use n=40,000, dz≤6). At this experiment's n/D≈20, 0.05-bit
**absolute** recovery is unreachable.

The prereg §5.1 explicitly anticipated this: *"the regularization bias of C=1.0 at
n=5,000, D=256 is expected well inside this — **if not, that is exactly a §5.9
proxy-quality amendment trigger.**"* So a FAIL here is a **pre-registered, in-protocol
outcome**, not a massaged result. Per the campaign escalation rule (pre-registration
content / threshold procedure = mandatory escalation), a decision request is filed:

> `queue/decisions/pending/runB-expB-s51-selftest-excess.md`

with two clean options (take the FAIL → raw-CE descriptive per §9; or a §5.9 dated
amendment redefining the §5.1 PASS as recovery of CE\* **differences** — the
estimand the gates actually use, where the common bias cancels). **run-B did not
choose** — that is fable's call. The script handles both outcomes mechanically: on
§5.1 FAIL the serializer downgrades every `CEhat` row to `raw_heldout_CE` (the
masking-decomposition LANGUAGE is withheld), while the difference-based gates
(G-B2/G-B4 pure frozen-transfer; G-B1/G-B3 CE-gap criteria) and C-B5 remain numerically
computable. **No gate's numeric threshold depends on the §5.1 ruling; only the CE\*
license wording does.**

## What was verified pre-commit (no trained-feature touch)
- Selftest runs and reports the FAIL above (planted posteriors only).
- The construction reproduces the pilot: naive ρ=0.9 train P(s==y)=0.903, the
  ρ=0.9 decorrelate/anticorrelate coincidence (≈0.10 each), lawful≡naive-at-ρ=0.1.
- The training script's output schema (probe naive_rho0.7_seed0): four 5000×256
  float32 feature sets, 0 non-finite, n_params=1,739,210, feat_dim=256.
- The battery rows are declaration-complete and pass `fourq.check_quantity_declarations`
  / serialize via the RULE-enforcing writers; seed-mean/δ_cell helpers correct
  (δ_cell = max(2·SE, 0.10), floor binds at the pilot SE).

## Grid status at commit
80-run grid launched (2026-06-11 ~22:15Z) as two campaign-queue jobs:
`expB-grid-dgx01` (40 naive) + `expB-grid-dgx02` (40 lawful), sequential per node
(GPU compute-bound; 2-way concurrency probed → ~2× per-job slowdown, zero net gain;
memory never the limit). ETA ~2.5–3.5 h. Setup manifest + analysis-agent
verification checklist: `queue/done/runB-setup.json`.

## POSTSCRIPT 2026-06-11 — escalation answered; §5.9 amendment executed (grid untouched)

The §5.1 FAIL escalation above was **answered**: ruling
`queue/decisions/answered/runB-expB-s51-selftest-excess.md` (fable-5, Option 2) invoked the
§5.9 amendment, redefining the §5.1 PASS as CE\*-**DIFFERENCE** recovery (the estimand the gates
consume; the ~0.371-bit common-mode MLE excess cancels in differences). The amendment is now
committed to `PREREGISTRATION.md` **§5.9.A** (dated 2026-06-11, the experiment's ONE permitted
amendment — none remain), and `analyze_masking.py`'s §5.1 selftest is rewritten to the
difference-recovery criterion (the absolute FAIL table above is preserved verbatim in the
selftest record, ruling condition 4). The amended selftest is **GREEN**:

| planted pair | planted ΔCE\* | recovered ΔCEhat | |err| | τ = 0.05 |
|---|---|---|---|---|
| gate-relevant (~0.5 bit; λ 0.30 vs 0.10) | +0.6305 | +0.6249 | 0.0056 | **PASS** |
| null calibration (~0 bit; λ 0.25 disjoint banks) | −0.0005 | +0.0002 | 0.0007 | **PASS** |

Licensing split (ruling condition 2): differences (Δ_info, Δ_transfer, arm CEhat_ID gaps)
licensed as CE\* differences; absolute CEhat downgraded to `raw_heldout_CE` with the disclosed
bias ≈ 0.371 bits (`K_readout/(2n ln2)`). The amendment touched **only** the selftest /
license framing — **no running grid job, no trained feature, and no gated quantity** was touched.
