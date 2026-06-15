# Results: Experiment B — Masking at Scale (CIFAR-10, Constructed Spurious Tint)

**Date:** 2026-06-12
**Experiment id:** `expB-masking-real-v1`
**Pre-registration:** `PREREGISTRATION.md`, committed before any naive arm was
trained and before any shift set was built for measurement:

- `d26850e` (2026-06-11 20:39 UTC) — the pre-registration itself;
- `c13792a` (2026-06-11 22:23 UTC) — the once-only analysis script
  `analyze_masking.py` + training scripts, committed pre-run (§6); the original
  §5.1 selftest was run at this commit and **FAILED** (escalated, see below);
- `dc7b1fd` (2026-06-11) — the **one permitted §5.9 amendment (§5.9.A)**:
  §5.1 PASS redefined as CE\*-**difference** recovery per ruling
  `queue/decisions/answered/runB-expB-s51-selftest-excess.md`, made strictly
  pre-unblinding. **Amendment budget after it: 0; no further amendment was made.**

**Raw data:** `results.csv` / `results_rows.json` (800 declaration-complete rows:
4 ρ × 10 seeds × 2 arms × 2 shifts × 5 measurements), `results_summary.json`
(cell means/SEs/CIs, gates, noise floors, sanity accuracies),
`machine_summary.json`. Feature archives: `/vault/datasets/features/expB/`
(80 × `features_*.npz` + run markers; not committed). Pins re-verified at run
start: `fourq.py` sha256 `8b66bc7d…`, `spurious_cifar.py` sha256 `ab8f6e61…` (both match §6/§2).

**Run provenance.** Training grid: **80/80 runs complete, all rc=0** —
40 naive cells on dgx01 (`expB-grid-dgx01`, run_batch END 02:50:02 UTC
2026-06-12, 80/80 markers verified), 40 lawful cells on dgx02
(`expB-grid-dgx02`, END 02:19:39 UTC; plus tail takeover job
`expB-takeover-dgx02` picking up the naive remainder). Disclosure: both grid
jobs' wrapper finalizers failed after successful completion; the queue records
were closed by the orchestrator on log evidence (infra bookkeeping only — every
cell's own run marker reports rc=0). The **once-only unblinding analysis**
(`analyze_masking.py --run`, job `runB-analysis`, dgx02) ran exactly once:
selftest pass logged 03:03:53 UTC, unblinding start **03:04:12 UTC**, end
**07:12:11 UTC 2026-06-12, rc=0** (`queue/logs/runB-analysis.log`). The C-B5
coincidence check was evaluated before gate publication and did not HALT.

**Headline:** **G-B1 PASS, G-B2 PASS, G-B3 PASS, G-B4 PASS; C-B5 not violated;
§5.1 (amended) difference-recovery license PASS.** Verdict per §5.3 composition:
**masking-at-scale REPRODUCED** (`machine_summary.json:
masking_at_scale_reproduced = true`). The Track-3 masking signature — an
ID-invisible (here ID-*advantaged*) spurious-cue-reliant representation whose
frozen interface collapses under a coupling-breaking shift, with the collapse
increasingly interface-borne at high coupling — reproduces on real images with
learned CNN representations under this construction.

---

## 0. The §5.1 license (as amended §5.9.A) and the licensing split

The original §5.1 criterion (absolute CE\* recovery within τ = 0.05 bits at
n = 5,000, D = 256, K = 10) **FAILED** pre-unblinding, exactly as the prereg's
own escape clause anticipated; the FAIL table is preserved verbatim in
`results_summary.json → selftest_5_1.absolute_recovery_history_PRESERVED`
(planted-sufficient +0.54, mild +0.42, insufficient-control +0.39 bits abs err
— the intrinsic K_readout/(2n ln2) ≈ 0.371-bit finite-sample MLE excess). The
§5.9.A amendment redefined PASS as CE\*-**difference** recovery — the estimand
every gate actually consumes. As-run amended selftest (re-run inside the
once-only analysis, matched n = 5,000/5,000, D = 256, K = 10, C = 1.0,
standardize = True, 6 draws per member):

| planted pair | planted ΔCE\* | recovered ΔCEhat | abs err | τ = 0.05 | verdict |
|---|---|---|---|---|---|
| gate-relevant (~0.5 bit; λ = 0.30 vs 0.10) | +0.6305 | +0.6249 | 0.0056 | ≤ 0.05 | **PASS** |
| null calibration (~0 bit; λ = 0.25, disjoint seed banks) | −0.0005 | +0.0002 | 0.0007 | ≤ 0.05 | **PASS** |

Member excesses are matched (gate pair +0.5918 vs +0.5974; null pair +0.6201
vs +0.6194 bits), confirming the common-mode cancellation mechanism. All
selftest fits converged; 0 unseen labels.
**§5.1 selftest (difference-recovery): PASS.** License ref:
`"expB-masking-real-v1 §5.1 planted-posterior PASS"`.

**Licensing split (binding throughout this document):**

- **Differences** — Δ_info, Δ_transfer, and the naive−lawful CEhat_ID gaps —
  carry the licensed CE\*-difference framing (the ~0.371-bit common-mode excess
  cancels to first order).
- **Absolute CE values** — every CEhat_ID and CEhat_Q_refit below — are
  `raw_heldout_CE`: **no CE\*-point license**, with the estimated bias disclosed:
  **≈ 0.3708 bits** (`K_readout/(2n ln2) = 2570/(2·5000·ln 2) = 0.3708`),
  common-mode across cells.
- CE_Q(r_P) is definitionally exact (quantity 4 of §2-bis), `not_an_entropy=True`,
  never an entropy, no license needed.

## 1. Measured cell means (10 seeds; decorrelate = gated PRIMARY shift)

All values bits/sample, seed-means over all 10 seeds (no seed omitted). ± is
cross-seed SE (`std_{ddof=1}/√10`); CIs are 2000-resample bootstrap CIs of the
seed-mean. CEhat columns are `raw_heldout_CE` (bias ≈ 0.3708 bits disclosed,
common-mode); CE_Q(r_P) is quantity (4), never an entropy.

**NAIVE arm, decorrelate (gated):**

| ρ | CEhat_ID (raw) | CE_Q(r_P) frozen | CEhat_Q-refit (raw) | Δ_info | Δ_transfer |
|---|---|---|---|---|---|
| 0.70 | 0.3154 ± 0.0035 | 0.6685 ± 0.0096 | 0.5666 ± 0.0051 | 0.2511 ± 0.0050 | 0.1020 ± 0.0066 |
| 0.90 | 0.2266 ± 0.0052 | 1.2874 ± 0.0161 | 0.7350 ± 0.0046 | 0.5084 ± 0.0084 | 0.5524 ± 0.0196 |
| 0.95 | 0.1807 ± 0.0041 | 1.9646 ± 0.0302 | 0.8661 ± 0.0053 | 0.6854 ± 0.0067 | 1.0985 ± 0.0314 |
| 0.99 | 0.1172 ± 0.0036 | **4.7737 ± 0.0559** | 1.2134 ± 0.0078 | 1.0962 ± 0.0079 | **3.5603 ± 0.0588** |

**NAIVE arm, anticorrelate (descriptive, no gate — reported in full):**

| ρ | CE_Q(r_P) frozen | CEhat_Q-refit (raw) | Δ_info | Δ_transfer |
|---|---|---|---|---|
| 0.70 | 0.5506 ± 0.0085 | 0.4979 ± 0.0043 | 0.1825 ± 0.0032 | 0.0527 ± 0.0075 |
| 0.90 | 1.2860 ± 0.0188 | 0.7244 ± 0.0051 | 0.4977 ± 0.0080 | 0.5616 ± 0.0175 |
| 0.95 | 2.0547 ± 0.0381 | 0.8736 ± 0.0057 | 0.6929 ± 0.0062 | 1.1811 ± 0.0376 |
| 0.99 | **5.2748 ± 0.0555** | 1.2040 ± 0.0060 | 1.0868 ± 0.0066 | **4.0708 ± 0.0540** |

**LAWFUL arm (ρ_eff = 0.1 in every cell), decorrelate:**

| ρ-cell | CEhat_ID (raw) | CE_Q(r_P) frozen | CEhat_Q-refit (raw) | Δ_info | Δ_transfer |
|---|---|---|---|---|---|
| 0.70 | 0.4172 ± 0.0054 | 0.4208 ± 0.0053 | 0.4378 ± 0.0058 | +0.0206 | −0.0170 |
| 0.90 | 0.4168 ± 0.0036 | 0.4186 ± 0.0028 | 0.4230 ± 0.0047 | +0.0062 | −0.0044 |
| 0.95 | 0.4102 ± 0.0045 | 0.4117 ± 0.0045 | 0.4290 ± 0.0054 | +0.0188 | −0.0173 |
| 0.99 | 0.4190 ± 0.0036 | 0.4212 ± 0.0036 | 0.4367 ± 0.0023 | +0.0176 | −0.0154 |

(Lawful anticorrelate values are in the G-B3 table below; lawful frozen CE_Q
under anticorrelate spans 0.4183–0.4280 across the four cells — flat there too.)

The bolded naive frozen-transfer values at ρ = 0.99 (4.7737 / 5.2748 bits)
**exceed log₂10 = 3.3219 bits** — the §6.5 tell. They are CE_Q(r_P), quantity
(4), a property of the frozen interface; no entropy bound applies and none is
implied (the serializer enforces `not_an_entropy=True` on every such row).

**Pipeline sanity (no gate):** standard untinted-CIFAR test accuracy — lawful
0.9242–0.9247 mean across cells (pilot smoke: 0.9240); naive **0.8910 / 0.8203
/ 0.7476 / 0.5583** at ρ = 0.7/0.9/0.95/0.99. Empirical train coupling
matched ρ per marker (e.g. naive ρ=0.7 seed 0: P(s=y) = 0.7017; lawful 0.10).
Non-finite feature counts: 0 in all 80 cells. Unseen-label counts on every Q
evaluation: 0 (expected 0, §4).

## 2. Pre-registered gates: measured vs threshold

δ_cell = max(2×SE_seed, 0.10) per compared cell (§5.2 formula, committed). The
0.10-bit materiality floor bound in nearly every cell; the largest realized
δ_cell anywhere in a gated comparison was **0.1177 bits** (naive ρ=0.99
Δ_transfer) — see §7 for the §5.5 evaluation.

### G-B1 — masking-ID premise (naive−lawful CEhat_ID gap ≤ +δ at every ρ) — licensed CE\*-difference

| ρ | naive CEhat_ID | lawful CEhat_ID | gap | band [−lawful, +δ] | verdict |
|---|---|---|---|---|---|
| 0.70 | 0.3154 | 0.4172 | **−0.1017** | [−0.4172, +0.10] | **PASS** |
| 0.90 | 0.2266 | 0.4168 | **−0.1901** | [−0.4168, +0.10] | **PASS** |
| 0.95 | 0.1807 | 0.4102 | **−0.2295** | [−0.4102, +0.10] | **PASS** |
| 0.99 | 0.1172 | 0.4190 | **−0.3018** | [−0.4190, +0.10] | **PASS** |

**G-B1: PASS.** The naive arm is not merely ID-indistinguishable — it is ID
**better** than lawful at every ρ, by a margin that grows with the coupling.
This is the pre-declared amplified-masking direction (§1.1, §5.3): at ρ = 0.99
the prereg derived an expected naive-better gap of ≈ 0.27 bits from the
cue-only ID anchor (0.1125) against the *pilot* lawful floor (0.3787); against
the *realized* lawful floor (0.4190) the same construction arithmetic gives
0.4190 − 0.1125 ≈ 0.31, and the measured gap is **−0.3018** — within 0.005 bits
of the construction's own expectation. Insufficiency is invisible (indeed
rewarded) in-distribution.

### G-B2 — naive transfer brittleness (Δ_transfer ≥ 0.5 bits at each ρ ∈ {0.9, 0.95, 0.99}, decorrelate)

| ρ | naive Δ_transfer | SE | 95% CI | δ_cell | threshold | verdict |
|---|---|---|---|---|---|---|
| 0.90 | **0.5524** | 0.0196 | [0.5176, 0.5901] | 0.10 | ≥ 0.5 | **PASS** |
| 0.95 | **1.0985** | 0.0314 | [1.0359, 1.1560] | 0.10 | ≥ 0.5 | **PASS** |
| 0.99 | **3.5603** | 0.0588 | [3.4485, 3.6692] | 0.1177 | ≥ 0.5 | **PASS** |

**G-B2: PASS** at every gated ρ; every δ_cell ≤ 0.25 as G-B2 additionally
requires (§5.5). Margin note (honest): at ρ = 0.9 the pass is real but not
enormous — the seed-mean clears the 0.5-bit minimum by 0.052 bits and the CI
lower edge by 0.018 bits. At ρ = 0.95 and 0.99 the margins are 0.60 and 3.06
bits — decisive. (ρ = 0.7, reported not gated: Δ_transfer = 0.1020
[0.0906, 0.1146] — at the materiality floor, covered by G-B4.)

### G-B3 — lawful arm flat (|Δ_info| ≤ δ and |Δ_transfer| ≤ δ at every ρ, decorrelate; anticorrelate beside)

| ρ-cell | Δ_info (dec) | Δ_transfer (dec) | δ | Δ_info (anti) | Δ_transfer (anti) | verdict |
|---|---|---|---|---|---|---|
| 0.70 | +0.0206 | −0.0170 | 0.10 | +0.0138 | −0.0036 | **PASS** |
| 0.90 | +0.0062 | −0.0044 | 0.10 | +0.0012 | +0.0056 | **PASS** |
| 0.95 | +0.0188 | −0.0173 | 0.10 | +0.0127 | −0.0047 | **PASS** |
| 0.99 | +0.0176 | −0.0154 | 0.10 | +0.0127 | −0.0038 | **PASS** |

**G-B3: PASS.** The largest lawful deviation anywhere, either shift, is
0.0206 bits — 4.9× inside δ. Sufficiency-by-construction yielded
shift-stability: the lawful interface and the lawful representation are both
flat under both shifts at every coupling.

### G-B4 — masking-curve monotonicity (naive Δ_transfer non-decreasing in ρ up to δ, decorrelate)

| pair | Δ_transfer(ρ_lo) | Δ_transfer(ρ_hi) | criterion | verdict |
|---|---|---|---|---|
| 0.70 → 0.90 | 0.1020 | 0.5524 | 0.5524 ≥ 0.1020 − 0.10 | **PASS** |
| 0.90 → 0.95 | 0.5524 | 1.0985 | 1.0985 ≥ 0.5524 − 0.10 | **PASS** |
| 0.95 → 0.99 | 1.0985 | 3.5603 | 3.5603 ≥ 1.0985 − 0.1177 | **PASS** |

**G-B4: PASS** — strictly increasing at every adjacent pair (no pair needed the
δ allowance), with super-linear growth toward ρ = 0.99.

### C-B5 — ρ = 0.9 dec/anti coincidence (validity check, not a gate, not a finding)

The two ρ = 0.9 shift sets are independent draws from the identical
distribution; per arm the frozen CE_Q(r_P) must agree within δ:

| arm | CE_Q(r_P) dec | CE_Q(r_P) anti | abs diff | δ | consistent |
|---|---|---|---|---|---|
| naive | 1.2874 | 1.2860 | **0.0014** | 0.10 | yes |
| lawful | 0.4186 | 0.4236 | **0.0050** | 0.10 | yes |

**C-B5: NOT VIOLATED** (agreement at the millibits level, 70× and 20× inside
δ). The analysis did not HALT; no pipeline-repair note was needed.

## 3. The masking curve vs the cue-only anchors (§1.1 committed framing)

Naive arm, decorrelate. "ID excess" = CE_Q(r_P) − CEhat_ID (derived column:
difference of the two transcribed seed-means; identically Δ_info + Δ_transfer).
Anchors are the committed §1.1 construction math, not measurements.

| ρ | naive ID excess | = Δ_info | + Δ_transfer | cue-only elev[dec] (anchor) | f(ρ) = Δ_transfer/elev | f 95% CI* |
|---|---|---|---|---|---|---|
| 0.70 | 0.3531 | 0.2511 | 0.1020 | 2.6354 | 0.0387† | [0.0344, 0.0435] |
| 0.90 | 1.0608 | 0.5084 | 0.5524 | 5.0719 | **0.1089** | [0.1020, 0.1163] |
| 0.95 | 1.7839 | 0.6854 | 1.0985 | 6.3052 | **0.1742** | [0.1643, 0.1834] |
| 0.99 | 4.6565 | 1.0962 | 3.5603 | 8.7214 | **0.4082** | [0.3954, 0.4207] |

\* f point values at ρ ≥ 0.9 are from `results_summary.json`
(`gates.G-B2.per_rho[].frac_of_envelope`); the f CIs are the Δ_transfer
bootstrap CIs divided by the committed anchor constant (a fixed rescaling,
derived for presentation). † ρ = 0.7 is outside the banded-prediction domain;
its f is the same rescaling of the reported Δ_transfer (descriptive).

- **Banded prediction (committed §1.1): 0.1 ≤ f ≤ 1 at ρ ≥ 0.9 — HELD at all
  three gated ρ**, with no exceedance of the calibrated-cue-reader upper
  envelope anywhere (largest f = 0.41; no miscalibration overshoot to discuss).
  At ρ = 0.9 the entire f CI [0.1020, 0.1163] sits just above the 0.1 lower
  edge — the trained system realises almost exactly the band's minimum at the
  weakest gated coupling, and an increasing fraction (0.17, then 0.41) of the
  cue-readable ceiling as the coupling tightens.
- Measured frozen CE_Q vs the cue-only CE_Q[dec] anchors (4.4677 / 5.8579 /
  6.7501 / 8.8339): measured 0.6685 / 1.2874 / 1.9646 / 4.7737 — always far
  below the pure-cue-reader value, as the committed framing predicts for a CNN
  that divides reliance between image content and cue.
- **Anticorrelate (descriptive):** measured Δ_transfer 0.0527 / 0.5616 / 1.1811
  / 4.0708. The anchor asymmetry is qualitatively reproduced in **both
  directions**: anti < dec at ρ = 0.7 (anchors 1.7569 < 2.6354; measured 0.0527
  < 0.1020), anti ≈ dec at ρ = 0.9 (anchors coincide; measured 0.5616 vs
  0.5524), anti > dec at ρ = 0.95 (measured 1.1811 vs 1.0985) and ρ = 0.99
  (measured 4.0708 vs 3.5603, frozen CE 5.2748 vs 4.7737).

### Pre-registered descriptive separation: Δ_transfer ≫ Δ_info (prediction: ratio ≥ 2 at ρ ≥ 0.9)

| ρ | naive Δ_transfer | naive Δ_info | ratio | prediction ≥ 2 |
|---|---|---|---|---|
| 0.90 | 0.5524 | 0.5084 | **1.09** | **NOT MET** (short by 0.91) |
| 0.95 | 1.0985 | 0.6854 | **1.60** | **NOT MET** (short by 0.40) |
| 0.99 | 3.5603 | 1.0962 | **3.25** | met |

**Reported in full, no gate attaches.** Δ_transfer exceeds Δ_info at every
gated ρ (the predicted direction), but the ≥ 2 dominance appears only at
ρ = 0.99; at ρ = 0.9 the collapse splits almost evenly between interface
brittleness and genuine information-accessibility loss. See §8 (discussion):
this is the largest deviation from prediction in the experiment.

## 4. Comparison to predictions — complete accounting (held / deviated)

1. **§1 prediction 1 / G-B1 (masking ID):** HELD, in the amplified direction —
   naive better-ID at every ρ, not only at 0.99. The ρ = 0.99 gap (−0.3018)
   matches the construction's own anchor arithmetic to < 0.005 bits.
2. **§1 prediction 2 / G-B2 (naive brittleness, refit recovers most):** HELD as
   gated. The refit does recover most of the frozen collapse at every gated ρ
   (e.g. ρ = 0.99: frozen 4.7737 → refit 1.2134 against ID 0.1172). But the
   sub-prediction that Δ_info is "small-positive" held in **direction and
   declared bound only**: Δ_info is positive and bounded by the cue's genuine
   ID information value as committed (1.0962 < H(Y) − H(Y|s) = 3.2094 bits at
   ρ = 0.99), yet it is not *small* at high ρ — 0.51 to 1.10 bits.
3. **Descriptive ratio ≥ 2 at ρ ≥ 0.9:** DEVIATED at ρ = 0.9 (1.09) and
   ρ = 0.95 (1.60); met at ρ = 0.99 (3.25). Reported in full above.
4. **§1 prediction 3 / G-B3 (lawful flat):** HELD, margin ≥ 4.9×, both shifts.
5. **§1 prediction 4 / G-B4 (monotone masking curve):** HELD, strictly, every
   adjacent pair.
6. **§1.1 banded f prediction (0.1 ≤ f ≤ 1 at ρ ≥ 0.9):** HELD at all three ρ;
   no envelope exceedance; lower edge grazed at ρ = 0.9.
7. **ρ = 0.9 coincidence (C-B5):** consistent at 0.0014 / 0.0050 bits
   (detector only; carries no evidential weight by design).
8. **Anticorrelate descriptives (committed full reporting):** anchor-predicted
   asymmetry (anti < dec below ρ = 0.9, equal at 0.9, anti > dec above)
   reproduced in sign at every ρ.
9. **Power projection (§7):** realized cross-seed SEs 0.0035–0.0588 bits,
   within the projected ~0.01–0.03 range except the largest-effect cells
   (naive ρ=0.99 transfer quantities, SE ≈ 0.054–0.059), where the effect is
   ~60× the SE; the materiality floor bound δ nearly everywhere as projected.

## 5. PQ1–PQ5 checklist (§5.6)

- **PQ1 (operational definitions): PASS.** Every uncertainty quantity is a
  held-out CE in bits of 𝒜_log (C = 1.0, standardize = True, recorded verbatim
  in every row's settings) on the 256-d penultimate features; K_readout = 2570
  (+ CNN params 1,739,210 recorded). E (direction-only, shared boxes): ~313 s
  train + ~3 s feature extraction per run on one GB10 (probe-verified;
  ~3.5 h wall per node for the grid); readout battery ≈ 62 s/cell (pilot §5).
- **PQ2 (CV > 20%): PASS.** CV across the committed grid of cell means
  ({ρ} × {arm} × {CEhat_ID, CE_Q^dec(r_P), CEhat_Q-refit}) = **1.267 (126.7%)**
  (`machine_summary.json: PQ2_cv = 1.2666`), threshold 0.20.
- **PQ3 (task relevance + declaration): PASS.** The proxy is the task's own
  label CE; all 800 rows carry the §2-bis declaration; ID/refit rows are
  `raw_heldout_CE` per the §5.9.A split (differences licensed); frozen rows are
  CE_Q(r_P), never entropy.
- **PQ4 (cross-regime consistency): PASS.** One readout class, one C, one
  standardize flag, one training recipe, identical splits and units across all
  arms, ρ cells, and both shifts; only the coupling and shift draw vary.
- **PQ5 (λ_var discipline): PASS.** No ℛ-objective optimized; λ_var appears in
  no criterion (recorded in `results_summary.json → pq`).

**§5.7 declaration checklist: TICKED.** All 800 rows carry `quantity`,
`distribution`, `fit_distribution`, `not_an_entropy`, `licensed`,
`license_ref`; `fourq.check_quantity_declarations` ran on the full row set
before serialization (the writers refuse undeclared tables); every absolute
CEhat row carries `abs_ce_bias_bits_disclosed = 0.3708`; every frozen-shift row
carries `not_an_entropy = True`; no row above log₂10 is declared as any
entropy. Readout convergence: the harness records the per-fit converged flag
(`n_iter < max_iter`) in its diagnostics; fit settings are serialized per
(cell × shift) (`convergence_sample`, sklearn 1.9.0, lbfgs, tol 1e-10,
max_iter 10000); all 24 selftest fits converged; the run raised no
non-convergence disclosure (rc=0).

**In-run noise floors (§4):** the lawful-arm 10-fold floor recomputed on every
lawful Z^B_ID: mean CE 0.3916–0.4549 bits across the 40 lawful cells (mean of
means 0.4222), 2×SE 0.0139–0.0485 — every value below the 0.10 materiality
floor, so the committed δ formula's floor term bound, as projected from the
pilot (0.3787, 2×SE 0.0343).

**Training-run nondeterminism (free replicate measurement, §2 disclosure):**
the lawful arm's four ρ-cells are replicate trainings of identical data at
fixed seed. Replicate spread of CEhat_ID (derived from the per-seed arrays):
per-seed range across the 4 replicates mean 0.0253 bits, max 0.0445; the four
cross-seed cell means agree within 0.0088 bits (0.4102–0.4190). Training
nondeterminism is real at the few-hundredths-of-a-bit scale and comfortably
inside δ — consistent with G-B3's flatness margins.

## 6. §6.5 / §6.1–§6.4 anti-pattern self-audit (§5.8)

- **QUANTITY CONFLATION (§6.5) — clean.** No CE_Q(r_P) is described as H or
  H_𝒜 anywhere in this document or the serialized rows. The > log₂10 tell was
  applied to every number: the only values above 3.3219 bits are the naive
  frozen CE_Q(r_P) at ρ = 0.99 (4.7737 dec, 5.2748 anti) — quantity (4),
  `not_an_entropy=True`, handled as the prereg §5.8 said they must be. No
  soft-readout CE is claimed to bound any hard-class H_𝒜. The §5.9.A split is
  respected: no absolute CEhat is given a CE\*-point reading anywhere.
- **§6.1 in-sample fits — clean.** Every CEhat is held-out (5,000-fit /
  5,000-eval P split; 5-fold Q-internal cross-fit); the frozen readout is never
  evaluated on re-tinted versions of images it was fit on (§4 split).
- **§6.2 proxy gaming — clean.** C = 1.0 was fixed pre-run for the documented
  pilot convergence reason and never revisited; τ was not tuned (the amended
  selftest passed on its committed design; the FAIL history is preserved).
- **§6.3 post-hoc threshold motion — clean.** The δ formula, floor, gate
  constants, and seed list are those of commit d26850e; the one amendment
  (dc7b1fd) moved the §5.1 PASS definition and license wording only, loosened
  no gate, and was spent pre-unblinding.
- **§6.4 seed selection / selective reporting — clean.** All 10 committed
  seeds enter every seed-mean (per-seed arrays serialized for audit); all 80
  cells and both shifts are reported; the anticorrelate arm is reported in
  full; the failed descriptive ratio-≥ 2 prediction is reported beside the
  passed gates; the wrapper-finalizer infra hiccup is disclosed rather than
  silently tidied.

## 7. §5.5 underpowered-clause evaluation

Realized per-cell δ_cell = max(2×SE_seed, 0.10) for every gated comparison:
the floor (0.10) bound in all gated cells except naive ρ = 0.99, where
2×SE_seed slightly exceeded it — δ_cell = 0.1119 (CE_Q frozen, dec), 0.1177
(Δ_transfer, dec; the experiment-wide maximum), and on the descriptive anti
side 0.1109 / 0.1081. **No gated cell approached the 0.25-bit trigger
(max 0.1177 ≤ 0.25); the clause did not fire; no gate degrades to
"underpowered — inconclusive."** G-B2's additional δ_cell ≤ 0.25 requirement
is satisfied at all three gated ρ.

## 8. Blindness accounting (§8, restated honestly)

Observed **pre-registration** (clean-arm-only pilot, permitted design input):
the lawful-arm ID magnitude (~0.3787 bits, SE 0.0171), the standard-CIFAR
sanity accuracy (0.9240), and the C = 1e8 vs C = 1.0 convergence comparison.
Consequently the lawful-arm ID *level* measured here (0.4102–0.4190) carries
reduced evidential weight as a prediction — it anchored the δ formula and the
G-B1 band discussion. Observed **pre-unblinding, post-registration** (planted
synthetic data only, no trained features): the original §5.1 absolute-recovery
FAIL and the ~0.37–0.54-bit excess values that motivated §5.9.A — disclosed in
the amendment trail (`PRERUN_NOTE_2026-06-11.md`, the escalation, dc7b1fd).
**Never observed before the once-only run:** any naive-arm quantity, any
arm comparison, any shift quantity, any CE_Q(r_P), Δ_info, Δ_transfer, gap,
f(ρ), or gate input. The masking results (G-B1–G-B4, the masking curve, the
f fractions, the anticorrelate descriptives) are genuinely pre-registered,
blind outcomes.

## 9. Honest discussion / surprises

1. **What this establishes.** The Track-3 masking finding generalizes to real
   images with **learned** CNN representations under this construction: a
   spurious-cue-reliant CNN is *better* than the sufficient one in-distribution
   at every coupling, and its frozen interface collapses under the
   coupling-breaking shift by up to 3.56 bits (Δ_transfer, ρ = 0.99) while the
   sufficient arm moves at most 0.02 bits. This is the first pre-registered
   real-data confirmation of the paper's headline phenomenon — on CIFAR-10
   with end-to-end-trained features rather than a constructed toy channel.
2. **The biggest deviation: the Δ_transfer-vs-Δ_info split at moderate ρ.** The
   prediction "ratio ≥ 2 at ρ ≥ 0.9" failed at ρ = 0.9 (1.09) and 0.95 (1.60).
   Mechanistically the data offer a coherent reading the prereg under-weighted:
   the naive CNN does not merely *add* a cue route on top of intact image
   features — it **cannibalizes** image-feature learning. The sanity metric
   shows it directly: naive standard-CIFAR accuracy falls 0.891 → 0.558 as ρ
   rises, so under shift even a *refit* readout has less accessible label
   information in Z (Δ_info up to 1.10 bits). The collapse is interface-borne
   *and* representation-borne in comparable parts at ρ = 0.9, and dominantly
   interface-borne (3.25×) only at extreme coupling. The qualitative claim
   "brittleness is interface-borne, not information-borne" survives in
   direction everywhere (Δ_transfer > Δ_info at all gated ρ) but its
   quantitative form was wrong at moderate ρ — recorded as such.
3. **The f(ρ) shape.** f = 0.109 / 0.174 / 0.408: the learned system realises a
   strongly increasing fraction of the cue-only ceiling, grazing the
   committed band's 0.1 lower edge at ρ = 0.9. Had the G-B2 minimum been set at
   12% rather than 10% of the smallest envelope, ρ = 0.9 would have failed —
   the conservative threshold did its job, but the ρ = 0.9 effect should be
   described as "one-tenth of the cue ceiling", not more.
4. **The anchor arithmetic was sharp where it applied.** The ρ = 0.99 ID gap
   (−0.3018 vs the anchor-derived ≈ −0.306 against the realized lawful floor)
   and the dec/anti asymmetry sign pattern at every ρ both landed; the C-B5
   coincidence agreed to 0.0014 bits. The construction behaves as its closed
   forms say; everything the cue-only idealisation could not fix (the CNN's
   reliance split) is where the deviations live.
5. **Lawful-arm nondeterminism replicates.** The free replicate measurement
   (four identical-data trainings per seed) puts CNN training nondeterminism at
   ~0.025 bits mean per-seed range (max 0.045) on this recipe — an order of
   magnitude inside δ, and itself a useful calibration number for future
   real-data gates at this scale.
6. **Scope limits (the prereg non-claims, restated).** One dataset (CIFAR-10),
   one architecture (CNN9, 1.74 M params), one nuisance design (global
   class-indexed chromatic tint at fixed strength 0.20), constructed —
   not natural — spurious coupling, one readout class, constructed shifts. No
   claim about natural spurious correlations; no claim beyond this
   construction; no MLR/GRIT/RIFT stack claim; no claim that any soft-readout
   CE bounds a hard-class H_𝒜; absolute CE levels here carry a disclosed
   ≈ 0.371-bit estimator bias and no CE\*-point license.

## 10. Consequences (per the pre-registration)

- **Verdict row realized (§9):** "G-B1–G-B4 all PASS → Track-3 masking
  signature reproduced at scale," with the Δ_transfer-vs-Δ_info decomposition
  quantifying interface-vs-information as committed — including the honest
  finding that the decomposition is more information-loaded at moderate ρ than
  predicted.
- **What this licenses:** the paper-sequel **empirical section** — a
  pre-registered real-data masking demonstration with the full discipline trail
  (registration → escalation → single pre-unblinding amendment → once-only
  unblinding), citable against `results_summary.json` and this document.
  **Nothing else**: no program-level claim strengthens, no theorem-side object
  changes, and the §5.9.A licensing split travels with every absolute CE number
  quoted downstream.
- **Amendment budget:** spent (1/1, pre-unblinding). Any future change to this
  experiment's protocol is a new experiment.
- The masking-at-scale machine verdict is recorded for the campaign:
  `machine_summary.json → masking_at_scale_reproduced: true`.

## 11. Gate verdicts

| Gate | Verdict | Basis |
|---|---|---|
| §5.1 (amended §5.9.A) | **PASS** | both planted CE\*-difference pairs recovered within τ = 0.05 (errs 0.0056 / 0.0007); absolute-recovery FAIL history preserved; licensing split applied |
| G-B1 | **PASS** | naive−lawful ID gap ≤ +δ at every ρ (in fact negative everywhere: −0.10 to −0.30) |
| G-B2 | **PASS** | naive Δ_transfer = 0.5524 / 1.0985 / 3.5603 ≥ 0.5 at ρ = 0.9/0.95/0.99; all δ_cell ≤ 0.25 |
| G-B3 | **PASS** | lawful \|Δ_info\|, \|Δ_transfer\| ≤ 0.021 ≪ δ = 0.10 at every ρ, both shifts |
| G-B4 | **PASS** | naive Δ_transfer strictly increasing: 0.1020 < 0.5524 < 1.0985 < 3.5603 |
| C-B5 | **consistent** | dec/anti at ρ = 0.9 agree to 0.0014 (naive) / 0.0050 (lawful) bits; no HALT |
| **Composite (§5.3)** | **masking-at-scale REPRODUCED** | G-B1 ∧ G-B2 ∧ G-B3 ∧ G-B4, C-B5 clean, license PASS |
