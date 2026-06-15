# Results: Track 4 Acceptance Gates 2–3 (Readout-Geometry Experiment B)

**Date:** 2026-06-09
**Pre-registration:** `PREREGISTRATION.md`, committed before the run in
`13377dd` ("experiments: pre-register track 4 experiment B (readout
geometry)"). **No amendments were made** (§7 of the pre-registration); no
parameter was changed after the first run; the experiment was run exactly
once (single process, runtime 585.9 s, consistent with the process lifetime;
all 120 rows present).
**Raw data:** `results.csv` (120 rows: 10 seeds × 4 \(d_z\) × 3 rungs),
`results_summary.json`, `analytic_predictions.json`. All 40 rung-(i) fits
converged (max final gradient inf-norm 3.7e-17, zero step-halvings anywhere,
so measured rung-(i) train FLOPs equal the deterministic formula exactly);
rung-(iii) decode accuracy 1.0 at every cell; atom injectivity held at every
cell (min pairwise atom distance 0.0416, as committed pre-run).

**Headline:** gates 2 and 3 are **met** under their pre-registered criteria.
B1, B2 (primary), B4, and B5 all passed; the §11 example falsifier was not
observed. The secondary criterion B3 is graded **INCONCLUSIVE under the
pre-registration's own underpowered provision** despite both point estimates
passing — reported in full below, no massaging.

> **Terminology clarification (2026-06-10, additive — the frozen record
> below is unchanged).** The quantities labeled H / \(\hat H\) / CE in
> this document are **held-out cross-entropies** of each rung's predictive
> distribution. They are valid in-distribution estimates of the declared
> readout-constrained referents (\(H_{\mathcal A}\)-style quantities) per
> the B1 proxy validation against the exact per-seed population floors
> \(CE^*_{lin}\) and the B2-a validation against the exact Bayes value
> \(H(Y \mid Z) = 0\); cross-entropy upper-bounds the corresponding
> entropy quantities, and the validation is what licenses the estimates.
> Paper-side notation reserves \(H\)/\(H_{\mathcal A}\) for theorem-side
> quantities and writes measured values as \(\widehat{CE}\)
> (`papers/lawful-compression-readout-geometry/main.tex` §2). Note also
> the rung-(iii) framing recorded there: the oracle proves zero task
> uncertainty is *available* given codebook storage + search; it does not
> prove ordinary training finds it cheaply. Falsifier referent update:
> per the 2026-06-10 correction in `READOUT_GEOMETRY_THEOREM.md` §4.3
> (Conjecture 6's basis-restricted version refuted by an in-basis
> quadratic decoder), B5's survival is now recorded against the narrowed
> Conjecture 6′ (affine-threshold compositions).
>
> **Final-notation mapping (2026-06-10, tag `paper-lcr-v3-draft`; per
> `MLR_FORMAL_DEFINITIONS.md` §2-bis):** the floor\* column is the exact
> per-seed \(CE^{\star}_{lin}(P)\); the measured rung values are
> \(\widehat{CE}\) estimates of each rung's declared-class \(CE^{\star}\)
> (B1-validated for the linear rung); no hard-class \(H_{\mathcal A}\) is
> measured here, and \(CE^{\star}\) bounds none in either direction.

---

## 1. Measured cell means (10 seeds; ± is cross-seed std)

H is held-out cross-entropy of each rung's predictive distribution,
bits/sample (the declared readout-constrained proxy; the unconstrained
\(H(Y \mid Z) = 0\) exactly at every cell by construction). floor\* is the
seed-mean of the per-seed exact population optimum of the linear class,
committed before the run.

| \(d_z\) | floor\* | rung (i) linear | rung (ii) two-layer | rung (iii) oracle | K (i/ii/iii) | E_inf (i/ii/iii) |
|---|---|---|---|---|---|---|
| 4 | 0.6053 | 0.6066 ± 0.0649 | 0.3729 ± 0.1211 | 1.44e-6 ± 0 | 5 / 385 / 9100 | 15 / 839 / 23661 |
| 6 | 0.4799 | 0.4798 ± 0.0989 | 0.0569 ± 0.0451 | 1.44e-6 ± 0 | 7 / 513 / 12740 | 19 / 1095 / 34581 |
| 8 | 0.3196 | 0.3196 ± 0.1689 | 0.0007 ± 0.0008 | 1.44e-6 ± 0 | 9 / 641 / 16380 | 23 / 1351 / 45501 |
| 12 | 0.0666 | 0.0668 ± 0.1192 | 3.6e-6 ± 3e-6 | 1.44e-6 ± 0 | 13 / 897 / 23660 | 31 / 1863 / 67341 |

The large cross-seed stds on rungs (i)–(ii) are **real construction
heterogeneity** (each seed has a different projection, hence a different
exact floor), not measurement noise: the *paired* per-seed statistics below
are tight.

## 2. Pre-registered criteria: PASS / FAIL / INCONCLUSIVE

\(\delta = 0.010\) bits (pre-registered). All values are seed-means over all
10 seeds; no run, seed, or condition is omitted. Paired SEs are SEs of the
seed-mean of per-seed differences (the comparison statistics).

### B1 — linear floor realized out-of-sample (proxy validation)

\(\lvert \overline{CE}_i - \overline{CE^*_{lin}} \rvert < 0.010\) at every \(d_z\):

| \(d_z\) | measured mean diff | paired SE | threshold | verdict |
|---|---|---|---|---|
| 4 | +0.0013 | 0.0004 | < 0.010 | **PASS** |
| 6 | −0.0001 | 0.0007 | < 0.010 | **PASS** |
| 8 | +0.0000 | 0.0005 | < 0.010 | **PASS** |
| 12 | +0.0002 | 0.0002 | < 0.010 | **PASS** |

The trained linear readout lands on the committed exact class floor to ≤
0.0013 bits at every cell — the cross-talk floor is not an artifact of
training budget. The pre-declared near-separable provision at \(d_z = 12\)
was **not needed** (criterion passed outright). Paired SEs are an order of
magnitude inside the 0.005 underpowered bound, so pass/fail grading is valid.

### B2 — strict gap with counted payment (primary, gate 2), cells \(d_z \in \{4,6,8\}\)

| \(d_z\) | B2-a: \(\overline{CE}_{iii}\) < 0.001 | B2-b: gap \(\overline{CE^*_{lin}} - \overline{CE}_{iii}\) ≥ 0.15 | B2-c: \(K_{iii} > K_i\), \(E^{inf}_{iii} > E^{inf}_i\) | verdict |
|---|---|---|---|---|
| 4 | 1.44e-6 ✓ | 0.6053 ✓ | 9100 > 5; 23661 > 15 ✓ | **PASS** |
| 6 | 1.44e-6 ✓ | 0.4799 ✓ | 12740 > 7; 34581 > 19 ✓ | **PASS** |
| 8 | 1.44e-6 ✓ | 0.3196 ✓ | 16380 > 9; 45501 > 23 ✓ | **PASS** |

Rung (iii) has zero cross-seed variance (exact decoding at every seed), so
the comparison is fully powered. The measured strict H reductions of 0.32 to
0.61 bits below the linear class's **exact, budget-independent** floor are
carried by a rung whose counted K is 1820–2275× larger and counted inference
E 1577–1978× larger — the quantified accounting of gate 2, under the
equivalent-budget reading pre-registered in §1 (no budget moves the linear
class below CE\*_lin; the rung that beats it pays counted K/E).

### B3 — trained nonlinear rung beats the linear floor (secondary, directional), cells \(d_z \in \{4,6\}\)

| \(d_z\) | \(\overline{CE}_{ii}\) | pre-reg threshold | point estimate | SE of \(\overline{CE}_{ii}\) | per-seed: seeds below own floor by ≥ 0.05 | verdict |
|---|---|---|---|---|---|---|
| 4 | 0.3729 | ≤ 0.4540 | passes | 0.0404 | 10/10 (worst margin 0.1096) | **INCONCLUSIVE** (underpowered provision) |
| 6 | 0.0569 | ≤ 0.3599 | passes | 0.0150 | 10/10 (worst margin 0.2836) | **INCONCLUSIVE** (underpowered provision) |

Both point estimates pass their pre-registered thresholds, and every single
seed's trained net lands below that seed's own exact floor by at least 0.05
bits. However, the pre-registration's underpowered provision ("any
comparison whose cross-seed seed-mean SE exceeds 0.005 bits is graded
INCONCLUSIVE rather than pass/fail") binds: the SEs are 0.0404 and 0.0150.
The provision was written with measurement noise in mind and did not
anticipate that real cross-projection heterogeneity would dominate rung-(ii)
variance; per the no-massaging rule it is applied as written. **B3 is
therefore INCONCLUSIVE**, with the per-seed evidence reported above as
description, not as a gate result. Neither gate verdict rests on B3 (B2 is
the primary strict-gap criterion). A future experiment wanting a powered B3
should pre-register the criterion on paired per-seed margins.

### B4 — accounting registration over the declared ladder (gate 2)

Deterministic verification: K and E_inf strictly increase (i) → (ii) → (iii)
at every one of the 40 cells (§1 table; values exactly as committed in the
pre-registration). Registration statement: the only pre-registered
strict-H-improvement criterion that **passed** (B2) is realized by rung
(iii), whose counted K and E_inf strictly exceed rung (i)'s at every cell.
**PASS** — and, per RGT Proposition 4, this is bookkeeping by construction
of the declared ladder; its empirical content is that the payments are
real counted quantities (PQ1), not that the coincidence is derived.

### B5 — falsifier non-occurrence (gate 3)

- **B5-a (§11 falsifier event: readout at linear K/E below the linear
  class's exact floor):** requires \(\overline{CE}_i < \overline{CE^*_{lin}} - 0.010\)
  at some cell. Measured mean paired diffs: +0.0013 / −0.0001 / +0.0000 /
  +0.0002 (paired SEs 0.0002–0.0007); worst single-seed diff −0.0032.
  **Not observed** at any cell or any seed. **PASS.**
- **B5-b (within-ladder cheap-beats-expensive by > \(\delta\)):**
  (i) vs (ii): per-seed paired diff \(CE_i - CE_{ii}\) is positive at every
  seed of \(d_z \in \{4,6,8\}\) (minima +0.1082 / +0.2862 / +0.0473 — the
  event, requiring < −0.010, is 6–16 paired SEs away) and ≈ 0 at \(d_z = 12\)
  (minimum +0.0000; both rungs at their floors ≈ 0 — equality, not the
  event). (ii) vs (iii) and (i) vs (iii): rung (iii) has the lowest CE
  everywhere (cross-rung differences vs (ii) at \(d_z = 12\) are ~2e-6,
  far inside \(\delta\)). **Not observed.** **PASS.**

No readout in this experiment reduced task-relevant uncertainty at
equal-or-lower counted K and E. Falsifier 1 (`FALSIFIABILITY_CONDITIONS.md`
§4, geometry-ignorant matching geometry-aware under equivalent K/E): not
observed — the geometry-ignorant linear rung stayed pinned to its cross-talk
floor while the geometry-aware rungs beat it only with counted payment.
Falsifier 2 (lower H at equal-or-lower K and E across regimes): not
observed at any cell. The anti-pattern self-audit is §4 below.

---

## 3. PQ1–PQ5 checklist, with evidence

- **PQ1 (observable operational meaning): PASS.** K = stored parameter /
  codebook-entry counts (deterministic, committed pre-run). E = analytic
  per-sample inference FLOPs by stated convention plus training FLOPs
  (rung (i): formula verified exact — zero halvings; rung (ii): declared
  step-count convention; rung (iii): 0, disclosed pre-run, with the cost
  sitting in K and E_inf as pre-declared) and wall-clock (direction only:
  mean inference wall-clock orders i < ii < iii at every cell). H = held-out
  cross-entropy in bits.
- **PQ2 (CV > 20% across test conditions): PASS.** CVs over the grid of
  cell means (`results_summary.json` `_PQ2_CV`): H 134%, K 145%, E_inf
  149%, E_train 150%, wall 132%. All ≫ 20%.
- **PQ3 (task-relevance): PASS.** The H proxy is the task's own held-out
  label cross-entropy; referent declared pre-run (readout-constrained
  \(H_{\mathcal A}\) per class). Validated against exact ground truth at
  both ends: the linear rung against the per-seed exact class floor
  (≤ 0.0013 bits, B1) and the oracle rung against the exact Bayes value
  \(H(Y\mid Z) = 0\) (measured 1.44e-6 bits = the pre-registered clip
  artifact exactly, B2-a).
- **PQ4 (cross-regime consistency): PASS.** Identical units (bits/sample,
  FLOPs, params), identical train/test draws within each cell across all
  three rungs (same data streams), identical evaluation sets; the only
  varied quantities are the pre-registered \(d_z\), seed, and rung.
- **PQ5 (\(\lambda_{var}\) discipline): PASS.** No \(\mathcal R\)-objective
  was optimized and no \(\lambda_{var}\) value was used or tuned anywhere.
  As pre-declared: the improving moves here have \(\Delta H < 0\) with
  \(\Delta K > 0\) and \(\Delta E > 0\), so the sign of \(\Delta\mathcal R\)
  is \(\lambda_{var}\)-dependent and **no \(\mathcal R\)-descent claim is
  made**. Gate 2 concerns the signs of the three terms separately, which is
  exactly what was measured. No identification with any other lambda-like
  object was made.

## 4. Anti-pattern self-audit (`FALSIFIABILITY_CONDITIONS.md` §6)

- **§6.1 in-sample fits — clean.** Every H measurement is on held-out data
  (fresh 100k-sample test sets per cell). Rung (ii)'s restart selection
  uses final *train* loss only (pre-declared); its reported CE is held-out.
  No in-sample quantity enters any criterion.
- **§6.2 symbol reuse without derivation — clean.** No lambda-like quantity
  appears anywhere (PQ5). H, K, E carry their stated operational
  definitions; the K/E conventions were committed before the run and not
  adjusted after.
- **§6.3 tautological identities — clean.** H (held-out cross-entropy on
  labels realized from the construction) is measured independently of K
  (parameter counts) and E (operation counts); none is an algebraic
  function of another. The analytic floors were computed from the exact
  atom distribution by convex optimization — no sampling, no training —
  and committed before the run.
- **§6.4 selective reporting — clean.** All 120 runs are in `results.csv`
  and enter the seed-means; no seed, cell, or rung was dropped. The
  pre-declared descriptive observables are reported (§5 item 4: rung (ii)
  at \(d_z \in \{8, 12\}\)); the B3 grading was *downgraded* to
  INCONCLUSIVE by the pre-registration's own provision even though its
  point estimates passed — the provision was applied against the apparent
  result, not for it. One infrastructure note disclosed: during the run,
  the asynchronous monitoring channel delivered several spurious
  progress/completion notifications inconsistent with the on-disk log
  (including fabricated-looking summary values); every number reported here
  was read directly from the committed artifacts after verified process
  exit, and the artifacts are deterministic given the declared seed streams
  (independently reproducible by rerunning the committed script).

## 5. Honest discussion / surprises

1. **No amendments were needed or made.** The single-amendment policy was
   not exercised; the near-separable provision was not needed.
2. **The exact floors were hit, seed by seed.** The trained linear readout
   reproduced the committed per-seed analytic floors to ≤ 0.003 bits at
   every one of 40 cells (seed-mean ≤ 0.0013). The cross-talk floor of RGT
   Theorem 3(ii)'s family is, in this construction, a measured out-of-sample
   quantity, not a narrative.
3. **B3's grading is the experiment's main self-inflicted lesson.** The
   underpowered provision was keyed to seed-mean SE without anticipating
   that per-seed floors vary by construction (different projections), which
   inflates unpaired SEs while paired margins are decisive (20/20 seeds
   below their own floor by ≥ 0.05 bits, worst margin 0.1096). Applied as
   written: INCONCLUSIVE. The class-level strict gap that B3 was probing is
   independently established by B2 (and theorem-side by RGT Theorem 3), so
   no gate rests on it. Future pre-registrations should pair such criteria.
4. **Pre-declared descriptive observations (no gate):** rung (ii) at
   \(d_z = 8\) reaches 0.0007 bits (essentially decodes) and at
   \(d_z = 12\) reaches 3.6e-6 bits; at \(d_z = 4\) (load 4.0) the trained
   net is clearly optimization/expressivity-limited (0.3729 bits — well
   below the linear floor on every seed, far above the oracle). The
   trained-rung gap narrows monotonically with load, consistent with the
   computation-in-superposition picture (context, not a claimed theorem).
   Init-selection diagnostics: all three restarts were exercised by the
   train-loss selection (chosen init counts 14/8/18 over the 40 cells), as
   pre-declared.
5. **The low-load end behaved as pre-declared.** At \(d_z = 12\), 5/10
   seeds are linearly separable (floor ≈ 0) and rungs (i)/(ii) both sit at
   ≈ 0 there: no superposition pressure, no gap — which is why the gap
   cells were pre-registered as \(d_z \in \{4, 6, 8\}\). The non-claim
   "larger readout classes need not help" is visible in the data exactly
   where the theorem says it should be.
6. **Scope.** This is one constructed family in the exact regime of RGT
   Theorem 3's general setting (noiseless superposition, exactly-k uniform
   sparsity, designated-feature task, declared ladder). It demonstrates the
   Track 4 gate discipline end-to-end with admissible proxies; it
   establishes nothing beyond the construction's scope, and the universal
   accounting statement remains **Conjecture 6** (open) — this experiment
   is one survival of its empirical shadow, not a proof.

## 6. Gate verdicts

| Gate | Verdict | Basis |
|---|---|---|
| 2 | **MET** | B1 (floor realized, ≤ 0.0013 bits, out-of-sample), B2 (strict gap 0.32–0.61 bits below the exact linear floor at counted ΔK, ΔE > 0), B4 (ladder accounting verified), PQ1–PQ5 all passed under pre-registered thresholds. B3 INCONCLUSIVE (own provision) — not load-bearing. |
| 3 | **MET within this construction's scope** | §4 falsifiers stated pre-run with their RGT referents (Conjecture 6 wiring) and survived: B5-a and B5-b events not observed at any cell or seed; anti-pattern self-audit §4. Survival is cumulative — this is the first recorded Track 4 survival, not a closed book. |
