# Pre-Registration: Track 4 Acceptance Gates 2–3 (Readout-Geometry Experiment B)

**Date:** 2026-06-09
**Track:** 4 — Readout-Geometry Theorem (`THEOREM_TRACKS.md`)
**Construction target:** `GRIT_ITERATION_001.md` §7 Experiment B (sparse superposed features; readout ladder)
**Theorem referents:** `READOUT_GEOMETRY_THEOREM.md` (cited below as **RGT**) Theorem 1, Theorem 3, Lemma 3.2, Proposition 4, Proposition 5, Conjecture 6
**Falsifiers in scope:** `FALSIFIABILITY_CONDITIONS.md` §4 (and the §11 example falsifier of `MLR_FORMAL_DEFINITIONS.md`), audited against the §6 anti-patterns

```text
Discipline: this document is committed BEFORE any experimental run.
Every pass/fail criterion below is fixed now. A failed gate honestly
reported is an acceptable outcome; a massaged pass is not.
```

**Commit discipline.** This pre-registration, the experiment script
(`readout_geometry_experiment.py`), and `analytic_predictions.json` are
committed and pushed before the experiment is run. The analytic predictions
were computed before this commit by `readout_geometry_experiment.py
--analytic`, which **draws no samples and trains no readout**: it enumerates
the exact finite atom distribution of the construction, checks injectivity,
and computes the exact population optimum of the linear-logistic readout
class on that distribution by damped Newton (a population-level convex
computation, the Track 4 analogue of Track 3's quadrature). The experimental
grid (§3) has not been run at commit time.

**Design disclosure (selection, pre-data).** The construction parameters
(d_feat, k, the d_z grid, and the projection seed stream) were selected by
analytic population-floor scans over candidate constructions — same
no-sampling, no-training computation as `--analytic` — to ensure (a) the
linear floor is bounded away from zero at the declared gap cells for every
seed, and (b) the floor varies strongly across the grid (PQ2 by design).
This is construction design, not measurement, and no trained readout or
sampled dataset existed before this commit. The rung (ii) optimizer was
never run on this construction (or any sparse-superposition data) before
this commit; its hyperparameters were fixed from generic practice, and §6
therefore gives rung (ii) an explicit inconclusive band rather than a
pretense of analytic precision.

---

## 1. What is being tested

Track 4 gates 2 and 3 (`THEOREM_TRACKS.md`); gate 1 is theorem-side and
already met (RGT §5.2):

- **Gate 2** — quantified accounting: a measured H reduction attributed to a
  larger readout class accompanied by counted \(\Delta K > 0\) and/or
  \(\Delta E > 0\) under equivalent-budget comparison, PQ1–PQ5 satisfied.
- **Gate 3** — the §4 falsifiers stated and survived; in particular the §11
  example falsifier (geometry-ignorant readout matching geometry-aware
  readout under equivalent K/E constraints) must remain unobserved.

**Declared H-proxy referent** (per `LAWFUL_COMPRESSION_THEOREM.md` §7.3 item
3 discipline): the proxy measures **readout-constrained** uncertainty — the
held-out cross-entropy (bits) of each rung's predictive distribution, i.e.
an upper bound on \(H_{\mathcal A}(Y \mid Z)\) for that rung's class that is
tight at the class optimum. The unconstrained referent \(H(Y \mid Z)\) is
**exactly 0 by construction** at every cell (atom injectivity, verified:
minimum pairwise atom distance 0.0416 over all 40 cells; RGT Lemma 3.2 is
the licensing statement). Every bit of measured uncertainty is therefore
interface-induced, which is precisely the Track 4 setting.

**Equivalent-budget comparison, realized analytically.** The linear class's
floor is an *exact population quantity*: no amount of extra training budget,
samples, or optimizer sophistication moves a linear-logistic readout below
`CE_lin*` (it is the class optimum). The equivalent-budget content of gate 2
is therefore: (a) the cheap rung cannot beat its analytic floor at ANY
budget within its class; (b) the rungs that do go below it carry counted,
strictly larger K and E. Both are auditable below.

## 2. Construction — geometry-dependence as a theorem of the construction

All randomness from `numpy.random.default_rng` with declared streams.

- **Features:** \(F\) uniform on the \(\binom{16}{4} = 1820\) binary
  patterns with exactly \(k = 4\) of \(d_{feat} = 16\) features active.
- **Superposition:** fixed per-seed projection \(A \in \mathbb R^{d_z \times 16}\),
  entries i.i.d. N(0,1), columns normalized to unit norm
  (`default_rng(20000 + seed)`); \(Z = AF\), noiseless. \(d_z < d_{feat}\)
  at every grid point: features are superposed, columns overlap, and the
  designated feature's direction receives cross-talk from the other three
  active features.
- **Task:** \(Y = F_1\) (the designated feature's value).
  \(P(Y{=}1) = 4/16 = 0.25\); \(H(Y) = h_2(0.25) = 0.8113\) bits.
  **Why one feature and not parity:** parity is already handled exactly,
  theorem-side (RGT Theorem 2, floor imported from LCT Prop 3); reading one
  designated feature out of superposition is the canonical
  computation-in-superposition task (corroboration-ledger line), and using
  it keeps this experiment's hardness purely cross-talk-geometric rather
  than entangling a second (parity) hardness source. One task, declared
  here, no post-hoc switching.
- **Ground truth:** generic unit-norm Gaussian columns give
  \(\mathrm{spark}(A) > 2k\)-type injectivity on the 1820 atoms; verified
  directly per cell (`analytic_predictions.json`: min pairwise atom distance
  ≥ 0.0416 ≫ 0 at all 40 cells), so \(H(F \mid Z) = H(Y \mid Z) = 0\)
  **exactly** at every cell.
- **Superposition load axis (PQ2):** \(d_z \in \{4, 6, 8, 12\}\) — loads
  \(d_{feat}/d_z = 4.0, 2.67, 2.0, 1.33\). The analytic linear floor moves
  from ~0.61 bits to ~0.07 bits across this axis (§5), so the H proxy has
  large real variance across conditions.

## 3. Fixed parameters (no changes after first run)

| Parameter | Value |
|---|---|
| \(d_{feat}\), \(k\), task | 16, 4, \(Y = F_1\) |
| \(d_z\) grid | \(\{4, 6, 8, 12\}\) |
| seeds | 0–9 (10 seeds); projection `default_rng(20000+seed)`; data `default_rng(30000+1000·seed+d_z)`; net inits `default_rng(40000+1000·seed+10·d_z+init)` |
| \(n_{train}\) / \(n_{test}\) | 20,000 / 100,000 (fresh held-out draws per cell) |
| rung (i) trainer | logistic regression, damped Newton/IRLS, **25 iterations fixed**, ridge \(10^{-8}\), step-halving; convergence diagnostics recorded (track-3 protocol) |
| rung (ii) | two-layer ReLU net, width \(m = 64\), minibatch Adam (lr \(3{\cdot}10^{-3}\), \(\beta = (0.9, 0.999)\)), batch 2000, **1000 epochs fixed**, 3 inits, selection by final **train** loss only |
| rung (iii) | oracle nearest-atom decoder over the 1820 constructed atoms; predictive probability clipped at \(1 - 10^{-6}\); no training |
| analytic floor | damped Newton, 300 iterations, ridge \(10^{-10}\), on the exact 1820-atom distribution |

## 4. Declared readout ladder, proxies, and accounting (PQ1)

The ladder instantiates RGT Definition 1.2. Conventions (stated, deterministic;
\(D = d_z + 1\), \(m = 64\), \(C = 1820\)):

| Rung | Class | K (params) | E_inf (FLOPs/sample) |
|---|---|---|---|
| (i) | linear logistic \(\sigma(w \cdot z + b)\) | \(D\) | \(2D + 5\) |
| (ii) | two-layer ReLU, width 64, logistic output | \(mD + (m+1)\) | \(2mD + m + 2(m+1) + 5\) |
| (iii) | oracle nearest-atom decoder (codebook + labels) | \(CD\) | \(3Cd_z + C + 1\) |

Numerically (from `analytic_predictions.json`):

| \(d_z\) | K_i | K_ii | K_iii | E_i | E_ii | E_iii |
|---|---|---|---|---|---|---|
| 4 | 5 | 385 | 9100 | 15 | 839 | 23661 |
| 6 | 7 | 513 | 12740 | 19 | 1095 | 34581 |
| 8 | 9 | 641 | 16380 | 23 | 1351 | 45501 |
| 12 | 13 | 897 | 23660 | 31 | 1863 | 67341 |

**K and E_inf strictly increase up-ladder at every cell by construction**
(the RGT Definition 1.2 declaration; verified deterministically in RESULTS).
Class inclusion note: the rung (ii) predictor class contains rung (i)'s
exactly (a ±u ReLU pair reproduces any linear map), so Theorem 1 applies to
the (i)→(ii) class step as well as the accounting registration.

- **H proxy** — held-out cross-entropy (bits/sample) on \(n_{test}\) fresh
  draws; referent declared in §1. Out-of-sample throughout.
- **E proxies** — (a) E_inf: analytic per-sample inference FLOPs (table
  above, stated convention); (b) training FLOPs: rung (i) by the track-3
  IRLS formula (measured halvings recorded); rung (ii) by stated convention
  steps × batch × 3·E_inf_ii + 10·K_ii per step, summed over 3 inits; rung
  (iii) **0** — the oracle decoder is not trained; its cost sits entirely in
  K (the stored codebook *is* the payment) and E_inf. To prevent post-hoc
  wiggle: the **ladder-monotonicity and falsifier comparisons below use
  (K, E_inf)**; training FLOPs and wall-clock are reported descriptively
  (wall-clock direction-only, shared box).
- **\(\lambda_{var}\) (PQ5)** — no \(\mathcal R\)-objective is optimized and
  no \(\lambda_{var}\) value is used anywhere. Unlike Track 3's lawful arm
  (where \(\Delta\mathcal R < 0\) held for every \(\lambda_{var} \ge 0\)),
  here the improving moves have \(\Delta H < 0\) **and** \(\Delta K, \Delta E > 0\),
  so the sign of \(\Delta\mathcal R\) is \(\lambda_{var}\)-dependent and
  **no \(\mathcal R\)-descent claim is made or implied**. Gate 2 concerns
  the signs of \(\Delta H, \Delta K, \Delta E\) separately. Declared
  explicitly to satisfy PQ5's no-silent-tuning clause.

## 5. Analytic ground truth (computed pre-commit, no sampling, no training)

From `analytic_predictions.json` (per-seed exact values; seed-summary here):

| \(d_z\) | load | CE_lin\* mean (bits) | CE_lin\* min | CE_lin\* max | \(H(Y\mid Z)\) |
|---|---|---|---|---|---|
| 4 | 4.00 | **0.6053** | 0.4804 | 0.7026 | 0 |
| 6 | 2.67 | **0.4799** | 0.3027 | 0.5995 | 0 |
| 8 | 2.00 | **0.3196** | 0.0457 | 0.5540 | 0 |
| 12 | 1.33 | **0.0666** | 0.0000 | 0.4107 | 0 |

CE_lin\* is the exact population optimum of the linear-logistic class —
the empirical face of RGT Theorem 3(ii)'s cross-talk floor for this family.
At \(d_z = 12\), 5 of 10 seeds are linearly separable (floor ≈ 0): the
low-load end where superposition pressure vanishes. This is expected,
pre-declared, and is why the gap cells below are \(d_z \in \{4, 6, 8\}\).

**Known estimation effects (inform tolerances):** rung (i) excess risk
≈ \(D/(2 n_{train} \ln 2) \le 13/(2 \cdot 20000 \cdot \ln 2) = 4.7{\cdot}10^{-4}\)
bits; held-out CE standard error ≲ 0.004 bits/seed at \(n_{test} = 10^5\),
seed-mean SE ≲ 0.0015 bits. Rung (iii) under noiseless injective atoms:
exact decoding, predicted CE \(= -\log_2(1 - 10^{-6}) \approx 1.4{\cdot}10^{-6}\)
bits.

## 6. Pre-registered pass/fail criteria

Tolerance \(\delta = 0.010\) bits on seed-mean comparisons (>2× headroom over
the §5 bias+noise estimates; same value as Track 3). All criteria evaluate
**seed-means over all 10 seeds** (no seed selection). Underpowered
provision: any comparison whose cross-seed seed-mean SE exceeds 0.005 bits
is graded INCONCLUSIVE rather than pass/fail.

**B1 — linear floor realized out-of-sample (proxy validation):** at every
\(d_z\), \(\lvert \overline{CE}_i - \overline{CE^*_{lin}} \rvert < \delta\).
Pre-declared near-separable provision: at \(d_z = 12\) only (analytic floor
< 0.05 for half the seeds), the trained 25-iteration IRLS and the
300-iteration analytic Newton have asymmetric budgets on separable cells; if
B1 fails there with \(\overline{CE}_i \in [\overline{CE^*_{lin}} - \delta,\ \overline{CE^*_{lin}} + 0.025]\),
that cell is graded INCONCLUSIVE (optimization-budget asymmetry), not FAIL.
This provision can never convert a fail into a pass at the gap cells.

**B2 — strict gap with counted payment (primary, gate 2):** at every gap
cell \(d_z \in \{4, 6, 8\}\):

- **B2-a:** \(\overline{CE}_{iii} < 0.001\) bits (the larger-class rung goes
  essentially to the Bayes value \(H(Y\mid Z) = 0\));
- **B2-b:** \(\overline{CE^*_{lin}} - \overline{CE}_{iii} \ge 0.15\) bits
  (strict gap at least 0.15 — conservative vs the analytic floors
  0.61/0.48/0.32);
- **B2-c:** the rung realizing the gap carries counted
  \(K_{iii} > K_i\) and \(E^{inf}_{iii} > E^{inf}_i\) (deterministic table §4).

**B3 — trained nonlinear rung beats the linear floor (secondary,
directional):** at \(d_z \in \{4, 6\}\):
\(\overline{CE}_{ii} \le \overline{CE^*_{lin}} - \max(0.05,\ 0.25 \cdot \overline{CE^*_{lin}})\)
— binding thresholds **0.4540** bits (\(d_z{=}4\), floor 0.6053) and
**0.3599** bits (\(d_z{=}6\), floor 0.4799). Pre-declared interpretation
bands: result in \((\overline{CE^*_{lin}} - 0.05,\ \overline{CE^*_{lin}} + \delta]\)
= INCONCLUSIVE (optimization-limited; the class-level gap is already
established by B2 and theorem-side by RGT Theorem 3); result
\(> \overline{CE^*_{lin}} + \delta\) = reported as a rung-(ii) optimization
failure — **not** a falsifier (Theorem 1 governs class infima, not trained
members) and **not** silently dropped.

**B4 — accounting registration over the declared ladder (gate 2,
bookkeeping per RGT Proposition 4):** deterministic verification that K and
E_inf strictly increase i→ii→iii at every cell, and the registration
statement: every pre-registered strict-H-improvement criterion that passes
(B2, B3) is realized by a rung with strictly larger counted K and E_inf than
the rung whose floor it beats. This is bookkeeping by construction of the
ladder and is reported as such — its empirical content is that the counted
payments are real, observable quantities (PQ1), not that the coincidence is
derived.

**B5 — falsifier non-occurrence (gate 3):**

- **B5-a (falsifier-grade, the §11 event):** at every cell,
  \(\overline{CE}_i \ge \overline{CE^*_{lin}} - \delta\). A violation would
  be a readout AT linear K/E achieving uncertainty below the linear class's
  exact floor — an admissible instance of the §11 example falsifier /
  `FALSIFIABILITY_CONDITIONS.md` §4 observation 2, and a counterexample
  event for RGT Conjecture 6's empirical shadow. Predicted **NOT** to occur.
- **B5-b (within-ladder cheap-beats-expensive):** at every cell, no rung
  with both smaller K and smaller E_inf has seed-mean CE more than \(\delta\)
  **below** a costlier rung's. Possible only through optimization failure of
  the costlier trained rung; pre-declared interpretation: such an event is
  an optimization observation, not a class-level falsifier, but it is
  reported in full if it occurs. Predicted not to occur at the gap cells.
- An anti-pattern self-audit against `FALSIFIABILITY_CONDITIONS.md` §6.1–6.4
  is written in RESULTS.md regardless of outcomes.

**PQ checklist (gates 2–3 jointly):** PQ1 — §4 operational definitions used
as stated. PQ2 — CV > 20% across the grid of cell means for H, K, E_inf,
E_train, wall-clock. PQ3 — H proxy is the task's own held-out label
cross-entropy, referent declared (§1), validated against the exact analytic
floor (B1) and the exact Bayes value 0 (B2-a). PQ4 — identical units
(bits/sample, FLOPs, params), identical data streams and evaluation sets
across all rungs within a cell; the only varied quantities are the
pre-registered \(d_z\), seed, and rung. PQ5 — the §4 declaration: no
\(\lambda_{var}\) used, no \(\mathcal R\)-descent claim made.

**Secondary, descriptive only (pre-declared to avoid selective reporting):**
\(\overline{CE}_{ii}\) at \(d_z \in \{8, 12\}\) and per-cell rung-(ii)
init-selection diagnostics are reported with no gate attached; wall-clock
per rung reported direction-only.

## 7. Amendment policy

Fixed seeds; all parameters in §3. No parameter changes after the first run.
If the design proves mismeasured for **proxy-quality reasons only**, one
amendment is permitted, with an explicit dated amendment note stating what
changed and why, made **before** looking at any H comparison, and never to
move a result toward passing. As of commit time, no amendment has been made
and no experimental sample has been drawn.

## 8. Outputs

The run writes `results.csv` (one row per seed × \(d_z\) × rung, all raw
measurements and diagnostics) and `results_summary.json` (cell means/stds/SEs
and PQ2 CVs). RESULTS.md will report every criterion above as PASS / FAIL /
INCONCLUSIVE with measured values beside these pre-registered thresholds,
the PQ1–PQ5 checklist with evidence, the §6 anti-pattern self-audit, and any
surprises. All raw files are committed. `THEOREM_TRACKS.md` Track 4 status
notes are updated additively (dated), and the one-line cross-reference in
`READOUT_GEOMETRY_THEOREM.md` §5.2 is completed, only as the criteria
support.
