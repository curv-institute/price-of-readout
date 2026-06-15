# Results: Track 3 Acceptance Gates 2–3 (Lawful Compression Toy)

**Date:** 2026-06-09
**Pre-registration:** `PREREGISTRATION.md`, committed before the run in
`fd72496` ("experiments: pre-register track 3 gates 2-3 (lawful compression
toy)"). **No amendments were made** (§7 of the pre-registration); no
parameter was changed after the first run; the experiment was run exactly
once.
**Raw data:** `results.csv` (110 rows: 10 seeds × {d_n, arm}),
`results_summary.json`, `analytic_predictions.json`. Total runtime 69.3 s.
All 110 readout fits converged (max final gradient inf-norm 2.3e-17; zero
step-halvings occurred anywhere, so measured train FLOPs equal the
deterministic F0 exactly).

**Headline:** every pre-registered criterion passed. Gates 2 and 3 are met
under their pre-registered thresholds; the §11 example falsifier was not
observed (G4 criterion passed).

> **Terminology clarification (2026-06-10, additive — the frozen record
> below is unchanged).** The quantities labeled \(\hat H\) in this document
> are **held-out cross-entropies** of trained logistic readouts. In
> distribution they are valid estimates of the declared
> readout-constrained referent \(H_{\mathcal A}(Y \mid \cdot)\) (and, for
> the X/S arms, of \(H(Y \mid \cdot)\)) per the G2-c proxy validation
> against analytic ground truth. The **post-shift values of the N arm**
> (\(\hat H_{shift} \approx 2.35\)–\(2.43\) bits) are **frozen-readout
> transfer cross-entropies** — the cross-entropy under the shifted
> distribution \(Q\) of a readout fit under \(P\) — **not conditional
> entropies** (they exceed the 1-bit binary entropy bound, which a
> conditional entropy cannot). The sign-flip shift leaves all marginals
> and \(I(N; Y)\) essentially unchanged, so \(H(Y \mid N)\) under \(Q\)
> equals its in-distribution value and a readout retrained under \(Q\)
> would land back at the ID numbers: the measured ~1.7-bit degradation is
> deployment brittleness of the \(P\)-fit interface (transfer risk), which
> reframes — and does not weaken — the masking finding. Paper-side
> notation: \(CE_Q(r_P)\), defined in
> `papers/lawful-compression-readout-geometry/main.tex` §2; source-of-truth
> correction trail in `READOUT_GEOMETRY_THEOREM.md` §4.3 and
> `THEOREM_TRACKS.md` (2026-06-10 notes).
>
> **Final-notation mapping (2026-06-10, tag `paper-lcr-v3-draft`; per
> `MLR_FORMAL_DEFINITIONS.md` §2-bis):** the \(\hat H_{ID}\) column is
> \(\widehat{CE}_{ID}\), a licensed estimate of \(CE^{\star}_{lin}(P)\)
> (G2-c proxy validation; equal to \(H(Y \mid \cdot)\) for the X/S arms by
> the logistic-linear construction); the \(\hat H_{shift}\) column is
> \(CE_Q(r_P)\), never an entropy.

---

## 1. Measured cell means (10 seeds; ± is cross-seed std)

H is held-out cross-entropy of the trained logistic readout, bits/sample
(the declared readout-constrained proxy; for X/S arms it equals \(H(Y\mid Z)\)
by construction).

| \(d_n\) | arm | K | \(\hat H_{ID}\) (bits) | \(\hat H_{shift}\) (bits) | train FLOPs | infer FLOPs | wall train (mean s) |
|---|---|---|---|---|---|---|---|
| 0 | X = S | 9 | 0.66625 ± 0.00239 | 0.66604 ± 0.00264 | 1.2375e8 | 3.00e6 | 0.070 |
| 8 | X | 25 | 0.66734 ± 0.00279 | 0.66818 ± 0.00337 | 5.5380e8 | 6.20e6 | 0.084 |
| 8 | S | 9 | 0.66718 ± 0.00278 | 0.66707 ± 0.00251 | 1.2375e8 | 3.00e6 | 0.069 |
| 8 | N | 17 | 0.67971 ± 0.00247 | **2.34627 ± 0.01123** | 2.9877e8 | 4.60e6 | 0.076 |
| 64 | X | 137 | 0.66697 ± 0.00215 | 0.67319 ± 0.01340 | 1.2530e10 | 2.86e7 | 0.255 |
| 64 | S | 9 | 0.66608 ± 0.00219 | 0.66611 ± 0.00257 | 1.2375e8 | 3.00e6 | 0.069 |
| 64 | N | 129 | 0.66845 ± 0.00216 | **2.39959 ± 0.01485** | 1.1154e10 | 2.70e7 | 0.229 |
| 256 | X | 521 | 0.67005 ± 0.00241 | 0.69293 ± 0.02461 | 1.7291e11 | 1.054e8 | 1.969 |
| 256 | S | 9 | 0.66610 ± 0.00223 | 0.66794 ± 0.00167 | 1.2375e8 | 3.00e6 | 0.065 |
| 256 | N | 513 | 0.67039 ± 0.00242 | **2.42587 ± 0.01032** | 1.6768e11 | 1.038e8 | 1.968 |

Analytic ground truth (pre-registered §5): \(H(Y\mid S) = H(Y\mid X) =
0.66654\) bits; \(H_{\mathcal A}(Y\mid N) = 0.67885 / 0.66813 / 0.66694\)
bits and predicted post-shift CE \(= 2.3414 / 2.4047 / 2.4118\) bits at
\(d_n = 8/64/256\).

## 2. Pre-registered criteria: PASS/FAIL

\(\delta = 0.010\) bits (pre-registered). All values are seed-means over all
10 seeds; no run, seed, or condition is omitted.

### Gate 2

**G2-a — lawful H preservation, out-of-sample: \(|\overline{\hat H}_{ID}(X) - \overline{\hat H}_{ID}(S)| < 0.010\) at every \(d_n\)**

| \(d_n\) | measured diff | threshold | verdict |
|---|---|---|---|
| 0 | +0.00000 | < 0.010 | **PASS** |
| 8 | +0.00015 | < 0.010 | **PASS** |
| 64 | +0.00089 | < 0.010 | **PASS** |
| 256 | +0.00395 | < 0.010 | **PASS** |

(The +0.0039 at \(d_n=256\) matches the pre-registered MLE excess-risk
estimate \(d/(2 n_{train} \ln 2) = 0.0038\) bits almost exactly.)

**G2-b — K and E strictly decrease X→S by the constructed factors (\(d_n > 0\))**

| \(d_n\) | \(K_X > K_S\) | F_train X vs S (ratio) | F in [F0, 1.5·F0] | F_infer X > S | wall median X > S (s) | verdict |
|---|---|---|---|---|---|---|
| 8 | 25 > 9 (×2.78, as constructed) | 5.5380e8 > 1.2375e8 (×4.5) | yes (= F0 exactly, both arms) | 6.20e6 > 3.00e6 | 0.088 > 0.071 | **PASS** |
| 64 | 137 > 9 (×15.2) | 1.2530e10 > 1.2375e8 (×101) | yes (= F0 exactly) | 2.86e7 > 3.00e6 | 0.299 > 0.071 | **PASS** |
| 256 | 521 > 9 (×57.9) | 1.7291e11 > 1.2375e8 (×1397) | yes (= F0 exactly) | 1.054e8 > 3.00e6 | 2.161 > 0.072 | **PASS** |

Measured train FLOPs equal the exact formula-defined F0 at every cell
(zero step-halvings; `results.csv` column `halvings` is 0 everywhere).
Disclosure: a first evaluation script compared measured FLOPs against the
5-significant-digit *rounded* F0 values displayed in the pre-registration
table and flagged \(d_n = 64, 256\) as below-band by < 1 part in 1e4; the
pre-registration defines F0 by the stated formula (§4, §5), and against the
exact formula values the measured FLOPs are exactly in band (ratio
1.000000). Both checks are reported here; nothing else changed.

**G2-c — proxy validation against analytic ground truth: \(|\overline{\hat H}_{ID} - H_{analytic}| < 0.010\) bits per cell**

| \(d_n\) | X | S | N | verdict |
|---|---|---|---|---|
| 0 | −0.00029 | −0.00029 | — | **PASS** |
| 8 | +0.00080 | +0.00065 | +0.00086 | **PASS** |
| 64 | +0.00044 | −0.00046 | +0.00032 | **PASS** |
| 256 | +0.00351 | −0.00044 | +0.00345 | **PASS** |

Worst deviation 0.0035 bits (X/N arms at \(d_n = 256\)), consistent with the
known finite-sample excess-risk bias; an order of magnitude inside tolerance.
This is the pre-registered PQ-evidence that the H proxy tracks the
generative-model ground truth.

**G2-d — PQ1–PQ5 checklist:** see §3 below. **PASS.**

**Gate 2 verdict: MET** (all four sub-criteria passed; K/E reduced by the
constructed factors at task-relevant H constant within 0.004 bits,
out-of-sample, against analytically validated proxies).

### Gate 3

**G3-a — lawful invariance under shift: \(|\overline{\hat H}_{shift}(S) - \overline{\hat H}_{ID}(S)| < 0.010\)**

| \(d_n\) | measured diff | verdict |
|---|---|---|
| 0 | −0.00021 | **PASS** |
| 8 | −0.00012 | **PASS** |
| 64 | +0.00003 | **PASS** |
| 256 | +0.00184 | **PASS** |

**G3-b — unlawful degradation under shift toward the analytic prediction**

| \(d_n\) | measured increase | pre-reg threshold | predicted CE | measured CE | agreement (< 0.15) | verdict |
|---|---|---|---|---|---|---|
| 8 | 1.6666 | ≥ 0.8313 | 2.3414 | 2.3463 | +0.0048 | **PASS** |
| 64 | 1.7311 | ≥ 0.8683 | 2.4047 | 2.3996 | −0.0051 | **PASS** |
| 256 | 1.7555 | ≥ 0.8724 | 2.4118 | 2.4259 | +0.0140 | **PASS** |

The measured post-shift cross-entropies land within 0.014 bits of the
analytic predictions computed (and committed) before the run — the
degradation is not merely "large", it is quantitatively the predicted
number.

**Gate 3 verdict: MET** (lawful arm invariant within 0.002 bits; unlawful
arm degrades by ~1.7 bits, double the pre-registered margin, and matches the
pre-registered analytic prediction to ≤ 0.014 bits).

### Gate 4 contribution

**G4 — §11 example falsifier not observed:** at every \(d_n > 0\), the
insufficient arm never reduces task-relevant uncertainty below the
sufficient arm (threshold: not lower by more than \(\delta\)):

| \(d_n\) | \(\overline{\hat H}_{ID}(N) - \overline{\hat H}_{ID}(S)\) | \(\overline{\hat H}_{shift}(N) - \overline{\hat H}_{shift}(S)\) | verdict |
|---|---|---|---|
| 8 | +0.01253 | +1.67921 | **PASS** |
| 64 | +0.00237 | +1.73347 | **PASS** |
| 256 | +0.00429 | +1.75793 | **PASS** |

No compression in this experiment reduced task-relevant uncertainty without
structure, energy, or sufficient information. Falsifier 1 (`FALSIFIABILITY_
CONDITIONS.md` §3) not observed; falsifier 2 not observed (the provably
non-sufficient map paid +1.7 bits under shift); falsifier 3 not observed
(the constructed sufficient statistic never increased task-relevant H
relative to raw X — G2-a). The anti-pattern self-audit is §4 below.

---

## 3. PQ1–PQ5 checklist (G2-d), with evidence

- **PQ1 (observable operational meaning): PASS.** K = readout parameter
  count + input dimension (stated convention, deterministic). E = analytic
  FLOPs by the stated per-iteration formula (verified: measured = formula
  value exactly, zero halvings) **and** wall-clock seconds (reported; noisy
  on this shared box, used for direction only as pre-registered). H =
  held-out cross-entropy in bits of a trained logistic readout — an
  observable, decision-relevant quantity.
- **PQ2 (CV > 20% across test conditions): PASS.** Computed over the grid
  of cell means as pre-registered (`results_summary.json` `_PQ2_CV`):
  H 66.9%, E-FLOPs 204%, E-wall-clock 166%, K 158%. All > 20%.
- **PQ3 (task-relevance): PASS.** The H proxy is the task's own held-out
  label cross-entropy — the classification instance of task-relevant entropy
  (`MLR_FORMAL_DEFINITIONS.md` §2). Validated against the analytic
  \(H(Y\mid\cdot)\) of the generative model to ≤ 0.0035 bits (G2-c).
- **PQ4 (cross-regime consistency): PASS.** Identical units (bits/sample,
  FLOPs, seconds), identical readout class, training protocol, sample sizes,
  and seed streams across all arms and both evaluation regimes; the only
  varied quantities are the pre-registered \(d_n\), arm, and \(\rho\).
- **PQ5 (\(\lambda_{var}\) discipline): PASS.** No \(\mathcal R\)-objective
  was optimized and no \(\lambda_{var}\) value was used or tuned anywhere.
  The Corollary-4 accounting consequence is reported in the pre-registered
  sign-robust form: measured \(\Delta K < 0\), \(\Delta E < 0\) (G2-b) and
  \(\Delta H_{\mathcal T} = 0\) within \(\delta\) (G2-a) give
  \(\Delta\mathcal R = \Delta K + \Delta H + \lambda_{var}\Delta E < 0\) for
  **every** \(\lambda_{var} \ge 0\). \(\lambda_{var}\) entered as a declared
  symbol only; no identification with any other lambda-like object was made.

## 4. Anti-pattern self-audit (`FALSIFIABILITY_CONDITIONS.md` §6)

- **§6.1 in-sample fits — clean.** Every H measurement is on held-out data
  (fresh 100k-sample test sets per seed); gate 3 is additionally
  out-of-distribution (sign-flipped S–N coupling never seen in training).
  No in-sample quantity appears in any criterion.
- **§6.2 symbol reuse without derivation — clean.** No lambda-like quantity
  was substituted for \(\lambda_{var}\); the only \(\lambda_{var}\) statement
  is the sign-robust quantifier over all \(\lambda_{var} \ge 0\) (PQ5 above).
  H, K, E each carry their stated operational definitions; no cross-layer
  identification was asserted.
- **§6.3 tautological identities — clean.** H (held-out cross-entropy on
  random labels realized from the generative model) is measured independently
  of K (a parameter count) and E (an operation count); none is an algebraic
  function of another. The analytic predictions were computed from the
  generative model by quadrature, independently of the experimental pipeline
  (no training, no sampling), and committed before the run.
- **§6.4 selective reporting — clean.** All 110 runs (10 seeds × 11
  arm-conditions) are in `results.csv` and enter the seed-means; no seed,
  arm, or \(d_n\) was dropped; the pre-declared descriptive observable
  (X-arm shift drift) is reported below even though no gate attaches to it;
  the one evaluation-script discrepancy encountered (FLOPs rounding band,
  G2-b) is disclosed rather than silently corrected.

## 5. Honest discussion / surprises

1. **No amendments were needed or made.** The single-amendment policy was
   not exercised.
2. **The analytic predictions were sharp.** Pre-registered post-shift CE
   predictions (2.3414 / 2.4047 / 2.4118 bits) were hit to within
   0.005–0.014 bits by the trained readouts. The pre-registered excess-risk
   bias estimate (0.0038 bits at \(d = 260\)) appeared in the data almost
   exactly (+0.0035 to +0.0040). The toy behaved as the theorem documents
   said it must.
3. **Pre-declared descriptive observation (no gate):** the raw X arm itself
   degrades slightly under shift — \(\hat H_{shift}(X) - \hat H_{ID}(X)\) =
   +0.0008 / +0.0062 / +0.0229 bits at \(d_n = 8/64/256\), with large
   cross-seed variance (e.g. ±0.025 at \(d_n=256\)). Mechanism as
   anticipated in the pre-registration: the MLE places small noise weight on
   the nuisance direction that is near-collinear with the signal
   in-distribution, and that direction flips under shift. Noteworthy because
   it makes the lawful arm \(\phi(X) = S\) *more* shift-stable than raw X at
   large \(d_n\) (+0.0018 vs +0.0229 bits): discarding the nuisance is not
   only cheaper, it removes a fragility. This is an observation of this toy,
   not a claimed theorem.
4. **In-distribution masking confirmed quantitatively.** The unlawful arm's
   in-distribution H excess over lawful (+0.0125 / +0.0024 / +0.0043 bits)
   shrinks toward proxy resolution as \(d_n\) grows (analytic: 0.0123 /
   0.0016 / 0.0004), while its shift degradation stays ~1.7 bits — the
   masking phenomenon that makes the shift protocol mandatory
   (`FALSIFIABILITY_CONDITIONS.md` §3, assumption 4) is visible in one
   table. As pre-registered, no measured ID strict-increase claim is made at
   \(d_n \ge 64\) (below \(\delta\)); the measured +0.0125 at \(d_n = 8\)
   does exceed \(\delta\) and matches the analytic 0.0123.
5. **Scope.** This is one constructed toy in the exact regime of Theorems
   1–2 (binary Y, logistic task, Gaussian signal/nuisance, well-specified
   readout). It demonstrates that the program's gate discipline can be
   executed end-to-end with admissible proxies; it does not establish
   anything beyond the construction's scope, and the Track 3 claim remains
   bounded by the theorem's stated assumptions.

## 6. Gate verdicts

| Gate | Verdict | Basis |
|---|---|---|
| 2 | **MET** | G2-a/b/c/d all passed under pre-registered thresholds |
| 3 | **MET** | G3-a/b passed; degradation matched pre-registered analytic prediction |
| 4 | **MET within this construction's scope** | §3 falsifiers stated (pre-registration §1, referent mapping §7.3) and survived one admissible out-of-sample/shift test; §11 example falsifier not observed; self-audit §4. Survival is cumulative — continued exposure in future experiments, not a closed book. |
