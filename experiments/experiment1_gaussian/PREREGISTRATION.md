# Pre-Registration: Track 3 Acceptance Gates 2–3 (Lawful Compression Toy)

**Date:** 2026-06-09
**Track:** 3 — Lawful-Compression Theorem and Sufficient-Statistic Exception (`THEOREM_TRACKS.md`)
**Construction target:** `GRIT_ITERATION_001.md` §7 Experiment A (signal/nuisance sufficient-statistic toy)
**Theorem referents:** `LAWFUL_COMPRESSION_THEOREM.md` Theorems 1–2, Proposition 3, Remark 5.2, §7.3 falsifier mapping
**Falsifiers in scope:** `FALSIFIABILITY_CONDITIONS.md` §3 (and the §11 example falsifier of `MLR_FORMAL_DEFINITIONS.md`), audited against the §6 anti-patterns

```text
Discipline: this document is committed BEFORE any experimental run.
Every pass/fail criterion below is fixed now. A failed gate honestly
reported is an acceptable outcome; a massaged pass is not.
```

**Commit discipline.** This pre-registration and the experiment script
(`lawful_compression_experiment.py`) are committed and pushed before the
experiment is run. The analytic predictions in §5 were computed before this
commit by running `lawful_compression_experiment.py --analytic`, which draws
no data and trains no readout — it evaluates the generative model by
Gauss-Hermite quadrature. The experimental grid (§4) has not been run at
commit time.

---

## 1. What is being tested

Track 3 gates 2 and 3 (`THEOREM_TRACKS.md`):

- **Gate 2** — lawful-compression toy with **constructed** (not inferred)
  sufficiency, showing K/E reduction at constant task-relevant H,
  out-of-sample, PQ1–PQ5 satisfied.
- **Gate 3** — unlawful-compression control showing measurable H increase
  under controlled distribution shift.
- **Gate 4 (partial)** — the §11 example falsifier (uncertainty reduced
  without structure, energy, or sufficient information) must remain
  unobserved under admissible proxies; anti-pattern self-audit written.

Per `LAWFUL_COMPRESSION_THEOREM.md` §7.3 item 3, we **declare the H-proxy
referent**: the proxy measures **readout-constrained** uncertainty
\(H_{\mathcal A}(Y \mid r(Z))\) for the logistic readout class
\(\mathcal A_{logit}\) (held-out cross-entropy of a trained logistic readout,
in bits). For the X and S arms the task is logistic-linear in S **by
construction**, so \(\mathcal A_{logit}\) contains the Bayes readout and
\(H_{\mathcal A}(Y\mid Z) = H(Y \mid Z)\) exactly — the lawful-vs-raw
comparison is therefore not confounded by Proposition-3 / Remark-5.2 effects.
For the N arm, both referents are computed analytically in §5; their gap is
< 3e-6 bits at every d_n, so the distinction is immaterial in this
construction (this computation is itself part of the proxy-quality evidence).

## 2. Generative model — sufficiency as a theorem of the construction

All draws i.i.d. across samples; all randomness from `numpy.random.default_rng`.

- Signal: \(S \sim \mathcal N(0, I_{d_s})\), \(d_s = 4\).
- Task: \(Y \mid S \sim \mathrm{Bernoulli}(\sigma(w \cdot S))\) with fixed
  \(w = (1,1,1,1)\), \(\lVert w \rVert = 2\); \(\sigma\) the logistic function.
- Signal scalar: \(u = w \cdot S / \lVert w \rVert \sim \mathcal N(0,1)\);
  the task logit is \(2u\).
- Nuisance: \(N_j = \rho\, u + \sqrt{1-\rho^2}\, \varepsilon_j\),
  \(\varepsilon_j \sim \mathcal N(0,1)\) i.i.d., \(j = 1..d_n\),
  \(d_n \in \{0, 8, 64, 256\}\).
- In-distribution coupling \(\rho_{train} = 0.9\); **shift protocol** flips
  the sign: \(\rho_{shift} = -0.9\). The marginals of \(S\), of \(N\), and of
  \(Y\) are unchanged under shift (Gaussian, covariance depends on \(\rho^2\));
  only the S–N coupling direction changes. This probes exactly the
  positive-probability set \(A\) of `LAWFUL_COMPRESSION_THEOREM.md` §1.4 that
  in-distribution evaluation fails to weight (falsifier 2 referent, §7.3).

**Arms (representations):**

| Arm | \(\phi(X)\) | dim | Status |
|---|---|---|---|
| X | identity, \((S, N)\) | \(4 + d_n\) | baseline |
| S | \(\phi_{lawful}(X) = S\) | 4 | **sufficient by construction**: \(p(Y\mid X) = \sigma(w\cdot S)\) is a function of \(S\) alone, so Definition 1.1 holds pointwise (Theorem 1 applies) |
| N | \(\phi_{unlawful}(X) = N\) | \(d_n\) (only \(d_n > 0\)) | **insufficient by construction**: \(\operatorname{Var}(u \mid N) = 1 - v_\mu(d_n) > 0\) (exact values §5), and \(\sigma\) is strictly monotone, so \(p(Y\mid X) = \sigma(2u)\) is non-degenerate given \(N\); hence \(p(\cdot\mid X) \ne p(\cdot\mid N)\) on a positive-probability set and Theorem 2 applies. What it destroys: the residual signal component \(u - \mathbb E[u\mid N]\), of variance \(1 - v_\mu > 0\). |

Insufficiency of the N arm is **masked in-distribution** at large \(d_n\)
(the analytic strict increase \(H(Y\mid N) - H(Y\mid S)\) is 0.0123 bits at
\(d_n=8\) but only 0.0016 / 0.0004 bits at \(d_n=64/256\), below the proxy
resolution \(\delta\) of §6). We therefore do **not** claim a measured
in-distribution strict increase at \(d_n \ge 64\); the strict-increase
direction is probed by the shift protocol, exactly as
`FALSIFIABILITY_CONDITIONS.md` §3 (assumption 4) prescribes.

## 3. Fixed parameters (no changes after first run)

| Parameter | Value |
|---|---|
| \(d_s\), \(w\) | 4, \((1,1,1,1)\) |
| \(d_n\) grid | \(\{0, 8, 64, 256\}\) |
| \(\rho_{train}\), \(\rho_{shift}\) | \(0.9\), \(-0.9\) |
| seeds | 0–9 (10 seeds; rng streams `default_rng(10000+seed)`) |
| \(n_{train}\) | 50,000 |
| \(n_{test}\) | 100,000 per evaluation condition (ID and shift, fresh draws per seed) |
| readout | logistic regression (weights + bias), damped Newton/IRLS, **25 iterations fixed**, ridge \(10^{-8}\) (numerical damping), deterministic step-halving line search; final gradient inf-norm recorded as convergence diagnostic |
| quadrature | 64-node probabilists' Gauss-Hermite (both levels) |

## 4. Proxies (PQ1 operational definitions)

- **H proxy** — held-out cross-entropy (bits/sample) of the trained logistic
  readout on fresh test data, \(\hat H(Y \mid r(Z))\). Task-relevant by
  construction (PQ3: it is the task's own label uncertainty). All H
  measurements are out-of-sample; the shift measurements use a test set drawn
  with \(\rho_{shift}\) while the readout was trained at \(\rho_{train}\).
- **K proxy** — \(K = (d+1) + d = 2d + 1\): readout parameter count
  (weights + bias) plus input dimension. Stated convention; deterministic.
- **E proxy** — (a) analytic FLOP count, stated convention: per IRLS
  iteration \(2nD^2 + 7nD + 14n + 4D + 3D^2 + \lfloor 2D^3/3 \rfloor\) with
  \(D = d+1\), plus one loss-evaluation \((2nD + 6n + 2D)\) per step-halving;
  inference \(2 n_{test} D + 5 n_{test}\) per evaluation set; **and**
  (b) wall-clock seconds (train and inference separately). Wall-clock on this
  shared box is noisy; it is reported for direction only, never as a gate's
  sole evidence.
- **\(\lambda_{var}\) (PQ5)** — no \(\mathcal R\)-objective is optimized and
  no \(\lambda_{var}\) value is used in any measurement. Gates 2–3 concern
  the signs of \(\Delta K\), \(\Delta E\), \(\Delta H\) separately. The
  Corollary-4 accounting consequence is reported in the
  \(\lambda_{var}\)-free, sign-robust form: if \(\Delta H = 0\) (within
  \(\delta\)), \(\Delta K < 0\), \(\Delta E < 0\), then
  \(\Delta\mathcal R < 0\) for **every** \(\lambda_{var} \ge 0\). No value is
  derived, asserted, or tuned; \(\lambda_{var}\) enters as a declared symbol
  only. This is reported explicitly to satisfy PQ5's no-silent-tuning clause.

## 5. Analytic ground truth (computed from the generative model, pre-commit)

From `analytic_predictions.json` (Gauss-Hermite quadrature; reproducible via
`--analytic`). \(v_\mu(d_n) = \rho^2 / (\rho^2 + (1-\rho^2)/d_n)\) is the
variance of \(\mathbb E[u \mid N]\).

**Sufficient arms (exact, all \(d_n\)):**

\[
H(Y \mid S) = H(Y \mid X) = \mathbb E_{u}\big[h_2(\sigma(2u))\big] = 0.66654 \text{ bits.}
\]

**Unlawful arm (per \(d_n\)):**

| \(d_n\) | \(v_\mu\) | \(H(Y\mid N)\) | \(H_{\mathcal A_{logit}}(Y\mid N)\) | optimal logit scale \(c^*\) | predicted post-shift CE | predicted shift increase |
|---|---|---|---|---|---|---|
| 8 | 0.97151 | 0.67885 | 0.67885 | 1.9584 | **2.3414** | **1.6626** |
| 64 | 0.99635 | 0.66813 | 0.66813 | 1.9945 | **2.4047** | **1.7366** |
| 256 | 0.99908 | 0.66694 | 0.66694 | 1.9986 | **2.4118** | **1.7449** |

(Post-shift CE = cross-entropy of the in-distribution-optimal logistic
readout \(\sigma(c^* m)\), \(m = \mathbb E[u\mid N]\), against the shifted
truth \(p_{shift}(Y{=}1\mid N) = 1 - g(m)\), which follows from the sign flip
\(m \mapsto -m\) and the symmetry \(g(-m) = 1 - g(m)\).)

**Known estimation bias (informs tolerances):** the asymptotic excess
log-loss of a well-specified logistic MLE is \(\approx d/(2 n_{train} \ln 2)\)
bits — at most \(261/(2 \cdot 50000 \cdot \ln 2) = 0.0038\) bits (X arm,
\(d_n = 256\)). Test-set noise: per-seed CE standard error
\(\approx 0.002\) bits at \(n_{test} = 10^5\); seed-mean (10 seeds) standard
error \(\lesssim 0.001\) bits.

**Deterministic K and FLOP predictions** (FLOPs at zero extra halvings, F0;
inference FLOPs cover both evaluation sets):

| \(d_n\) | arm | dim | K | F0 train | F infer |
|---|---|---|---|---|---|
| 0 | X = S | 4 | 9 | 1.2375e8 | 3.00e6 |
| 8 | X | 12 | 25 | 5.5380e8 | 6.20e6 |
| 8 | S | 4 | 9 | 1.2375e8 | 3.00e6 |
| 8 | N | 8 | 17 | 2.9877e8 | 4.60e6 |
| 64 | X | 68 | 137 | 1.2530e10 | 2.86e7 |
| 64 | S | 4 | 9 | 1.2375e8 | 3.00e6 |
| 64 | N | 64 | 129 | 1.1154e10 | 2.70e7 |
| 256 | X | 260 | 521 | 1.7291e11 | 1.054e8 |
| 256 | S | 4 | 9 | 1.2375e8 | 3.00e6 |
| 256 | N | 256 | 513 | 1.6768e11 | 1.038e8 |

Constructed K reduction factors X→S: 25/9 = 2.78 (\(d_n{=}8\)),
137/9 = 15.2 (\(d_n{=}64\)), 521/9 = 57.9 (\(d_n{=}256\)).

## 6. Pre-registered pass/fail criteria

Tolerance \(\delta = 0.010\) bits on seed-mean comparisons, justified by the
bias and noise estimates of §5 (max systematic bias 0.0038 bits + seed-mean
noise \(\lesssim 0.001\) bits, with > 2x headroom). All criteria evaluate
**seed-means over all 10 seeds** (no seed selection — §6.4 anti-pattern).

**Gate 2:**

- **G2-a (lawful H preservation, out-of-sample):** at every \(d_n\),
  \(\lvert \overline{\hat H}_{ID}(X) - \overline{\hat H}_{ID}(S) \rvert < \delta\).
- **G2-b (K and E strictly decrease by constructed factors):** for every
  \(d_n > 0\): \(K_S = 9 < K_X = 2(4{+}d_n){+}1\) (exact table §5);
  measured train FLOPs satisfy \(F0 \le F_{train} \le 1.5 \times F0\) per arm
  and \(F_{train}(S) < F_{train}(X)\); \(F_{infer}(S) < F_{infer}(X)\)
  (deterministic); median wall-clock (train+infer) of arm S < arm X
  (direction only; wall-clock is noisy on this shared box).
- **G2-c (proxy validation against ground truth):** for every (arm, \(d_n\))
  cell in-distribution,
  \(\lvert \overline{\hat H}_{ID} - H_{analytic} \rvert < 0.010\) bits, where
  \(H_{analytic} = 0.66654\) for X/S arms and
  \(H_{\mathcal A_{logit}}(Y\mid N)\) from §5 for the N arm.
- **G2-d (PQ checklist):** PQ1 — operational definitions of §4 used as
  stated. PQ2 — coefficient of variation > 20% across the grid of cell means
  for: H (all (arm, \(d_n\), {ID, shift}) cells), E-FLOPs, E-wall-clock, and
  K (across (arm, \(d_n\)) cells). PQ3 — H proxy is the task's own held-out
  label cross-entropy. PQ4 — identical units (bits/sample, FLOPs, seconds),
  identical readout class and training protocol across all compared arms and
  regimes. PQ5 — the \(\lambda_{var}\)-free sign-robust statement of §4
  reported; no tuning anywhere.

**Gate 3:**

- **G3-a (lawful invariance under shift):** at every \(d_n\),
  \(\lvert \overline{\hat H}_{shift}(S) - \overline{\hat H}_{ID}(S) \rvert < \delta\).
- **G3-b (unlawful degradation toward the analytic prediction):** at every
  \(d_n > 0\),
  \(\overline{\hat H}_{shift}(N) - \overline{\hat H}_{ID}(N) \ge \max(0.25,\ 0.5 \times \text{predicted increase})\)
  bits (predicted increases: 1.6626 / 1.7366 / 1.7449, so the binding
  thresholds are 0.8313 / 0.8683 / 0.8724 bits); **secondary agreement
  check:** \(\lvert \overline{\hat H}_{shift}(N) - CE_{shift,analytic} \rvert < 0.15\) bits.

**Gate 4 (this experiment's contribution):**

- **G4 (§11 example falsifier not observed):** at every \(d_n > 0\), the
  insufficient arm does not beat the sufficient arm:
  \(\overline{\hat H}_{ID}(N) \ge \overline{\hat H}_{ID}(S) - \delta\) and
  \(\overline{\hat H}_{shift}(N) \ge \overline{\hat H}_{shift}(S) - \delta\).
  (The S arm beating the X arm on K/E at equal H is the lawful exception of
  Theorem 1, not the falsifier — the falsifier requires uncertainty reduction
  **without** sufficiency.) An anti-pattern self-audit against
  `FALSIFIABILITY_CONDITIONS.md` §6.1–§6.4 is written in RESULTS.md.

**Secondary, descriptive only (pre-declared to avoid selective reporting):**
\(\hat H_{shift}(X)\) is measured and reported at every \(d_n\). Expected to
exceed \(\hat H_{ID}(X)\) slightly (the X-arm MLE places small noise weight
on the near-collinear N̄ direction, which flips under shift); no gate
attaches to it.

## 7. Amendment policy

Fixed seeds; all parameters in §3. No parameter changes after the first run.
If the design proves mismeasured for **proxy-quality reasons only** (e.g.
PQ2 CV unexpectedly below 20%, readout non-convergence), one amendment is
permitted, with an explicit dated amendment note stating what changed and
why, made **before** looking at any H comparison, and never to move a result
toward passing. As of commit time, no amendment has been made.

## 8. Outputs

The run writes `results.csv` (one row per seed × \(d_n\) × arm, all raw
measurements including convergence diagnostics) and `results_summary.json`
(cell means/stds and PQ2 CVs). RESULTS.md will report every criterion above
as PASS/FAIL with measured values beside these pre-registered thresholds,
the PQ1–PQ5 checklist with evidence, the §6 anti-pattern self-audit, and any
surprises. All raw files are committed.
