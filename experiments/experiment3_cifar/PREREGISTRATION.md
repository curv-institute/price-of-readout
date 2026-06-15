# Pre-Registration: Experiment B — Masking at Scale (CIFAR-10, Constructed Spurious Tint)

**Date of registration (the commit timestamp of this file IS the registration):** 2026-06-11
**Experiment id:** `expB-masking-real-v1`
**Campaign:** Curv experiment campaign (`/gfs/curv-campaign/README.md`); design ruling
`queue/decisions/answered/phase2-pilot-B-prereg-design.md` (fable-5 main loop, 2026-06-11) is
binding throughout. Pilot evidence: `/gfs/curv-campaign/artifacts/pilot-B-report.md`.
**Status:** committed pre-registration. A review gate with the program owner (Jason Miller)
follows this commit and precedes any run. Pre-run amendments are permitted under §5.9 and
**must be dated**. **One §5.9 amendment has now been made — §5.9.A (2026-06-11): §5.1 PASS
redefined as CE\*-difference recovery, per ruling
`queue/decisions/answered/runB-expB-s51-selftest-excess.md`. The amendment budget is now
exhausted (0 remaining); it was made strictly pre-unblinding.** Nothing in this document depends
on any arm-comparison or shift quantity — none has been computed; the pilot was clean-arm-only (§8).

```text
Discipline (Track-3 apparatus, replicated). Every threshold, formula, gate,
and decision rule below is fixed at this commit, BEFORE any naive arm is
trained and BEFORE any shift set is constructed for measurement. A failed
gate honestly reported is an acceptable outcome; a massaged pass is not.
The analysis script runs EXACTLY ONCE on the committed protocol (§6). One
pre-unblinding amendment is permitted, for proxy-quality reasons only (§5.9).
```

**Vocabulary (binding).** The four task-uncertainty quantities of
`../../MLR_FORMAL_DEFINITIONS.md` §2-bis (propagated from
`papers/lawful-compression-readout-geometry/main.tex` §2, tag `paper-lcr-v3-draft`);
logs base 2; finite alphabet \(\mathcal Y\), \(|\mathcal Y| = K = 10\); no side channel
(\(Y \to X \to Z\)):

1. \(H(Y\mid Z)\) — task-conditional entropy; information property of \((Y,Z)\); \(\le \log_2 10 = 3.3219\) bits.
2. \(H_{\mathcal A}(Y\mid Z)\) — readout-constrained, hard classes. **Not measured here**; never substituted for (3) (no bound either way — proven in the paper).
3. \(CE^{\star}_{\mathcal A}(P) = \inf_{q\in\mathcal A_{soft}} \mathbb E_P[-\log q(Y\mid Z)]\) — population soft-readout optimum; \(\ge H(Y\mid Z)\) (the bridge). The population referent of every in-distribution CE here, **licensed only via §5.1**.
4. \(CE_Q(r_P) = \mathbb E_Q[-\log q_{r_P}(Y\mid Z)]\) — frozen-readout transfer cross-entropy; property of the fitted interface, **not an entropy** (obeys no entropy bound; the §6.5 tell: it may exceed \(\log_2 10\)).

Every measured number carries its quantity-and-distribution declaration (the §2-bis RULE;
anti-pattern referent: QUANTITY CONFLATION, `FALSIFIABILITY_CONDITIONS.md` §6.5). The
measurement library (`../lib/fourq.py`) enforces the RULE mechanically and is **pinned by
sha256** in §6.

---

## 1. The question and hypotheses

> **Primary question.** Does the Track-3 masking signature — an insufficient (spurious-cue-reliant)
> representation that is **indistinguishable from, or better than, a sufficient one in-distribution**
> but whose frozen interface collapses under a coupling-breaking shift, with the collapse attributable
> to **interface brittleness (\(\Delta_{transfer}\)), not information loss (\(\Delta_{info}\))** —
> reproduce on real images with **learned** representations (a CNN trained end-to-end on CIFAR-10
> with a constructed class-indexed spurious tint)?

**Arms (the §2 construction supplies the images; the CNN learns the representation):**

| Arm | Training data | Status |
|---|---|---|
| **NAIVE** | CIFAR-10 train with tint coupled to the label at \(\rho \in \{0.7, 0.9, 0.95, 0.99\}\) | spurious-by-construction: the tint encodes the indicator \(s\), and \(s\) is informative about \(y\) only through the train-time coupling |
| **LAWFUL** | CIFAR-10 train with tint **decorrelated** (\(\rho_{\mathrm{eff}} = 1/K = 0.1\), s independent of y) | **sufficiency-by-construction for the semantic task**: the tint is present with identical marginal statistics but carries zero label information, so the learned representation cannot exploit it; the two arms differ ONLY in the \(s \leftrightarrow y\) coupling |

**Pre-registered per-\(\rho\) predictions (gated versions in §5.3):**

1. **Masking ID (G-B1):** the naive arm's reliance on the cue is invisible in-distribution —
   naive \(\widehat{CE}_{ID}\) is **not materially worse** than lawful \(\widehat{CE}_{ID}\) at any
   \(\rho\). At high \(\rho\) the naive arm may be *better* ID (the cue carries real ID information;
   cue-only anchor §1.1) — that is the masking phenomenon amplified, not a failure.
2. **Naive brittleness under decorrelate (G-B2):** the naive arm's frozen readout suffers an
   elevated \(CE_Q(r_P)\) under the decorrelate shift; the Q-refit recovers most of it (the image
   content is intact under shift), so **\(\Delta_{transfer} \gg \Delta_{info}\)**. \(\Delta_{info}\)
   for the naive arm is predicted small-positive (bounded by the cue's genuine ID information value,
   which is genuinely absent under Q), not zero.
3. **Lawful flat everywhere (G-B3):** the lawful arm's \(\Delta_{info}\) and \(\Delta_{transfer}\)
   are both within noise at every \(\rho\) and under both shifts.
4. **Masking-curve direction (G-B4):** naive \(\Delta_{transfer}\) is non-decreasing in \(\rho\)
   (stronger train-time coupling → heavier learned cue reliance → bigger frozen-transfer collapse).

### 1.1 Point predictions — the cue-only semi-analytic anchors (framing committed)

The construction admits **closed forms** for an idealised **cue-only** readout (one that reads
\(s\) perfectly — licensed by the injectivity of the tint in \(s\), §2 — and is Bayes-calibrated
to the train coupling). These were computed from the generative model before this commit
(`/gfs/curv-campaign/artifacts/semianalytic_predictions.json`, regenerated and verified by the
pilot; they are **construction math, not measurements of any trained system**). Table (bits):

| \(\rho\) | \(H(Y)\) | cue-only \(CE^{\star}_{id} = H(Y\mid s)\) | cue-only \(CE_Q\)[decorrelate] | cue-only \(CE_Q\)[anticorrelate] | elev[dec] | elev[anti] |
|---|---|---|---|---|---|---|
| 0.70 | 3.3219 | 1.8323 | 4.4677 | 3.5892 | 2.6354 | 1.7569 |
| 0.90 | 3.3219 | 0.7860 | **5.8579** | **5.8579** | 5.0719 | 5.0719 |
| 0.95 | 3.3219 | 0.4449 | 6.7501 | 7.1210 | 6.3052 | 6.6761 |
| 0.99 | 3.3219 | 0.1125 | 8.8339 | 9.7158 | 8.7214 | 9.6033 |

**Declared relationship to the measured quantities (honest framing, committed):** these anchors
describe the **cue-readable component** of the channel, not the CNN's behaviour. A trained naive
CNN sees image content *and* cue and divides its reliance between them in an unknown proportion —
that proportion is precisely the thing under test. Therefore:

- The anchors are **NOT exact point predictions of any learned-feature quantity.** No gate
  requires a measured number to equal a table entry.
- **Banded prediction (committed):** at \(\rho \ge 0.9\), the naive arm's frozen-transfer
  elevation under decorrelate (\(\Delta_{transfer}\)) is predicted to be **of the order of the
  cue-only elevation** — specifically in the band \([0.5\ \text{bits},\ \text{elev[dec]}(\rho)]\),
  whose lower edge is the G-B2 gate (10% of the smallest gated cue-only elevation, §5.3) and whose
  upper edge is the cue-only envelope. The upper edge holds under the calibrated-Bayes-cue-reader
  idealisation; a *miscalibrated (overconfident) learned readout could exceed it* — exceedance is
  reported and discussed, never gated.
- **Secondary quantitative comparison (descriptive, committed):** the fraction-of-envelope
  statistic \(f(\rho) = \Delta_{transfer}^{naive}(\rho)\, /\, \text{elev[dec]}(\rho)\) is reported
  per \(\rho\) with bootstrap CIs, against the banded prediction \(0.1 \le f \le 1\) at
  \(\rho \ge 0.9\). It quantifies how much of the cue-readable ceiling the learned system realises.
- **ID anchors:** cue-only \(CE^{\star}_{id}(\rho)\) bounds what the cue route alone can deliver
  ID. At \(\rho = 0.99\) it (0.1125 bits) is **below** the pilot lawful-arm floor (0.3787 bits),
  so a naive-better-than-lawful ID gap of up to ≈ 0.27 bits is the construction's own expectation
  there — pre-declared so it cannot be mistaken for an anomaly (G-B1 band, §5.3).
- **Exact pass/fail gates are effect-direction + threshold based** (§5.3); the table enters gates
  only through the declared 10%-of-envelope scaling of G-B2 and the \(\rho=0.9\) coincidence
  check (§5.4).

**Conclusion asymmetry (committed).** A G-B2 PASS shows the learned naive interface is brittle in
the predicted direction; a G-B2 FAIL (small \(\Delta_{transfer}\)) would mean the CNN largely
ignored the cue at that coupling — an honest negative for masking-at-scale at this tint strength,
reported as such, NOT salvaged by post-hoc strength increases (one nuisance strength is the
ruling's binding choice).

**Non-claims.** No claim about natural (non-constructed) spurious correlations; no claim beyond
this architecture, tint strength, dataset, and readout class; no MLR/GRIT/RIFT stack claim; no
claim that any soft-readout CE bounds a hard-class \(H_{\mathcal A}\).

## 2. Construction (pinned)

**Module (pinned, in this directory):** `spurious_cifar.py`,
**sha256 = `ab8f6e61d54be4ea13b23aee326aabccb2f272c4bfffa4431ace45b70c9050ba`** — verified
identical to the pilot's verified module (pilot report §1). All train/test/shift image sets are
built by this module and no other code.

**Data.** CIFAR-10 from `/vault/datasets/cifar10/cifar-10-batches-py` (5 train batches →
**50,000 train**, `test_batch` → **10,000 test**), CHW float in [0,1]. Labels exactly uniform
(5,000/class train; 1,000/class test), so \(H(Y) = \log_2 10\) exactly.

**Coupling (the indicator draw).** Per image, indicator \(s \in \{0..9\}\):
\(s = y\) w.p. \(\rho\); \(s \sim \mathrm{Unif}(\{0..9\}\setminus\{y\})\) w.p. \(1-\rho\), so
\(P(s{=}y) = \rho\) exactly and the off-diagonal is uniform \((1-\rho)/9\)
(`draw_indicator`). Lawful arm: \(\rho_{\mathrm{eff}} = 1/K = 0.1\), the no-coupling fixed point.
Pilot empirical check: \(P(s{=}y) = 0.9030\) at \(\rho = 0.9\) (6,000-sample slice).

**Tint injection (FIXED strength 0.20, per ruling Q1 — a recorded constant, not a swept axis).**
`inject_tint`: \(X' = \mathrm{clip}_{[0,1]}\big(X + 0.20 \cdot \mathrm{PALETTE}[s]\big)\), the
per-class RGB offset broadcast over all pixels. PALETTE = 10 maximally-separated hues
(HSV \(h = i/10\), s=v=1 → RGB, then **row-mean subtracted** so the tint is a chromatic shift,
not a brightness scalar; pilot-verified row-mean max \(|\cdot|\) = 3.7e-17). The 10-hue table
(class → base zero-mean RGB, and the ×0.20 applied offset):

| s | class | hue | base RGB (zero-mean) | offset ×0.20 |
|---|---|---|---|---|
| 0 | airplane   | 0.0 | [ 0.6667 −0.3333 −0.3333] | [ 0.1333 −0.0667 −0.0667] |
| 1 | automobile | 0.1 | [ 0.4667  0.0667 −0.5333] | [ 0.0933  0.0133 −0.1067] |
| 2 | bird       | 0.2 | [ 0.2000  0.4000 −0.6000] | [ 0.0400  0.0800 −0.1200] |
| 3 | cat        | 0.3 | [−0.2000  0.6000 −0.4000] | [−0.0400  0.1200 −0.0800] |
| 4 | deer       | 0.4 | [−0.4667  0.5333 −0.0667] | [−0.0933  0.1067 −0.0133] |
| 5 | dog        | 0.5 | [−0.6667  0.3333  0.3333] | [−0.1333  0.0667  0.0667] |
| 6 | frog       | 0.6 | [−0.4667 −0.0667  0.5333] | [−0.0933 −0.0133  0.1067] |
| 7 | horse      | 0.7 | [−0.2000 −0.4000  0.6000] | [−0.0400 −0.0800  0.1200] |
| 8 | ship       | 0.8 | [ 0.2000 −0.6000  0.4000] | [ 0.0400 −0.1200  0.0800] |
| 9 | truck      | 0.9 | [ 0.4667 −0.5333  0.0667] | [ 0.0933 −0.1067  0.0133] |

**Licensing assumption (stated per ruling Q1):** the 10 offsets are distinct, so the tint is
**injective in \(s\)** — a cue reader can in principle recover \(s\) exactly. This is the
assumption licensing the §1.1 cue-only closed forms.

**Shift definitions** (`draw_indicator_shift`; coupling broken at evaluation time, image content
untouched):

- **PRIMARY = decorrelate** (ruling Q3; **all gates attach here**): \(s \sim \mathrm{Unif}(\{0..9\})\)
  independent of \(y\) — the canonical "cue uninformative at deployment", the
  information-preserving analogue of Track-3's shift.
- **SECONDARY = anticorrelate** (descriptive, **no gate**, reported in full — no selective
  reporting): \(s = y\) only w.p. \(1-\rho\), else uniform-other — the strict sign-flip analogue.

**\(\rho\) grid (ruling Q2):** \(\{0.7, 0.9, 0.95, 0.99\}\); lawful arm at
\(\rho_{\mathrm{eff}} = 0.1\) in every cell.

**The \(\rho = 0.9\) coincidence (pre-registered internal-consistency check, ruling Q3).** At
\(\rho = 0.9\), the decorrelate and anticorrelate shift distributions are **identical**
(\(1-\rho = 0.1\) = uniform-effective); the cue-only \(CE_Q\) predictions coincide to 8.9e-16.
The two measured \(CE_Q(r_P)\) values at \(\rho = 0.9\) (same frozen readout, two independently
drawn shift sets from the same distribution) must agree within \(\delta\) — a free internal
validity test. **Disagreement is a pipeline-bug detector, not a finding** (§5.4).

**Committed rng streams (via `SpuriousConfig.seed`; the module uses `default_rng(10000 + cfg.seed)`).**
For run seed \(\sigma \in\) §3's list: train-set construction `cfg.seed = σ`; ID test-set
indicator `cfg.seed = 1000 + σ`; decorrelate shift set `cfg.seed = 2000 + σ`; anticorrelate
shift set `cfg.seed = 3000 + σ`. All four sets per (cell, seed) use the same underlying images
(train: 50,000 train images; the other three: the same 10,000 test images, re-tinted per draw) —
so CE differences cannot come from image-content sampling, only from the coupling. **Disclosed:**
the lawful arm's constructed dataset is identical across the four \(\rho\) cells at fixed seed
(its coupling is \(\rho\)-independent and the stream depends only on σ); the four lawful
trainings per seed are therefore replicate trainings of the same data, and their CE spread is
reported as a free measurement of training-run nondeterminism (descriptive).

## 3. Models and training (pinned from the pilot smoke)

**Architecture: `CNN9`** exactly as in the pilot smoke (`train_smoke.py`, pilot report §2):
9 Conv-BN-ReLU blocks, channels 64,64 / pool / 128,128 / pool / 256,256 / pool / 256
(32→16→8→4), global average pool, linear classifier. **Param count 1,739,210 (~1.74 M).**
**Penultimate features \(Z\): the 256-d post-GAP vector**, extracted float32; non-finite counts
recorded (pilot: 0).

**Training recipe (all pinned; no changes after first run):**

| Parameter | Value |
|---|---|
| optimizer | SGD, momentum 0.9, nesterov, weight_decay 5e-4 |
| LR schedule | OneCycleLR, max_lr = 0.1 |
| epochs / batch | 30 / 256 |
| loss | cross-entropy; abort on non-finite loss |
| augmentation | random crop (reflect pad 4) + horizontal flip, batch-level, as implemented in the pilot smoke |
| input normalisation | per-channel mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616), applied **after** tint injection |
| torch seed | `torch.manual_seed(σ)`, `np.random.seed(σ)` per run |
| device | one GB10 per run; cuDNN `LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu` shim (BINDING on the GB10 nodes, pilot infra finding) |
| sanity metric | standard (untinted) CIFAR test accuracy, recorded per run (pipeline sanity only; no gate; pilot smoke: 0.9240) |

**Grid: 4 \(\rho\) × 10 seeds × 2 arms = 80 training runs** (full paired grid; pilot projection
7.1 GPU-h ≈ 3.6 h wall on the two GB10s). **Seeds (enumerated, committed): 0, 1, 2, 3, 4, 5, 6,
7, 8, 9.** No seed selection ever (§6.4 anti-pattern); every gate evaluates seed-means over all
10 seeds.

## 4. Measurements (declaration-complete, the harness battery)

All task-uncertainty numbers are produced by `fourq.battery(...)` (+ `fourq.noise_floor`,
`fourq.class_floors`); every result row carries `quantity` + `distribution` + `not_an_entropy`
and is serialized by the RULE-enforcing writers (which refuse undeclared tables).

**Declared soft readout class \(\mathcal A_{log}\) (a declared class choice).** Multinomial
(softmax-linear) logistic regression on the 256-d penultimate features + per-class bias (sklearn
`LogisticRegression`, lbfgs), **L2 with C = 1.0 (recorded finite), standardize = True**
(train-fold statistics only), `tol = 1e-10`, `max_iter = 10000`; convergence (`n_iter_ < max_iter`)
recorded per fit, any non-converged fit disclosed. **This departs from the harness default
C = 1e8 for the documented convergence reason (pilot report §3, binding recommendation): the
effectively-unregularized MLE is non-convergent on the near-separable CNN features (pilot:
C=1e8 → converged=False, meaningless CE 4.64; C=1.0 → converged in ~725 iters, CE 0.40) and
would silently poison every CE in the run.** The override is passed explicitly and recorded
verbatim in every result's settings, exactly as the harness contract licenses
(`lib/README.md` §2). Quantity declarations per §2-bis are unchanged by this class choice.
Readout K/E declaration: \(D = 256\), \(K_{readout} = |\mathcal Y|(D{+}1) = 2570\).

**Per-cell evaluation sets (committed split).** The 10,000 test images are split once,
stratified 50/50 with `train_test_split(stratify=y, test_size=0.5, random_state=0)`:
**half A (5,000) = readout-fit pool; half B (5,000) = evaluation pool.** Per (ρ, seed, arm) cell:
\(Z^{A}_{ID}\) (A under the cell's ID coupling) is the P-fit set; \(Z^{B}_{ID}\) the P-held-out
set; \(Z^{B}_{dec}\), \(Z^{B}_{anti}\) (B re-tinted under each shift draw) the Q sets. The frozen
readout is never evaluated on re-tinted versions of images it was fit on. Q-refit:
stratified **5-fold** cross-fit within the Q set (`k_refit_folds = 5`,
`StratifiedKFold(shuffle=True, random_state=0)`, the battery contract).

**The battery per (ρ, seed, arm) — run for shift = decorrelate (gated) and, identically,
shift = anticorrelate (descriptive):**

| Measurement | Protocol | Declared quantity |
|---|---|---|
| **\(\widehat{CE}_{ID}\)** | fit \(r_P\) on \(Z^A_{ID}\); held-out CE on \(Z^B_{ID}\) | `raw_heldout_CE` (per §5.9.A: an **absolute** CE, bias ≈ 0.371 bits disclosed; no CE\*-point license) — it enters gates only through licensed **differences** |
| **\(CE_Q(r_P)\)** | frozen \(r_P\) on the full Q set (5,000) | \(CE_Q(r_P)\) — `not_an_entropy=True` (enforced); never called entropy; definitionally exact, no license needed |
| **\(\widehat{CE}_{Q\text{-refit}}\)** | 5-fold cross-fit of \(\mathcal A_{log}\) within Q | `raw_heldout_CE` (per §5.9.A: absolute, bias-disclosed; enters gates only through licensed differences) |
| **\(\Delta_{info}\)** | \(\widehat{CE}_{Q\text{-refit}} - \widehat{CE}_{ID}\) | estimates \(CE^{\star}(Q)-CE^{\star}(P)\): information-accessibility change under shift — **licensed as a CE\* DIFFERENCE by §5.9.A** (common-mode bias cancels) |
| **\(\Delta_{transfer}\)** | \(CE_Q(r_P) - \widehat{CE}_{Q\text{-refit}}\) | transfer risk of the frozen interface; deployment property, **never** entropy / "information loss"; **licensed as a CE\* DIFFERENCE by §5.9.A** |

**Reference constants** (`fourq.class_floors`, per distribution): \(H(Y) = \log_2 10 = 3.3219\)
bits (uniform; the cardinality bound), `majority_class_ce_bits`, `perfect_prediction_floor = 0`.
Reference only; **no gate attaches**. Unseen-label counts (`n_unseen_labels`, CLIP_EPS = 1e-12)
expected 0 and reported regardless.

**Uncertainty.** Per-cell: 1,000-resample nonparametric bootstrap of \(CE_Q(r_P)\) over the Q
samples (masking-curve CIs, descriptive). Cross-seed: per (ρ, arm, quantity), seed-mean and
\(SE_{seed} = \mathrm{std}_{ddof=1}(\text{10 per-seed values})/\sqrt{10}\) — the SE entering
§5.2/§5.5. In-run noise floor: `fourq.noise_floor` (10-fold, 2×SE) recomputed on each lawful
arm's \(Z^B_{ID}\), reported beside the pilot value.

## 5. Thresholds, gates, and policies

### 5.1 Proxy validation — the \(CE^{\star}\) license (§2-bis RULE item 2)

> **AMENDED 2026-06-11 (§5.9.A).** The PASS criterion below — recovery of **absolute** \(CE^{\star}\)
> within τ = 0.05 — was the registered criterion and it **FAILED** at the committed n, D, K (the
> pre-registered §5.9 trigger fired). It is **superseded** by the CE\*-**DIFFERENCE** recovery
> criterion of **§5.9.A**, the experiment's one permitted amendment. The text below is retained as
> the registered-then-superseded criterion (history preserved); read it together with §5.9.A,
> which governs. The licensing split (differences licensed, absolutes `raw_heldout_CE` + disclosed
> ≈ 0.371-bit bias) is defined in §5.9.A.

No analytic \(H(Y\mid Z)\) exists for learned CNN features, so the license is established at the
estimator level: the committed analysis script runs, as a selftest, the full \(\mathcal A_{log}\)
pipeline (C = 1.0, standardize = True, the §4 split sizes: 5,000 fit / 5,000 eval, \(D = 256\),
\(K = 10\)) on synthetic feature sets with **planted softmax-linear posteriors** whose
\(CE^{\star}_{log}\) is analytically known, plus a planted-insufficient control. **PASS iff** each
known \(CE^{\star}\) is recovered within **\(\tau = 0.05\) bits** (half the materiality floor;
the regularization bias of C = 1.0 at \(n = 5{,}000\), \(D = 256\) is expected well inside this —
if not, that is exactly a §5.9 proxy-quality amendment trigger). The `fourq` validation anchors
(`lib/test_fourq.py`: Track-3 regeneration to 6.4e-9 bits ID; analytic closed forms) are the
upstream arithmetic evidence; §5.1 adds recovery at this experiment's \(n, D, K\) **and at the
declared C = 1.0**. License ref recorded in the rows:
`"expB-masking-real-v1 §5.1 planted-posterior PASS"`. **Without PASS**, every \(\widehat{CE}\)
serializes as `raw_heldout_CE` with no population referent and no masking language; the
serializer refuses any relabel. (\(CE_Q(r_P)\) and the gates G-B2/G-B4 that are pure
frozen-transfer comparisons remain defined either way; G-B1/G-B3 require the license.)

### 5.2 Thresholds (committed)

**Materiality floor: 0.10 bits** (≈ 3% of \(H(Y) = 3.3219\); pilot report §3 binding
recommendation). Pilot clean-arm cross-split floor: mean CE 0.3787 bits, **SE = 0.0171,
2×SE = 0.0343 bits** — well-conditioned but below the floor, so a bare 2×SE rule would call a
0.04-bit blip "significant", which is not a meaningful masking effect at this task scale.

\[
\boxed{\ \delta = \max(2 \times SE,\ 0.10\ \text{bits}) = 0.10\ \text{bits}\ }
\]

— the **formula** is committed, not just the number: every gate below uses, per compared cell,
\(\delta_{cell} = \max(2 \times SE_{seed},\ 0.10)\) with \(SE_{seed}\) the realized cross-seed SE
(§4). From the pilot SE the floor binds (\(2 \times 0.0171 = 0.034 < 0.10\)), so the operative
\(\delta\) is expected to be 0.10 bits throughout; if a realized \(SE_{seed}\) is larger, the
formula (not a new number) governs, and §5.5 caps how large it may grow before the gate is
declared underpowered.

### 5.3 Gates (committed) — all on seed-means over the 10 seeds; all shift gates attach to the PRIMARY (decorrelate) shift only

| Gate | Name | Quantity | Criterion (committed) | Type |
|---|---|---|---|---|
| **G-B1** | masking-ID premise | naive vs lawful \(\widehat{CE}_{ID}\), per ρ | **PASS** iff at every ρ: \(\overline{\widehat{CE}}_{ID}^{naive} - \overline{\widehat{CE}}_{ID}^{lawful} \le +\delta\). Declared band \([-\overline{\widehat{CE}}_{ID}^{lawful},\ +\delta]\): the upper edge (= the gate) is the pilot-noise + materiality-floor band of §5.2 — ID evaluation must not expose the naive arm; the lower edge is the construction's hard floor (naive CE ≥ 0, so the gap cannot exceed lawful's own CE). A negative gap is **expected** at ρ = 0.99 (≈ up to 0.27 bits; cue-only ID anchor 0.1125 < pilot lawful floor 0.3787) and reported against the §1.1 anchors — masking amplified, not an anomaly. | pass/fail |
| **G-B2** | naive transfer brittleness | naive \(\Delta_{transfer}\) under decorrelate | **PASS** iff \(\overline{\Delta}_{transfer}^{naive}(\rho) \ge 0.5\) bits at **each** ρ ∈ {0.9, 0.95, 0.99}. **Justification of 0.5:** 10% of the smallest gated cue-only elevation (elev[dec](0.9) = 5.0719 bits) — the gate fires only if the trained system realises at least one-tenth of the cue-readable ceiling at the weakest gated coupling; a deliberately conservative direction-plus-materiality threshold (also 5× the floor), because the CNN's cue-vs-image reliance split is the unknown under test. The same 0.5 minimum (not scaled up) is used at 0.95/0.99; per-ρ envelope comparison is the §1.1 descriptive statistic. ρ = 0.7 is reported, not magnitude-gated (covered by G-B4). | pass/fail |
| **G-B3** | lawful arm flat | lawful \(|\Delta_{info}|\), \(|\Delta_{transfer}|\) | **PASS** iff at every ρ: \(|\overline{\Delta}_{info}^{lawful}| \le \delta\) **and** \(|\overline{\Delta}_{transfer}^{lawful}| \le \delta\) (decorrelate; anticorrelate values reported descriptively beside them). | pass/fail |
| **G-B4** | masking-curve monotonicity | naive \(\overline{\Delta}_{transfer}\) vs ρ | **PASS** iff non-decreasing up to noise across the grid: \(\overline{\Delta}_{transfer}(\rho_{i+1}) \ge \overline{\Delta}_{transfer}(\rho_i) - \delta\) for every adjacent pair in {0.7, 0.9, 0.95, 0.99}. | pass/fail |
| **C-B5** | ρ = 0.9 dec/anti coincidence | \(CE_Q(r_P)\) at ρ = 0.9, both shift sets | consistency check, **not a gate, not a finding**: per arm, \(|\overline{CE}_Q^{dec}(r_P) - \overline{CE}_Q^{anti}(r_P)| \le \delta\) at ρ = 0.9 (identical shift distributions, independent draws). **Violation ⇒ pipeline bug: the analysis HALTS, the bug is found and fixed, and the event is disclosed in RESULTS.md**; it licenses a §5.9-style dated correction note (pipeline repair, not threshold motion). | validity check |

**Pre-registered descriptive separation (no gate):** naive \(\Delta_{transfer} \gg \Delta_{info}\)
at ρ ≥ 0.9 (prediction: ratio ≥ 2; the brittleness is interface-borne, not information-borne).
Reported with both quantities side by side per the §2-bis RULE item 3.

**Verdict composition:** "masking-at-scale REPRODUCED" requires G-B1 ∧ G-B2 ∧ G-B3 ∧ G-B4. Any
partial outcome is reported gate-by-gate, measured-vs-threshold, no aggregation that hides a FAIL.

### 5.4 The coincidence check is a detector, not evidence

C-B5 sits outside the hypothesis structure by design (ruling Q3): the two ρ = 0.9 shift sets are
the same distribution, so agreement carries no information about masking and disagreement carries
no information about the hypotheses — only about the pipeline. It is pre-registered so that it
cannot be repurposed post-hoc in either direction.

### 5.5 Underpowered bands (committed power clause)

If, for any gated cell, the realized \(\delta_{cell} = \max(2 \times SE_{seed}, 0.10) > 0.25\)
bits (half the G-B2 minimum effect), then every gate touching that cell **degrades to
"underpowered — inconclusive"** for that cell (reported as such, never as PASS or FAIL); G-B2's
0.5-bit criterion additionally requires \(\delta_{cell} \le 0.25\) so a pass is never claimed
from a noise-dominated mean. From the pilot SE (0.017 per-split) this clause is not expected to
fire; it is retained for honesty.

### 5.6 PQ1–PQ5 (evaluated in RESULTS.md)

- **PQ1 (operational definitions):** every reported uncertainty quantity is a held-out CE in bits
  of \(\mathcal A_{log}\) (C = 1.0, standardize = True) on the 256-d penultimate features; K =
  \(K_{readout} = 2570\) (+ CNN params 1,739,210 recorded); E = training/readout wall-clock +
  per-run seconds (reported, direction-only — shared boxes are noisy).
- **PQ2 (CV > 20%):** coefficient of variation (`fourq.pq2_cv`) across the grid of cell means
  {ρ} × {arm} × {\(\widehat{CE}_{ID}\), \(CE_Q^{dec}(r_P)\), \(\widehat{CE}_{Q\text{-refit}}\)}.
- **PQ3 (task relevance + declaration):** the proxy is the task's own label CE; every cell
  carries its §2-bis declaration (ID → \(CE^{\star}_{log}(P)\) [licensed §5.1]; frozen →
  \(CE_Q(r_P)\) [never entropy]; refit → \(CE^{\star}_{log}(Q)\) [licensed]).
- **PQ4 (cross-regime consistency):** identical units, splits, readout settings (one C, one
  standardize flag), training recipe across all arms, ρ cells, and distributions; only the
  coupling and the shift draw vary.
- **PQ5 (\(\lambda_{var}\) discipline):** no \(\mathcal R\)-objective optimized; \(\lambda_{var}\)
  appears in no criterion.

### 5.7 Quantity-declaration checklist

Every row carries `quantity`, `distribution`, `fit_distribution`, `not_an_entropy`, `licensed`,
`license_ref`. The serializers refuse: missing declarations; unlicensed \(CE^{\star}\); a declared
"entropy" above \(\log_2 10\) (the §6.5 tell); a frozen-shift row missing `not_an_entropy=True`.
Reproduced and ticked in RESULTS.md.

### 5.8 §6.5 anti-pattern audit plan

RESULTS.md includes the QUANTITY-CONFLATION self-audit (no \(CE_Q(r_P)\) described as \(H\) or
\(H_{\mathcal A}\) anywhere; the > \(\log_2 10\) tell applied to every number — the predicted
naive \(CE_Q(r_P)\) values **will** exceed \(\log_2 10\) and must be handled as quantity (4)
throughout), plus the `FALSIFIABILITY_CONDITIONS.md` §6.1–§6.4 self-audit: no seed selection
(seeds fixed §3), no proxy gaming (C fixed pre-run for a documented convergence reason, not
tuned on results), no post-hoc threshold motion (formulas committed §5.2), no selective
reporting (anticorrelate reported in full; every gate measured-vs-threshold).

### 5.9 Amendment policy (Track-3 style)

Fixed at this commit: the construction pins (§2), the recipe (§3), the readout class incl.
C = 1.0 (§4), the split seeds, the δ formula and floor, the gates and their constants, the §5.1
tolerance, the seed list. **One amendment** is permitted, **before any arm-comparison or shift
quantity is unblinded**, for **proxy-quality reasons only** (e.g. §5.1 selftest fails; readout
non-convergence at the declared settings; PQ2 degeneracy), with a dated amendment note stating
what changed and why, never to move a result toward passing. As of this commit, no amendment has
been made.

### 5.9.A — AMENDMENT 2026-06-11: §5.1 PASS redefined as CE\*-DIFFERENCE recovery (the ONE permitted amendment; none remain)

**This is the single §5.9 amendment this experiment is permitted. After it, no amendment
remains.** It is dated 2026-06-11, made strictly **pre-unblinding** (no arm-comparison or shift
quantity has been computed; the 80-run grid only produces features), and it is **proxy-quality
only** — it fixes the §5.1 license test to validate the estimand the gates consume; it loosens
**no** gate, moves **no** result toward passing.

**Authority.** Ruling `queue/decisions/answered/runB-expB-s51-selftest-excess.md` (fable-5 main
loop, 2026-06-11), **Option 2**, answering escalation
`queue/decisions/done/runB-expB-s51-selftest-excess.md` (mandatory escalation: pre-registration
content / threshold procedure). The original §5.1 text anticipated this exact path: *"the
regularization bias of C = 1.0 at n = 5,000, D = 256 is expected well inside this — **if not,
that is exactly a §5.9 proxy-quality amendment trigger.**"*

**The trigger (preserved FAIL table — history, not superseded by erasure; ruling condition 4).**
The original §5.1 criterion — recover each planted **absolute** CE\*_log within τ = 0.05 bits at
n = 5,000, D = 256, K = 10 — **FAILED**, as the committed selftest ran and reported (also recorded
in `PRERUN_NOTE_2026-06-11.md` and the escalation):

| planted case | CE\* (200k MC) | recovered CEhat | abs err | τ = 0.05 |
|---|---|---|---|---|
| planted-sufficient | 1.52 | 2.07 | +0.54 | **FAIL** |
| planted-sufficient-mild | 2.31 | 2.73 | +0.42 | **FAIL** |
| planted-insufficient-control | 3.32 | 3.72 | +0.39 | **FAIL** |

This is **not a script defect**. It is the intrinsic finite-sample held-out-CE excess of the
K×(D+1) multinomial-logistic MLE, `≈ K_readout/(2 n ln2) = 2570/(2·5000·ln2) ≈ 0.371 bits`,
confirmed by a fixed-n dimension sweep (D = 256 → +0.54; 64 → +0.08; 16 → +0.05; 8 → +0.01,
tracking the analytic `K·(D+1)/(2n ln2)`). At this experiment's n/D ≈ 20, **absolute** 0.05-bit
recovery is unreachable. The superseded criterion and this FAIL table are retained verbatim in
the selftest record (`run_selftest(...)["absolute_recovery_history_PRESERVED"]`).

**The amended criterion (committed).** §5.1 **PASS is redefined as recovery of planted CE\*
DIFFERENCES** within τ = 0.05 bits, at the same matched (n = 5,000, D = 256, K = 10, C = 1.0,
standardize = True). The amended selftest plants ≥ 2 posteriors that share **one fixed
softmax-linear law** (so the feature geometry, the posterior peakedness, and hence the
finite-sample MLE excess are common-mode across members) and moves CE\* **only** via an explicit
label-noise blend `P_eff = (1−λ)·softmax + λ/K`; per-member CE\* and recovered CEhat are averaged
over 6 independent planted draws (the same seed-averaging the gates apply over the 10 seeds).
The committed pairs (ruling condition 1: one ~0.5-bit gate-relevant pair, one near-zero null
pair) and their **as-run results**:

| planted pair | planted ΔCE\* | recovered ΔCEhat | |err| | τ = 0.05 |
|---|---|---|---|---|
| gate-relevant (~0.5 bit; λ = 0.30 vs 0.10) | +0.6305 | +0.6249 | 0.0056 | **PASS** |
| null calibration (~0 bit; λ = 0.25, disjoint seed banks) | −0.0005 | +0.0002 | 0.0007 | **PASS** |

Member excesses are matched (gate pair: +0.592 vs +0.597 bits; null pair: +0.620 vs +0.619),
which is *why* the common-mode bias cancels in the difference. Amended selftest: **PASS** (green).

**Why this is the right estimand (ruling rationale).** Every gate consumes a **difference** of
CEs measured with the same estimator at the same (n, D, K), where the ~0.371-bit common-mode
excess cancels to first order: G-B1 (naive − lawful CEhat_ID gap), G-B2/G-B4 (frozen-transfer
Δ_transfer = CE_Q_frozen − CEhat_Q_refit), G-B3 (lawful |Δ_info|, |Δ_transfer|), C-B5 (dec vs
anti at the same cell). Validating absolute-CE recovery was the wrong estimand; validating
difference recovery is the right one.

**Licensing split (committed; ruling condition 2).** From this amendment forward:
- **DIFFERENCE quantities** — Δ_info, Δ_transfer, and the naive−lawful arm CEhat_ID gaps —
  carry the licensed CE\*-difference framing, citing this amended (difference-recovery) §5.1
  selftest. These are the quantities the gates consume.
- **ABSOLUTE CE values** in RESULTS — CEhat_ID, CEhat_Q_refit per cell — are **downgraded to
  `raw_heldout_CE`** serialization (no CE\*-point license), with the **estimated bias magnitude
  disclosed alongside: ≈ 0.371 bits** (`K_readout/(2n ln2) = 2570/(2·5000·ln2)`), common-mode and
  cancelling in the differences. The analysis script implements this split by running the pinned
  `fourq.battery` UNLICENSED (so absolute rows serialize raw) and re-licensing only the
  difference rows — the pinned harness (sha256 §6) is untouched.

CE_Q(r_P) is definitionally exact (quantity 4) and needs no license either way; G-B2/G-B4 were
always pure frozen-transfer comparisons. No gate's numeric threshold changes; **only the §5.1
PASS definition and the license wording move.**

**Shared-root-cause lesson (ruling item, recorded).** The original §5.1 validated **absolute**
CE\* recovery while every gate consumes **CE\* differences** — a mismatch between the validated
quantity and the estimand the gates actually use. The lesson, committed for this program:
**validate the estimand the gates consume, not a more demanding cousin of it.** The amendment
re-points the proxy-quality test at the difference estimand; the absolute referent is honestly
demoted to `raw_heldout_CE` + disclosed bias rather than over-claimed.

**Amendment budget after this note: 0 remaining.**

## 6. Procedure

1. **Training (80 runs)** dispatched as campaign-queue jobs (`/gfs/curv-campaign/queue`,
   background + exit markers, decoupled from agent lifetimes) **split across the two GB10 nodes
   dgx01 and dgx02** (~3.6 h wall projected). Each run: build the cell's train set (§2 streams) →
   train CNN9 (§3) → extract 256-d \(Z\) for the four evaluation sets (\(Z^A_{ID}, Z^B_{ID},
   Z^B_{dec}, Z^B_{anti}\)) → archive features + training log + sanity accuracy to
   `/gfs/curv-campaign/expB_masking_run/` (large arrays stay on /gfs; not committed).
2. **Readout / refit / bootstrap sweep** (CPU): one §4 battery per (cell × shift) = 160
   batteries + bootstraps, **fanned out over the trinity nodes — astra, orbital, nebula — via
   ssh dispatch** (`ssh jwm@<node>.ojos-betta.ts.net`; ~62 s/cell measured at C = 1.0, pilot §5;
   embarrassingly parallel).
3. **Analysis script `analyze_expB_masking.py`** — committed at the review gate BEFORE any
   full-grid training starts; imports `fourq` by path at the pinned sha256 below and
   `spurious_cifar.py` at its §2 sha256; runs **EXACTLY ONCE** on the committed protocol:
   §5.1 selftest → (PASS) license set → `class_floors` per distribution → lawful noise floors →
   batteries → realized SEs and \(\delta_{cell}\) → C-B5 coincidence check (HALT on violation) →
   gates §5.3 → outputs.
4. **Harness (pinned).** `../lib/fourq.py`
   **sha256 = `8b66bc7d07c2f0d1cc11180a5c629b156c014c413cb1b5644a6e1539abdd255a`**.
5. **Outputs land in this directory:** `results.csv` (one declaration-complete row per
   measurement), `results_summary.json` (cell means/SEs, \(\delta_{cell}\), PQ2 CVs, convergence
   diagnostics, §5.1 selftest record, training-nondeterminism replicate spread), **`RESULTS.md`**
   (every gate measured-vs-threshold, the CE-pair per cell with declarations, the §1.1
   fraction-of-envelope table, C-B5 outcome, PQ1–PQ5 + §5.7 checklist + §5.8 audit, honest
   discussion). Raw feature archives + queue job records referenced by path.

## 7. Power

From the pilot: per-split clean-arm SE = 0.0171 bits (10-fold cross-split on 10,000 features;
the §4 cells use 5,000-sample evaluation halves, so per-cell measurement noise ≈
0.017×√2 ≈ 0.024 bits). With 10 seeds, the cross-seed seed-mean SE is expected at or below
~0.01–0.03 bits unless seed-to-seed training variance dominates (it is measured by the design;
the lawful replicate spread of §2 gives a direct estimate). **Minimum detectable effects vs the
gates:** at the committed δ = 0.10 bits, G-B1/G-B3/G-B4 resolve effects ≥ 0.10 bits — an order
of magnitude below nothing-of-interest and ~25× below the smallest gated brittleness effect;
G-B2's minimum effect (0.5 bits) sits ≥ 5× above the floor and ~20–50× above the projected
seed-mean SE, so all gated effects are detectable with very large margin at 10 seeds (pilot §6:
the binding uncertainty was readout convergence, resolved by the declared C = 1.0, not seed
count). The §5.5 clause covers the residual risk that realized cross-seed variance is
unexpectedly large.

## 8. Blindness accounting (honest disclosure)

**Observed pre-registration (clean-arm only).** The pilot trained **one lawful (decorrelated)
configuration** (seed 0) and computed, on its features: the standard-CIFAR sanity accuracy
(0.9240), the clean-arm noise floor (mean CE 0.3787 bits, SE 0.0171, 2×SE 0.0343), and the
C-convergence comparison (C = 1e8 non-convergent vs C = 1.0 convergent). Consequences, stated
explicitly:

> Because the lawful-arm ID magnitude (~0.38 bits) was observed pre-registration, the lawful-arm
> ID level carries reduced evidential weight as a prediction — it anchors thresholds (the δ
> formula, the G-B1 lower-edge discussion) rather than confirming a blind forecast. The C = 1.0
> readout choice and the 0.10-bit materiality floor were **motivated by these clean-arm pilot
> observations** (a permitted, clean-arm-only design input).

> **No arm comparison or shift quantity has been computed, in any form.** No naive arm has been
> trained; no shift set has been built for measurement; no \(CE_Q(r_P)\), \(\widehat{CE}_{Q\text{-refit}}\),
> \(\Delta_{info}\), \(\Delta_{transfer}\), or naive-vs-lawful number of any kind exists. The
> predecessor agent's aborted, uncommitted working tree was **audited script-by-script by the
> successor** (pilot report §0): every script either builds inputs, trains one clean arm, or
> evaluates the generative model analytically; its one staged measurement artifact (a noise
> floor) was found INVALID (non-convergent readout) and was discarded and recomputed. The §1.1
> cue-only table is closed-form construction math evaluated from the generative model — not a
> measurement of any trained system. The masking conclusions (G-B1–G-B4) are the genuinely
> pre-registered, blind results of this experiment.

## 9. Verdict mapping (skeleton; constants fixed above)

| Outcome | Verdict |
|---|---|
| §5.1 (amended §5.9.A) difference-recovery PASS | CE\*-**difference** license granted: Δ_info, Δ_transfer, arm CEhat_ID gaps carry the CE\*-difference framing; **absolute** CEhat serialize as `raw_heldout_CE` with the ≈ 0.371-bit bias disclosed; full masking-decomposition language on the differences. (This is the realized outcome — selftest green 2026-06-11.) |
| §5.1 (amended §5.9.A) difference-recovery FAIL | would mean even the difference estimand is not recovered at τ: HALT and escalate (no τ tuning); no CE\*-difference license; raw-CE descriptive only; frozen-transfer gates G-B2/G-B4 still evaluable |
| C-B5 violated | pipeline bug: HALT, repair, disclose; no result stands until C-B5 holds |
| G-B1–G-B4 all PASS | Track-3 masking signature **reproduced at scale**: ID-indistinguishable arms, frozen-interface collapse under decorrelate growing with ρ, lawful arm flat — with the \(\Delta_{transfer}\)-vs-\(\Delta_{info}\) decomposition quantifying interface-vs-information |
| G-B2 FAIL (small naive \(\Delta_{transfer}\)) | the CNN largely ignored the cue at strength 0.20 — honest negative for masking-at-scale under this construction; **no** post-hoc strength change (ruling Q1) |
| G-B1 FAIL (naive materially worse ID) | masking premise broken — the insufficiency is ID-visible here; reported as a disanalogy with Track-3 |
| G-B3 FAIL | lawful arm not flat — sufficiency-by-construction did not yield shift-stability; a finding against the lawfulness story itself, reported with priority |
| G-B4 FAIL (non-monotone) | masking-curve direction not reproduced; reported per adjacent pair |
| §5.5 fires | affected cells "underpowered — inconclusive"; remaining gates stand |

Anticorrelate results reported in full beside every decorrelate result (descriptive, no gate).
