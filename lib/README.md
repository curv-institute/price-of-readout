# `lib/` — the readout-measurement instrument (`fourq.py`)

`fourq.py` is the shared measurement instrument used by **Experiment 3**
(`../experiments/experiment3_cifar/`). All of that experiment's
task-uncertainty measurements flow through this library, so the measured
quantities cannot drift in definition between conditions.

SHA-256 of `fourq.py`: `8b66bc7d07c2f0d1cc11180a5c629b156c014c413cb1b5644a6e1539abdd255a`
(asserted at run time by `analyze_masking.py` before any measurement).

Dependencies: Python standard library + NumPy + scikit-learn.

## The four evaluation quantities

The instrument is organized around the four evaluation quantities of the
paper (Section "Four evaluation quantities"). Logs are base 2; the label
space is finite; there is no side channel (Y -> X -> Z).

1. **H(Y | Z)** — task-conditional entropy; a property of the pair
   (Y, Z) alone. <= log2|Y|.
2. **H_A(Y | Z) := inf over r in A of H(Y | r(Z))** —
   readout-constrained task uncertainty over a hard deterministic class
   A. >= H(Y | Z) (data-processing). Per the proved no-order-either-way
   relation it must never be substituted for (3) or vice versa; **not
   measured by this library**.
3. **CE*_A(P) := inf over q in A_soft of E_P[-log q(Y | Z)]** — the
   population soft-readout optimum; the in-distribution referent of every
   cross-entropy measurement here. >= H(Y | Z), with equality when the
   class contains/approaches Bayes.
4. **CE_Q(r_P) := E_Q[-log q_{r_P}(Y | Z)]** — frozen-readout transfer
   cross-entropy; a property of the *fitted interface*, not of (Y, Z)
   under Q. It is not an entropy and obeys no entropy bound.

Every result row records which of these quantities it estimates; the
library refuses to emit an undeclared table.

## Fitting and measurement

Soft readouts are fit with scikit-learn `LogisticRegression` (lbfgs) on
the declared features plus per-class bias; for a binary label space this
is the binomial fit. Cross-entropy is reported in bits on a held-out
split. The library also provides a noise-floor estimate and the
estimator selftest used by Experiment 3's once-only analysis.

## Tests

`test_fourq.py` regenerates the analytic anchors and checks per-seed
agreement against committed predictions. It reads Experiment 1's
committed results by the monorepo-relative path `../track3_gate2/`; in
this repository that experiment lives at
`../experiments/experiment1_gaussian/`, so that path must be made
reachable before running. The test needs scikit-learn installed.

## Provenance

`fourq.py` and `test_fourq.py` are copied byte-for-byte from
`mlr-proof-program/experiments/lib/`; content SHA-256 unchanged. See
`../evidence/PROVENANCE.md`.
