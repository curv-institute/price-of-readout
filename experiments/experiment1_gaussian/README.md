# Experiment 1 — Gaussian lawful compression

Maps to **Section "Experiment 1: analytic calibration on a constructed
Gaussian channel"** (`sec:exp1`) and **Figure 1** (`fig:exp1-shift`) of
the paper.

A constructed Gaussian channel with closed-form, quadrature-grade
analytic predictions. The experiment checks that trained readouts on
sampled data land on the committed analytic numbers out-of-sample, and
exhibits the frozen-readout shift effect under a cue-breaking shift.

## Files

| File | Role |
| --- | --- |
| `PREREGISTRATION.md` | committed **before any run** (construction, fixed parameters, seed streams, proxy conventions, pass/fail thresholds, amendment policy). SHA-256 pinned by the timestamped evidence manifest. |
| `analytic_predictions.json` | committed analytic predictions (Gauss–Hermite quadrature) |
| `lawful_compression_experiment.py` | experiment script |
| `results.csv` | raw per-seed results (110 rows) |
| `results_summary.json` | aggregated summary |
| `RESULTS.md` | results writeup with pass/fail evaluation |

## Reproduce

```sh
python3 lawful_compression_experiment.py --analytic   # committed analytic predictions
python3 lawful_compression_experiment.py              # full pre-registered run; writes results.csv
```

Deterministic; all randomness flows through declared seed streams.
Python 3.12.3, NumPy 2.4.6 (the only third-party dependency).

## Provenance

Original location: `mlr-proof-program/experiments/track3_gate2/`
(internal label "track 3 / gate 2"). Pre-registration commit
`3b05461…`, results commit `341851…`; see `../../evidence/`.
