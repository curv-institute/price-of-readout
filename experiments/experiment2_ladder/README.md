# Experiment 2 — the readout ladder

Maps to **Section "Experiment 2: the readout ladder"** (`sec:exp2`) and
**Figure 2** (`fig:exp2-ladder`) of the paper.

Measured cross-entropy versus readout load, with exact per-seed
population floors overlaid. The floors are computed by exact enumeration
over a finite atom distribution plus population-level convex
optimization, committed before the run.

## Files

| File | Role |
| --- | --- |
| `PREREGISTRATION.md` | committed **before any run**. SHA-256 pinned by the timestamped evidence manifest. |
| `analytic_predictions.json` | exact per-seed population floors (convex optimization) |
| `readout_geometry_experiment.py` | experiment script |
| `results.csv` | raw per-seed results (120 rows) |
| `results_summary.json` | aggregated summary |
| `RESULTS.md` | results writeup with pass/fail evaluation |

The machine verification of every finite enumeration used by this
experiment's theory (the XOR minimal-E edge at E=12, the LTF censuses,
etc.) lives in `../../theory_verification/verify_xor_min_e.py`.

## Reproduce

```sh
python3 readout_geometry_experiment.py --analytic     # exact per-seed population floors
python3 readout_geometry_experiment.py                # full pre-registered run; writes results.csv
```

Deterministic. Python 3.12.3, NumPy 2.4.6.

## Provenance

Original location: `mlr-proof-program/experiments/track4_gateB/`
(internal label "track 4 / experiment B"). Pre-registration commit
`6a3645…`, results commit `1dca82…`; see `../../evidence/`.
