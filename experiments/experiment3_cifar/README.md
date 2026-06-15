# Experiment 3 — CIFAR-10 masking at scale

Maps to **Section "Experiment 3: masking at scale"** (`sec:exp3`,
`sec:exp3-cannibal`) and **Figures 3–5** (`fig:exp3-masking`,
`fig:exp3-stimuli`, `fig:exp3-cannibal`) of the paper.

CIFAR-10 CNNs trained with a controlled spurious color cue at four
coupling strengths (rho in {0.7, 0.9, 0.95, 0.99}), two arms
(cue-reliant "naive" vs. cue-decorrelated "lawful"), ten seeds each = 80
training cells. Readout measurements use the pinned `fourq` battery
under both a gated (decorrelate) shift and a descriptive (anticorrelate)
shift. The registered pass/fail criteria are met; one descriptive
prediction fails and exposes cue cannibalization.

## Files

| File | Role |
| --- | --- |
| `PREREGISTRATION.md` | committed **before any run** (with one dated, pre-unblinding amendment its registered policy permits) |
| `PRERUN_NOTE_2026-06-11.md` | pre-run note |
| `spurious_cifar.py` | **pinned construction module** (SHA-256 `ab8f6e61…9050ba`, asserted before import by both the analysis and the figure scripts). Defines the controlled spurious color cue. **Do not modify.** |
| `train_run.py` | training script (one of the 80 committed training cells per invocation; GPU) |
| `run_batch.sh` | batch driver for the 80 cells |
| `analyze_masking.py` | once-only analysis script, committed pre-run (estimator selftest, all measurements, pass/fail; CPU). Asserts the SHA-256 of `fourq.py` and `spurious_cifar.py`. |
| `results.csv` | raw per-row results (800 rows) |
| `results_rows.json`, `results_summary.json`, `machine_summary.json` | structured results |
| `RESULTS.md` | results writeup with pass/fail evaluation |

## Reproduce

The 80 training runs need a GPU; the once-only analysis is CPU-bound.

```sh
python3 train_run.py --arm naive|lawful --rho 0.7|0.9|0.95|0.99 --seed 0..9
python3 analyze_masking.py --run
```

`analyze_masking.py` imports `fourq` from a sibling `../lib` directory
(the monorepo layout). In this repository `fourq.py` lives at the
repository-root `lib/`; to re-run the analysis, make it reachable, e.g.

```sh
ln -s ../../lib experiments/lib       # from the repository root
# or:  PYTHONPATH=$(pwd)/lib python3 experiments/experiment3_cifar/analyze_masking.py --run
```

The script's internal SHA-256 assertion on `fourq.py` (`8b66bc7d…`) and
on `spurious_cifar.py` (`ab8f6e61…`) passes against the copies here.

Environment: PyTorch 2.12.0 (CUDA 13.0) for training; scikit-learn 1.9.0
for the readout measurements (settings serialized into every result
row); Python 3.12.3, NumPy 2.4.6.

## Provenance

Original location: `mlr-proof-program/experiments/expB_masking_real/`
(campaign label "Experiment B"). Pre-registration `d26850e…`, amendment
`dc7b1fd…`, results `23a901…`, raw artifacts `478df16…`; see
`../../evidence/`.
