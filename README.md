# The Price of Readout — reproduction repository

Public reproduction materials for

> J.W. Miller (CURV Institute). **The Price of Readout: Exact Interface
> Floors and Frozen-Readout Shift Evaluation.** 2026.

The built paper is at [`paper/main.pdf`](paper/main.pdf). This
repository contains **everything the paper's Reproducibility section
cites and nothing else**: the three pre-registered experiments
(pre-registrations, experiment scripts, raw per-seed results), the
machine-verification scripts for every finite enumeration claimed in the
paper, the shared measurement instrument, the paper sources and figure
generator, and the externally timestamped evidence attestations.

## The git dates here are not the evidence — the OpenTimestamps attestations are

This repository was created with **fresh git history** so that it reads
as a clean paper artifact. **Its commit dates carry no evidentiary
weight.** The claim that matters — that each experiment's
pre-registration was committed *before any sample was drawn* — is
established by:

1. **Recorded commit hashes** in the repositories where those commits
   actually live (a commit hash is a content address over tree, parents,
   and dates), and
2. **OpenTimestamps attestations** (Bitcoin-anchored, hash-only): the
   `.ots` files under [`evidence/`](evidence/).

A reviewer can verify the entire chain **today**. Start with
[`evidence/PROVENANCE.md`](evidence/PROVENANCE.md).

## Layout

```
paper/                      the paper
  main.tex, main.pdf
  figures/  make_figures.py + the five figure PDFs
experiments/
  experiment1_gaussian/     Exp 1 — constructed Gaussian channel  (paper §"Experiment 1")
  experiment2_ladder/       Exp 2 — finite-atom readout ladder    (paper §"Experiment 2")
  experiment3_cifar/        Exp 3 — CIFAR-10 spurious-cue masking  (paper §"Experiment 3")
  experimentY_pythia/       Pythia-410m external replication (paper Discussion §"External
                            replication on a pretrained language model"). POST-HOC, NOT
                            pre-registered — see its POSTHOC_NOTE.md.
lib/                        fourq.py — the shared readout-measurement instrument (+ tests)
theory_verification/        re-runs every finite enumeration in the paper, with hard assertions
evidence/                   the timestamped evidence manifests + .ots attestations; PROVENANCE.md
```

Each subdirectory has its own README mapping files to paper sections.

## Section → file map

| Paper element | Files |
| --- | --- |
| Experiment 1 (Gaussian), Figure "exp1-shift" | `experiments/experiment1_gaussian/` |
| Experiment 2 (readout ladder), Figure "exp2-ladder" | `experiments/experiment2_ladder/` |
| Experiment 3 (CIFAR masking), Figures "exp3-masking/stimuli/cannibal" | `experiments/experiment3_cifar/` |
| Discussion §"External replication on a pretrained language model" (`sec:pythia`), Tables `tab:pythia-quant` / `tab:pythia-ood` — **post-hoc, not pre-registered** | `experiments/experimentY_pythia/` |
| Propositions on the XOR floor / minimal-E edge / decoders, Appendix enumerations | `theory_verification/verify_xor_min_e.py` |
| Superposition-frontier lemmas/propositions | `theory_verification/superposition_min_e.py` |
| The readout-measurement battery used by Exp 3 | `lib/fourq.py` (+ `lib/test_fourq.py`) |
| The paper itself | `paper/main.tex`, `paper/main.pdf`, `paper/figures/` |

## Verifying the evidence chain

```sh
cd evidence

# 1. The combined paper anchor (created 2026-06-15 for this repo):
ots upgrade PRICE_OF_READOUT_EVIDENCE.txt.ots      # pull the on-chain proof
ots verify  PRICE_OF_READOUT_EVIDENCE.txt.ots      # against PRICE_OF_READOUT_EVIDENCE.txt

# 2. The Experiment 1 & 2 manifest (Bitcoin-confirmed, 2026-06-10):
ots verify  EVIDENCE_MANIFEST_exp1_exp2.txt.ots    # against EVIDENCE_MANIFEST_exp1_exp2.txt

# 3. The Experiment 3 manifest (2026-06-11/12):
ots upgrade EXPERIMENT_EVIDENCE_MANIFEST_exp3.txt.ots
ots verify  EXPERIMENT_EVIDENCE_MANIFEST_exp3.txt.ots
```

Then confirm the pinned-file digests are byte-for-byte unchanged
(`PROVENANCE.md` lists them and the expected values). For example:

```sh
sha256sum experiments/experiment1_gaussian/PREREGISTRATION.md
sha256sum experiments/experiment2_ladder/PREREGISTRATION.md
sha256sum theory_verification/verify_xor_min_e.py
sha256sum theory_verification/superposition_min_e.py
sha256sum experiments/experiment3_cifar/spurious_cifar.py
sha256sum lib/fourq.py
```

The commit hashes pinned by the manifests live in
`curv-institute/por-reproduction` (Experiments 1 & 2; history-preserving
filter-repo extraction, with its own timestamped full-history bundle)
and in the development monorepo (Experiment 3). The manifests list them.

## Reproducing

All experiments are deterministic (randomness flows through declared
seed streams), except that Experiment 3's CNN training needs a GPU.

```sh
# Experiment 1
cd experiments/experiment1_gaussian
python3 lawful_compression_experiment.py --analytic
python3 lawful_compression_experiment.py

# Experiment 2 + theory verification
cd ../experiment2_ladder
python3 readout_geometry_experiment.py --analytic
python3 readout_geometry_experiment.py
cd ../../theory_verification
python3 verify_xor_min_e.py
python3 superposition_min_e.py

# Experiment 3 (GPU for training; CPU for analysis)
cd ../experiments/experiment3_cifar
python3 train_run.py --arm naive|lawful --rho 0.7|0.9|0.95|0.99 --seed 0..9
PYTHONPATH=../../lib python3 analyze_masking.py --run

# The shared instrument's own test suite
cd ../../lib && PYTHONPATH=. python3 test_fourq.py
```

For the **Pythia-410m external replication** (Discussion §"External replication
on a pretrained language model"; **post-hoc, not pre-registered**), regenerate
the two tables from the committed raw data — CPU, standard library only:

```sh
cd experiments/experimentY_pythia
python3 verify.py            # reprints tab:pythia-quant / tab:pythia-ood + correlations, asserts vs EXPECTED_NUMBERS.md
python3 verify.py --smoke    # ALSO re-runs two cells end-to-end on a GPU (needs torch + a corpus)
```

Full end-to-end regeneration needs a CUDA GPU and the corpora fetched by
`experimentY_pythia/fetch/`; see that directory's `README.md` and `MANIFEST.md`.

### Figures and paper build

```sh
cd paper
python3 figures/make_figures.py            # reads the committed results.csv files by relative path
pdflatex -interaction=nonstopmode main.tex # run 2–3× to resolve cross-references
pdflatex -interaction=nonstopmode main.tex
```

`make_figures.py` transcribes no measured value by hand: it reads the
three `results.csv` files and Experiment 3's `results_summary.json`, and
builds the stimulus strip by importing the pinned `spurious_cifar.py`
(its SHA-256 is asserted before import) against the CIFAR-10 test batch.

## Environment

- Python 3.12.3
- NumPy 2.4.6 (the experiments' only third-party dependency; the
  `theory_verification/` scripts need only the standard library)
- Experiment 3 training: PyTorch 2.12.0 (CUDA 13.0); readout
  measurements: scikit-learn 1.9.0
- Figures: Matplotlib 3.10.9 (Agg backend, vector PDF)
- Paper build: pdfTeX 1.40.25, TeX Live 2023

## License

Two licenses (see [`LICENSE`](LICENSE)):

- **Code and data** (`experiments/`, `lib/`, `theory_verification/`, and
  `paper/figures/make_figures.py`): MIT.
- **Paper text and figures** (`paper/`, excluding
  `make_figures.py`): CC BY 4.0.
