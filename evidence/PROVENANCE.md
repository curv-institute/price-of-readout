# Provenance and integrity crosswalk

This repository is a **clean, paper-scoped reproduction** of the evidence
behind

> J.W. Miller (CURV Institute). *The Price of Readout: Exact Interface
> Floors and Frozen-Readout Shift Evaluation.* 2026.

The files here were copied **byte-for-byte** from the authors' research
monorepo and from two pre-existing timestamped evidence archives. This
document records the exact source of every file and proves, by SHA-256,
that the files whose digests are pinned by the existing OpenTimestamps
attestations are unchanged — so those attestations still verify against
the copies in this repository.

## The git dates in THIS repository are not the evidence

This repository was created with **fresh git history** (clean commits)
specifically so that it reads as a paper artifact rather than a slice of
an internal program monorepo. **The commit dates here carry no
evidentiary weight.** The before-any-run ordering of each experiment
(pre-registration committed before any sample was drawn) is established
by two independent, externally anchored mechanisms, both reproduced
verbatim in this directory:

1. **Recorded commit hashes** in the original repositories (a commit
   hash is a content address over its tree, parents, and author/commit
   dates), listed below and in `PRICE_OF_READOUT_EVIDENCE.txt`.
2. **OpenTimestamps attestations** (Bitcoin-anchored, hash-only): the
   `.ots` files in this directory. Only the SHA-256 digest of each
   manifest ever left the machine; the attestation is anchored in the
   Bitcoin blockchain via public calendar servers. Verify with
   `ots verify <manifest>.ots` against the corresponding manifest file.

## Evidence directory contents (copied verbatim, authoritative)

| File | Source | Covers | Attestation state |
| --- | --- | --- | --- |
| `EVIDENCE_MANIFEST_exp1_exp2.txt` (+`.ots`) | `curv-institute/por-reproduction` (the Exp 1 & 2 evidence archive), dated 2026-06-10 | Experiments 1 and 2: prereg/results commit hashes, full-history bundle digest, and the SHA-256 of three pinned files | Bitcoin-confirmed (block headers 953145 / 953150) |
| `lcr.bundle.ots` | same archive | the full-history git bundle digest of the Exp 1 & 2 archive | Bitcoin-confirmed |
| `EXPERIMENT_EVIDENCE_MANIFEST_exp3.txt` (+`.ots`) | program campaign record, dated 2026-06-12 | Experiment 3: prereg / amendment / results commit hashes (post-unblinding closure appended) | pending at calendars (upgrade with `ots upgrade`) |
| `EXPERIMENT_EVIDENCE_MANIFEST_exp3.pre-results.txt.ots` | program campaign record, dated 2026-06-11 | the **pre-results** state of the Exp 3 manifest (registrations only, before any unblinding) | pending at calendars |
| `PRICE_OF_READOUT_EVIDENCE.txt` (+`.ots`) | **fresh**, created 2026-06-15 for this repository | a single combined paper manifest: all three experiments' commit hashes plus the SHA-256 of every evidence file *as it sits in this repository* | stamped 2026-06-15 |

> **Note on the Exp 1 & 2 manifest text.** That manifest and the
> archive it came from use the paper's *earlier* title ("Lawful
> Compression and the Price of Readout: ...") and the earlier repository
> name (`lcr-reproduction`, since renamed to `por-reproduction`). Every
> digest it pins is over content, not names, and is unchanged. It also
> labels Experiments 1 and 2 by their internal development names
> ("track 3 / gate 2" and "track 4 / experiment B"); see the crosswalk
> below.
>
> **Note on the Exp 3 manifest text.** The campaign manifest pins the
> commit hashes of two experiments run in the same campaign: "Experiment
> A" (a JEPA study) and "Experiment B" (= this paper's **Experiment 3**,
> CIFAR masking). **Only Experiment B is part of this paper.** Experiment
> A is banked for a sequel and its materials are deliberately NOT
> included in this repository; the manifest is reproduced verbatim
> (one cannot edit an attested file) and its Experiment-B lines are the
> ones relevant here.

## Pinned-hash verification (must match, byte-for-byte)

The existing attestations pin the SHA-256 of the following files. Each
was copied unmodified into this repository; the digests below were
recomputed from the copies here and match the values pinned in the
manifests.

| File in this repo | Pinned by | Expected SHA-256 | Recomputed | OK |
| --- | --- | --- | --- | --- |
| `theory_verification/verify_xor_min_e.py` | Exp 1&2 manifest | `aa9f4eed57e7ccd7acc7936fc2c7dce06063ea32fee1795843ac3e4d012fd165` | `aa9f4eed57e7ccd7acc7936fc2c7dce06063ea32fee1795843ac3e4d012fd165` | YES |
| `experiments/experiment1_gaussian/PREREGISTRATION.md` | Exp 1&2 manifest | `a493e875c781e3e3f9fb02574c6517444d105f1df242968c1e1b23dd550826bc` | `a493e875c781e3e3f9fb02574c6517444d105f1df242968c1e1b23dd550826bc` | YES |
| `experiments/experiment2_ladder/PREREGISTRATION.md` | Exp 1&2 manifest | `0950877b95efcefbd2910b2b61cb817b3983a7078264c8936781d86ce63526a5` | `0950877b95efcefbd2910b2b61cb817b3983a7078264c8936781d86ce63526a5` | YES |
| `theory_verification/superposition_min_e.py` | paper (Sec. Reproducibility) | `c8de5f8c4eb3c9573086c14682a0b094256c7c7f2a2f555624467cba7b89b61a` | `c8de5f8c4eb3c9573086c14682a0b094256c7c7f2a2f555624467cba7b89b61a` | YES |
| `experiments/experiment3_cifar/spurious_cifar.py` | paper (Appendix details-e3); asserted by `make_figures.py` and `analyze_masking.py` | `ab8f6e61d54be4ea13b23aee326aabccb2f272c4bfffa4431ace45b70c9050ba` | `ab8f6e61d54be4ea13b23aee326aabccb2f272c4bfffa4431ace45b70c9050ba` | YES |
| `lib/fourq.py` | asserted by `analyze_masking.py` | `8b66bc7d07c2f0d1cc11180a5c629b156c014c413cb1b5644a6e1539abdd255a` | `8b66bc7d07c2f0d1cc11180a5c629b156c014c413cb1b5644a6e1539abdd255a` | YES |

All pinned digests match. The `lcr.bundle.ots` additionally pins the
full commit history of the Exp 1 & 2 evidence archive (content, order,
dates); to verify it, obtain that archive's git bundle independently and
run `ots verify lcr.bundle.ots` against it.

## Path crosswalk (monorepo / evidence archive -> this repository)

Paper-native names are used here; the table maps every file to its
original location. Content SHA-256 is unchanged for every file.

| This repository | Original location |
| --- | --- |
| `paper/main.tex`, `paper/main.pdf` | `mlr-proof-program/papers/lawful-compression-readout-geometry/` (paper text pass **v17.6**, commit `e18054a…`, which adds the Discussion §`sec:pythia` external-replication subsection and Tables `tab:pythia-quant`/`tab:pythia-ood`; v17.5 was the prior pass) |
| `paper/figures/*.pdf`, `paper/figures/make_figures.py` | `mlr-proof-program/papers/lawful-compression-readout-geometry/figures/` |
| `experiments/experiment1_gaussian/` | `mlr-proof-program/experiments/track3_gate2/` (internal label "track 3 / gate 2") |
| `experiments/experiment2_ladder/` | `mlr-proof-program/experiments/track4_gateB/` (internal label "track 4 / experiment B") |
| `experiments/experiment3_cifar/` | `mlr-proof-program/experiments/expB_masking_real/` (campaign "Experiment B") |
| `experiments/experimentY_pythia/` (runners) | `mlr-proof-program/experiments/expD_quantization/pilot/` (`y1_pretrained_por.py`, `y5_pretrained_actquant.py`) — **post-hoc; no pre-registration, no OTS attestation** (see below) |
| `experiments/experimentY_pythia/fetch/` | `/gfs/curv-campaign/scripts/` (`fetch_wikitext103.py`, `fetch_expc_q_corpora.py`, `fetch_ood_corpora.py`, `fetch_ood_latin.py`) |
| `experiments/experimentY_pythia/results/` | `expY1_run/`, `expY3_run/`, `expY3b_run/`, `expY5_run/` (raw per-cell `*.json` + `*.log`) |
| `lib/` | `mlr-proof-program/experiments/lib/` |
| `theory_verification/verify_xor_min_e.py` | `mlr-proof-program/experiments/track4_gateB/verify_xor_min_e.py` |
| `theory_verification/superposition_min_e.py` | `mlr-proof-program/experiments/kef_frontier/superposition_min_e.py` |

### Two operational scripts were path-adjusted for this layout

Two **non-pinned** scripts contain hard-coded relative paths to the old
monorepo layout and were edited *only* to resolve files at this
repository's paper-native paths. The edits change nothing about what the
scripts compute; they are recorded here in full for transparency:

- `paper/figures/make_figures.py` — `PROGRAM_ROOT` now resolves to the
  repository root (`HERE.parent.parent`), and the three results.csv,
  the results_summary.json, and the pinned `spurious_cifar.py` are read
  from `experiments/experiment{1_gaussian,2_ladder,3_cifar}/`. The
  SHA-256 pin on `spurious_cifar.py` is unchanged and still asserted
  before import. (No measured value is transcribed by hand; all numbers
  are read from the committed results.)

### The Pythia external replication is post-hoc — no pre-registration, no OTS attestation

`experiments/experimentY_pythia/` packages a **post-hoc, exploratory** external
replication on Pythia-410m, reported in the paper's Discussion
(§`sec:pythia`, Tables `tab:pythia-quant` / `tab:pythia-ood`, paper v17.6
commit `e18054a…`). Unlike Experiments 1–3 it was **run after the fact and was
not pre-registered**, so — deliberately — there is:

- **no `PREREGISTRATION.md`** for it (a `POSTHOC_NOTE.md` occupies that slot and
  says so explicitly), and
- **no OpenTimestamps `.ots` attestation** in this directory for it, and none
  should be minted. Creating a fresh `.ots` would falsely imply a before-the-run
  ordering that does not exist for this leg.

Its provenance instead rests **only** on the development-monorepo *results*
commit hashes recorded in `experimentY_pythia/POSTHOC_NOTE.md` (Y1 `b78770f…`,
Y3 `91244b0…`, Y3b `ceb19bf…`, Y5 `fc545de…`) and the paper fold-in `e18054a…`.
The expD weight-quant `PREREGISTRATION.md` (prereg `ae03b1a…`) is the
*methodological lineage* the Y-series scales up; it is a **different** experiment
and is **not** this leg's pre-registration. None of the existing manifests in
this directory pins any Y-series file; the package is reproducible (the two
tables regenerate from its committed raw data via its `verify.py`), but its
before-the-run status is — honestly — not attested, because it was not
pre-registered.

### A note on textual references to a sibling study

The shared instrument `lib/fourq.py` is **hash-pinned** (its SHA-256 is
asserted at run time and recorded in the evidence manifests), so it is
reproduced **byte-for-byte** and its docstrings are unedited. Those
docstrings mention, by way of design provenance, a sibling study
("Experiment A", a JEPA representation study) that shares the same
instrument. **No material from that study — no data, no scripts, no
results — is part of this paper or this repository;** only the
instrument's own comments refer to it, and they cannot be edited without
breaking the integrity pin. The same applies to the verbatim Experiment 3
manifest in this directory, which pins commit hashes for both that
sibling study and this paper's Experiment 3 and is reproduced unaltered.

### Operational-script path note

`analyze_masking.py` (the once-only Experiment 3 analysis script) is
left **byte-identical** to the committed evidence version. It resolves
its dependency `fourq.py` via `../lib` relative to the experiment
directory; to re-run it in this layout, make `lib/fourq.py` reachable at
that relative path (e.g. symlink `experiments/lib -> ../lib`) or set
`PYTHONPATH` to the repository's `lib/`. Its internal SHA-256 assertion
on `fourq.py` (`8b66bc7d...`) passes against the copy in `lib/`.
