# Post-hoc note — this is NOT a pre-registration

**This experiment is a post-hoc, exploratory external replication. It was run
*after* Experiments 1–3 and was *not* pre-registered.** This file occupies the
slot that `PREREGISTRATION.md` occupies for Experiments 1–3, and it exists to
say so plainly. There is no `PREREGISTRATION.md` for the Y-series, and none
should be inferred or fabricated. The paper labels it the same way:

> "As a post-hoc external check (not pre-registered, unlike Experiments 1–3),
> we ask whether the frozen-versus-refit decomposition and the two failure
> modes it separates survive on a real, off-the-shelf pretrained model:
> Pythia-410m." — paper §`sec:pythia` (v17.6)

## How this differs from the pre-registered experiments

| | Experiments 1–3 | Y-series (this package) |
| --- | --- | --- |
| Pre-registration committed before any sample drawn | yes (SHA-256-pinned, OTS-anchored) | **no** |
| Frozen pass/fail gates declared in advance | yes | **no** |
| Seed grid / per-seed raw data | yes (`results.csv`, many seeds) | single deterministic run per cell (no seed grid) |
| Blinding / amendment policy | yes | n/a |
| OpenTimestamps attestation | yes (`evidence/*.ots`) | **no** — provenance rests only on the monorepo *results* commit hashes listed below |

The Y-series replicates an already-pre-registered, already-published finding
(the frozen-vs-refit damage taxonomy of Experiments 1–3 and the expD weight-quant
decomposition) on a real pretrained model. It is a *confirmation on new ground*,
not a new registered claim, which is why it is reported in the Discussion and
labeled post-hoc rather than added as Experiment 4.

## What it claims

- The frozen-vs-head-refit decomposition separates the **same two failure modes**
  on a real pretrained model (Pythia-410m) as on the constructed/CIFAR systems:
  - **weight quantization is information-borne** — an irreducible refit floor with
    a sharp 4→3-bit onset (`tab:pythia-quant`);
  - **distribution shift is interface-borne** — head-refit-recoverable, and the
    recoverable fraction grows monotonically with OOD distance (`tab:pythia-ood`);
  - **activation quantization is a distinct mixed locus** (~40% interface-borne;
    Y5 footnote).

## What it does NOT claim (scope / caveats, verbatim from the paper)

- **A single model family and scale** (Pythia-410m only).
- **Round-to-nearest fake quantization** (per-output-channel symmetric; not a
  production quantizer).
- **Bits per byte is comparable only across same-script-width corpora.** A
  multi-byte-script point (**Telugu**, `results/shift_te.json`, frozen ≈ 0.638)
  is **excluded** from the OOD-distance correlation as a bits-per-byte artifact:
  its 3-byte glyphs deflate per-byte cross-entropy, so its frozen "OOD-ness"
  proxy is not comparable to the Latin-script points. The reported correlation
  is over the **n = 8** points in `tab:pythia-ood`.
- Numbers reproduce **within an epsilon**, not bit-exactly: head-refit uses
  bf16 autocast, so cell values reproduce to ~2–3 decimals on the same GPU class
  but are not guaranteed identical across hardware/library versions. See
  `EXPECTED_NUMBERS.md` for the tolerance.

## Provenance (monorepo *results* commit hashes — there are no prereg hashes)

These are the commits in the development monorepo (`curv-wiki`) where the raw
results in `results/` were committed. They are the honest provenance anchor for a
post-hoc study; **none of them is a pre-registration**, by design.

| Leg | Monorepo results commit |
| --- | --- |
| Y1 (quant sweep + shift id/code/Japanese) | `b78770f19827a67573373a2875e06e06beb2ab35` |
| Y3 (OOD shift: Swahili, Telugu) | `91244b0a44e254944033f3badd76bb7d574d5c7f` |
| Y3b (8-language OOD curve: vi/id/fi/cy/yo) | `ceb19bf262e45b73fe52486cb829d1f25b64d936` |
| Y5 (activation quant) | `fc545de2179684b165de9ca118949c0c8b3d62a4` |
| Paper v17.6 fold-in (added §`sec:pythia` + both tables) | `e18054a641a291648935fc701ce0b299aea9d318` |

**Methodological lineage (pre-registered, but a *different* experiment):** the
weight-quant frozen-vs-refit decomposition the Y-series scales up was first
pre-registered as the expD weight-quant battery, prereg commit
`ae03b1af5c8429f064677c9c9fac157a6528670d`. The Y-series reuses that *method* on
a real model; it does **not** inherit that experiment's pre-registration as its
own, and the expD `PREREGISTRATION.md` must not be presented as the Y-series prereg.
