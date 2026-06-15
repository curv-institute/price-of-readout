# Theory verification

Two standard-library-only Python scripts that re-run, with hard
assertions, every finite enumeration claimed in the paper. Each prints
its checks and asserts every claim; any failure raises. Both run in
seconds and need no third-party dependency.

## `verify_xor_min_e.py`

SHA-256 `aa9f4eed…012fd165` (pinned by the Experiment 1 & 2 evidence
manifest). Backs:

- the linear-threshold-function (LTF) census for fan-in 0..3 against the
  known counts 2 / 4 / 14 / 104 (OEIS A000609);
- the closure/reduction lemmas used to prune the circuit search;
- the **exhaustive E ≤ 11 circuit enumeration** behind
  **Proposition `prop:minE`** — the XOR family's minimal-E edge at
  exactly E = 12 (no affine-threshold composition circuit at E ≤ 11
  computes XOR or its complement);
- the E ≤ 5 check of **Proposition `prop:budget-trivial`**;
- the quadratic-decoder and sine-decoder checks
  (**Propositions `prop:quadratic`, `prop:sine`**);
- Remark `rem:costing-artifact` and Theorem `thm:superposition`(iii);
- the Appendix A floor censuses.

```sh
python3 verify_xor_min_e.py    # prints ALL CHECKS PASSED
```

## `superposition_min_e.py`

SHA-256 `c8de5f8c…7b89b61a` (pinned in the paper's Reproducibility
section). Exact `Fraction` arithmetic; 23 hard-asserted checks. Backs the
superposition-frontier enumerations and case analyses:
**Lemma `lem:breakpoint`** through **Proposition `prop:frontier-sup`**
(Appendix "Enumerations: superposition frontier",
`app:enum-sup-frontier`), including the width-2 / width-3 superposition
enumerations.

```sh
python3 superposition_min_e.py
```

## Provenance

`verify_xor_min_e.py` — from
`mlr-proof-program/experiments/track4_gateB/`.
`superposition_min_e.py` — from
`mlr-proof-program/experiments/kef_frontier/`.
Content SHA-256 unchanged; see `../evidence/PROVENANCE.md`.
