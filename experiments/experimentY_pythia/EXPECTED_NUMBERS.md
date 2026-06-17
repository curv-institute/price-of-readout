# Expected numbers — the v17.6 Discussion tables, with derivation and tolerances

This file transcribes the two paper tables from §`sec:pythia` **exactly as
published (v17.6)**, maps every cell to the raw JSON in `results/` that produces
it, gives the exact derivation, and records the reproduction tolerance. `verify.py`
checks the committed raw data against these values automatically.

All cross-entropies are in **bits per byte** (sum of token NLL in bits divided by
raw UTF-8 byte count). The reference for the quant table is the **16-bit
head-refit** score, `results/quant_b16.json` → `refit_id = 0.860535`, which the
paper rounds to **0.861**.

## Tolerances

The runner is single-run and deterministic-by-seed, but head-refit and evaluation
use **bf16 autocast**, so exact bit-reproduction is hardware/library-version
dependent (same posture the repo records for Experiment 3). Reproduction is checked
**within an epsilon**, not for exact equality:

- per bits/byte cell: **|Δ| ≤ 0.01 bits/byte**  (`EPS_BPB` in `verify.py`)
- per correlation: **|Δ| ≤ 0.01**  (`EPS_CORR`)
- `% interface` (a ratio of two ε-tolerance numbers): **±1.5 percentage points**
- Y5 mean `% interface`: **±5 percentage points** of the "~40%" footnote

Verdict to report: *reproduces within epsilon* (not bit-exact).

---

## Table `tab:pythia-quant` — weight quantization, ID (WikiText-103)

Each row `b` comes from `results/quant_b{b}.json` (`frozen_id`, `refit_id`).
With `ref = quant_b16.json:refit_id = 0.860535`:

- `total`         = `frozen_id(b) − ref`
- `recoverable`   = `frozen_id(b) − refit_id(b)`   (the interface-borne part head-refit removes)
- `irrecoverable` = `refit_id(b) − ref`             (the information-borne floor head-refit cannot remove)
- `% interface`   = `recoverable / total`

| bits | total | recoverable (interface) | irrecoverable (info) | % interface | source file |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 0.001 | --- | 0.001 | free | `results/quant_b8.json` |
| 4 | 0.389 | 0.222 | **0.166** | 57% | `results/quant_b4.json` |
| 3 | 1.990 | 0.601 | **1.390** | 30% | `results/quant_b3.json` |
| 2 | 2.451 | 0.894 | **1.557** | 36% | `results/quant_b2.json` |

**8-bit "free" row — what the `0.001` means.** At 8 bits the raw frozen damage is
`total = frozen_id − ref = 0.997 − 0.861 ≈ 0.137`, but head-refit recovers
essentially all of it: `irrecoverable = refit_id(8) − ref = 0.8611 − 0.8605 ≈
0.001`. The paper's 8-bit row shows that **irrecoverable** value (`0.001`) in its
"total" column and labels `% interface` as **free**, because the kept (information)
loss is ~zero — 8-bit is lossless after a head-refit. `verify.py` checks the 8-bit
*irrecoverable* cell against `0.001`.

## Table `tab:pythia-ood` — distribution shift (8 languages)

Each row is `results/shift_*.json` (`frozen`, `refit`); `recovery = frozen − refit`
(the head-refit-recoverable, interface-borne part). Frozen CE is the OOD-distance
proxy. Rows in paper order:

| shift | frozen (OOD-ness) | refit recovery | source file |
| --- | ---: | ---: | --- |
| code | 0.68 | +0.003 | `results/shift_q1.json` |
| Japanese | 1.10 | +0.021 | `results/shift_q2.json` |
| Vietnamese | 1.24 | +0.042 | `results/shift_lang_vi.json` |
| Indonesian | 1.54 | +0.063 | `results/shift_lang_id.json` |
| Finnish | 1.57 | +0.031 | `results/shift_lang_fi.json` |
| Yoruba | 2.22 | **+0.295** | `results/shift_yo.json` |
| Swahili | 2.34 | **+0.252** | `results/shift_sw.json` |
| Welsh | 2.46 | **+0.274** | `results/shift_cy.json` |

Correlation of `recovery` vs `frozen` over these **n = 8** points:
**Pearson 0.93** (recomputed 0.9287), **Spearman 0.86** (recomputed 0.8571).

> **`split` field caveat.** The five `shift_lang_*` / `shift_cy` / `shift_yo`
> JSONs all record `"split": "q1"`. That field is a vestigial argparse default:
> the language was selected via `--textfile` (which overrides the corpus path while
> `--split` stays at its default), so **the JSON `split` field does NOT identify
> the language**. Language identity is carried only by the filename and by
> `MANIFEST.md`. `verify.py` and the mapping table key off the filename, never the
> `split` field.

> **Telugu excluded.** `results/shift_te.json` (frozen ≈ 0.638) is **not** in the
> n = 8 set. Telugu is multi-byte script (≈3 UTF-8 bytes/glyph), which deflates
> bits/byte and makes its frozen "OOD-ness" not comparable to the Latin-script
> points — so it is excluded from the OOD-distance correlation as a bits-per-byte
> artifact (the paper caption: "a multi-byte-script point is excluded as a
> bits-per-byte artifact"). It is shipped for completeness only. (Note: code and
> Japanese are retained; their per-byte CE is not deflated the way Telugu's
> multi-byte glyphs are, so "Latin-script, n=8" loosely scopes the set the
> correlation is computed over.)

## Activation-quant footnote (Y5): "~40% interface-borne"

`results/actq_a{a}.json` (`frozen_id`, `refit_id`); reference
`aref = actq_a16.json:refit_id = 0.860535`. `% interface = (frozen−refit)/(frozen−aref)`:

| abits | total | recoverable | % interface | source file |
| ---: | ---: | ---: | ---: | --- |
| 8 | 0.484 | 0.220 | 45% | `results/actq_a8.json` |
| 6 | 1.800 | 0.713 | 40% | `results/actq_a6.json` |
| 4 | 3.530 | 1.390 | 39% | `results/actq_a4.json` |

Mean ≈ 41%, i.e. the **"~40% interface-borne"** footnote (a distinct *mixed* locus,
between the information-borne weight-quant side and the interface-borne shift side).
