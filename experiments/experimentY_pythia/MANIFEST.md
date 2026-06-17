# MANIFEST — Pythia-410m external replication (post-hoc)

Everything needed to regenerate the two v17.6 Discussion tables: the model, the
corpora (with sources + SHA-256), the exact command per paper cell, and the
expected numbers with tolerances. See `EXPECTED_NUMBERS.md` for the table
transcription/derivation and `POSTHOC_NOTE.md` for the honest-labeling and provenance.

## Model

| | |
| --- | --- |
| HF model id | **`EleutherAI/pythia-410m`** |
| License | Apache-2.0 |
| Scale | ~410M params, trained on ~300B Pile tokens |
| Tokenizer | the model's own (GPT-NeoX BPE) |

## Corpora (fetched via `fetch/`, never vendored — sources + SHA-256 pinned)

Each fetch script writes `data.txt` + a `MANIFEST.json` under `$DATA_ROOT/<name>/`
(default `$DATA_ROOT = /vault/datasets/text`). Re-fetching reproduces these digests.

| split / language | dir under `$DATA_ROOT` | HF source | bytes | sha256 (data file) | fetched by |
| --- | --- | --- | ---: | --- | --- |
| ID (WikiText-103 **test**) | `wikitext103_raw/test.txt` | `Salesforce/wikitext` `wikitext-103-raw-v1` test | 1,287,656 | `bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a` | `fetch_wikitext103.py` |
| code (q1) | `expc_q1_code_python/data.txt` | `codeparrot/codeparrot-clean-valid` (python `content`), 64 MiB cap | 67,136,840 | `cbfd259747c03b57e1d2e9af2614cbfaed062c7b83ddc748410488c00a9a819e` | `fetch_expc_q_corpora.py` |
| Japanese (q2) | `expc_q2_ja_wikipedia/data.txt` | `wikimedia/wikipedia` `20231101.ja`, 64 MiB cap | 67,112,647 | `df74a58f443efd50a555ab18f8e02c9b0b6b2ff56745869257141b52cc692be6` | `fetch_expc_q_corpora.py` |
| Swahili (sw) | `ood_sw_wikipedia/data.txt` | `wikimedia/wikipedia` `20231101.sw`, 16 MiB cap | 16,782,614 | `84ee59acfc351f342f0418a499aefcb306d98af1c4dfec47b296a2a0d7c3f2be` | `fetch_ood_corpora.py` |
| Telugu (te) — *excluded* | `ood_te_wikipedia/data.txt` | `wikimedia/wikipedia` `20231101.te`, 16 MiB cap | 16,781,266 | `52aaffcd350f48a9048f03cfb458fca3073f8bab91e197ff25896992296c4130` | `fetch_ood_corpora.py` |
| Vietnamese (vi) | `ood_vi_wikipedia/data.txt` | `wikimedia/wikipedia` `20231101.vi`, 16 MiB cap | 16,784,810 | `ced21e9db8a69c40f185e31a252b98e49fe4c89e39a0f6ecf7266b659e16d985` | `fetch_ood_latin.py` |
| Indonesian (id) | `ood_id_wikipedia/data.txt` | `wikimedia/wikipedia` `20231101.id`, 16 MiB cap | 16,777,788 | `9bc74e84965307c638ba0826d0ee65983719a8337c17f4134015d02923c2ef02` | `fetch_ood_latin.py` |
| Finnish (fi) | `ood_fi_wikipedia/data.txt` | `wikimedia/wikipedia` `20231101.fi`, 16 MiB cap | 16,842,474 | `f001788dd7c48b287d5e75d1fd9a63a4420f09cd604a4cdcdd58ec7c339f27fa` | `fetch_ood_latin.py` |
| Welsh (cy) | `ood_cy_wikipedia/data.txt` | `wikimedia/wikipedia` `20231101.cy`, 16 MiB cap | 16,777,445 | `e27c8266ab7b4a075facd960f6079f458687d59d7e5d74023ff8adf0596cab12` | `fetch_ood_latin.py` |
| Yoruba (yo) | `ood_yo_wikipedia/data.txt` | `wikimedia/wikipedia` `20231101.yo`, 16 MiB cap | 15,999,908 | `ae341edc2b849cc9cf3a741f3ae256da6f1877756e1598fcf7495f64a47b1070` | `fetch_ood_latin.py` |

Licenses: WikiText / Wikipedia CC-BY-SA; codeparrot-clean per CodeParrot terms;
Pythia Apache-2.0. We ship the **fetch scripts + the pinned digests**, not the
corpus bytes (size + attribution), matching the repo's existing no-vendoring pattern.

## Fetch the data

```sh
# default writes under /vault/datasets/text/... ; edit OUT in a script, or just
# fetch then point the runner at the result with $DATA_ROOT / --textfile / --idfile.
uv run fetch/fetch_wikitext103.py        # ID (WikiText-103 test)
uv run fetch/fetch_expc_q_corpora.py     # code (q1) + Japanese (q2)
uv run fetch/fetch_ood_corpora.py        # Swahili (sw) + Telugu (te, excluded)
uv run fetch/fetch_ood_latin.py          # vi / id / fi / cy / yo
```

## Exact command per paper cell

Common args (every cell): `--model EleutherAI/pythia-410m --ctx 512
--eval-bytes 2000000 --refit-bytes 8000000 --refit-tok 400000`.
The runner reads corpora from `$DATA_ROOT` (default `/vault/datasets/text`);
override the corpus root with `DATA_ROOT=...`, the ID path with `--idfile`, or any
shift path with `--textfile` (see "Packaging note" in `README.md`).

### `tab:pythia-quant` (mode=quant; ID corpus)

| paper bits | command (after the common args) | output → committed |
| --- | --- | --- |
| 16 (ref) | `--mode quant --bits 16 --out quant_b16.json` | `results/quant_b16.json` |
| 8 | `--mode quant --bits 8 --out quant_b8.json` | `results/quant_b8.json` |
| 4 | `--mode quant --bits 4 --out quant_b4.json` | `results/quant_b4.json` |
| 3 | `--mode quant --bits 3 --out quant_b3.json` | `results/quant_b3.json` |
| 2 | `--mode quant --bits 2 --out quant_b2.json` | `results/quant_b2.json` |

### `tab:pythia-ood` (mode=shift)

In-mixture / built-in splits use `--split`; the OOD-curve languages use `--textfile`
(which is why their JSON `split` field is the vestigial `q1` — see below).

| shift | command (after the common args) | output → committed |
| --- | --- | --- |
| code | `--mode shift --split q1 --out shift_q1.json` | `results/shift_q1.json` |
| Japanese | `--mode shift --split q2 --out shift_q2.json` | `results/shift_q2.json` |
| Swahili | `--mode shift --split sw --out shift_sw.json` | `results/shift_sw.json` |
| Telugu (excluded) | `--mode shift --split te --out shift_te.json` | `results/shift_te.json` |
| Vietnamese | `--mode shift --textfile $DATA_ROOT/ood_vi_wikipedia/data.txt --out shift_lang_vi.json` | `results/shift_lang_vi.json` |
| Indonesian | `--mode shift --textfile $DATA_ROOT/ood_id_wikipedia/data.txt --out shift_lang_id.json` | `results/shift_lang_id.json` |
| Finnish | `--mode shift --textfile $DATA_ROOT/ood_fi_wikipedia/data.txt --out shift_lang_fi.json` | `results/shift_lang_fi.json` |
| Welsh | `--mode shift --textfile $DATA_ROOT/ood_cy_wikipedia/data.txt --out shift_cy.json` | `results/shift_cy.json` |
| Yoruba | `--mode shift --textfile $DATA_ROOT/ood_yo_wikipedia/data.txt --out shift_yo.json` | `results/shift_yo.json` |
| (ID sanity) | `--mode shift --split id --out shift_id.json` | `results/shift_id.json` |

> **`split` field is vestigial for the `--textfile` cells.** Vietnamese, Indonesian,
> Finnish, Welsh, and Yoruba JSONs all record `"split": "q1"` because `--split`
> stayed at its default while the corpus was selected by `--textfile`. The language
> is identified by the **filename**, not by the JSON `split` field.

### Y5 activation-quant footnote (~40% interface-borne)

| abits | command | output → committed |
| --- | --- | --- |
| 16 (ref) | `uv run y5_pretrained_actquant.py --abits 16 --ctx 512 --eval-bytes 2000000 --refit-bytes 8000000 --refit-tok 400000 --out actq_a16.json` | `results/actq_a16.json` |
| 8 / 6 / 4 | `... --abits {8,6,4} --out actq_a{8,6,4}.json` | `results/actq_a{8,6,4}.json` |

## Verify

```sh
uv run verify.py            # CPU, stdlib only: regenerate both tables from results/ and assert vs expected
uv run verify.py --smoke    # ALSO re-run two cells end-to-end on a GPU and check within tolerance
```

Expected output ends with `VERIFY: PASS — all cells reproduce within tolerance`.
See `EXPECTED_NUMBERS.md` for the per-cell expected values and the epsilon.

## Provenance of the package files

| file(s) | original location | note |
| --- | --- | --- |
| `y1_pretrained_por.py` | `mlr-proof-program/experiments/expD_quantization/pilot/y1_pretrained_por.py` | shipped as the **working-tree (`--textfile`) version** that produced Y3b; `$DATA_ROOT`/`--idfile` added for portability (see README). |
| `y5_pretrained_actquant.py` | same pilot dir | `--idfile` added for parity; imports `y1_pretrained_por`. |
| `fetch/*.py` | `/gfs/curv-campaign/scripts/` | `fetch_expc_q_corpora.py` docstring fixed (`the-stack-smol` → `codeparrot-clean-valid`; the call/sha256 were always codeparrot). |
| `results/*.json`, `results/*.log` | `expY1_run/`, `expY3_run/`, `expY3b_run/`, `expY5_run/` | raw per-cell data + run logs, byte-for-byte. |

`quantize_lm.py` / `actquant_lm.py` from the pilot dir are **deliberately NOT
shipped**: those are the small-LM (expC BPE) cores and pull in monorepo-only deps
(`train_lm`, an expC checkpoint). The Pythia weight-quant is self-contained inside
`y1_pretrained_por.py` (`per_out_quant_` / `quant_body_`); the activation-quant is
self-contained inside `y5_pretrained_actquant.py`.
