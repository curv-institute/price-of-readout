# Experiment Y — external replication on a pretrained language model (Pythia-410m)

> **POST-HOC, EXPLORATORY REPLICATION — NOT PRE-REGISTERED.** Unlike
> Experiments 1–3, this was run *after the fact* on an off-the-shelf model and
> carries **no pre-registration, no frozen gates, and no OpenTimestamps
> attestation.** It confirms an already-published finding on new ground; it is not
> a new registered claim. Read [`POSTHOC_NOTE.md`](POSTHOC_NOTE.md) first — it
> occupies the slot `PREREGISTRATION.md` holds for the other experiments and states
> exactly what this does and does not claim.

Maps to the paper's Discussion subsection **"External replication on a pretrained
language model"** (`sec:pythia`, **v17.6**) and its two tables **`tab:pythia-quant`**
and **`tab:pythia-ood`**, plus the activation-quant **~40% interface-borne** footnote.

The same frozen-vs-head-refit protocol used in Experiments 1–3 is applied to
**Pythia-410m** (≈300B Pile tokens), scored in bits/byte:

- **Weight quantization is information-borne** — per-output-channel round-to-nearest
  fake-quant of the transformer-block weights produces an irreducible refit floor
  with a sharp 4→3-bit onset (`tab:pythia-quant`).
- **Distribution shift is interface-borne** — head-refit recovers a fraction that
  grows monotonically with OOD distance, near zero for in-mixture shifts (code,
  Japanese) up to ≈0.27 bits/byte for clearly-OOD low-resource languages
  (`tab:pythia-ood`; Pearson 0.93, Spearman 0.86, n=8).
- **Activation quantization is a distinct mixed locus** (~40% interface-borne; Y5).

## Files

| File | Role | Paper element |
| --- | --- | --- |
| `POSTHOC_NOTE.md` | **honest-labeling note** (replaces `PREREGISTRATION.md`): post-hoc, no prereg/gates/OTS; claims + caveats; monorepo results-commit provenance | §`sec:pythia` framing |
| `MANIFEST.md` | model id, corpora + sources + SHA-256, **exact command per cell**, run→cell map, verify instructions | both tables |
| `EXPECTED_NUMBERS.md` | the two tables transcribed verbatim + per-cell source file + derivation + **tolerances** | `tab:pythia-quant`, `tab:pythia-ood` |
| `y1_pretrained_por.py` | unified runner (`--mode {quant,shift}`); produces the quant + shift cells | both tables |
| `y5_pretrained_actquant.py` | activation-quant runner (imports the Y1 harness) | ~40% footnote |
| `fetch/` | corpus fetch scripts (WikiText-103; code/Japanese; sw/te; vi/id/fi/cy/yo) | data acquisition |
| `results/` | **raw per-cell data**: `*.json` (frozen/refit bits/byte) + `*.log` (run logs), byte-for-byte from the monorepo run dirs | the measured values |
| `results_summary.json` | aggregated tables + correlations + Y5 + an `_env` block (regenerated from `results/`) | both tables |
| `verify.py` | regenerates both tables from `results/` and **asserts they match `EXPECTED_NUMBERS.md` within tolerance**; `--smoke` re-runs two cells on a GPU | the check |

## Reproduce

Single deterministic run per cell; no seed grid (it is a confirmation, not a
registered seed study). The full run needs a CUDA GPU; the table-regeneration check
runs on CPU with the standard library only.

### 1. Fetch corpora (writes `$DATA_ROOT/<name>/data.txt` + a `MANIFEST.json`)

```sh
uv run fetch/fetch_wikitext103.py        # ID (WikiText-103 test)
uv run fetch/fetch_expc_q_corpora.py     # code (q1) + Japanese (q2)
uv run fetch/fetch_ood_corpora.py        # Swahili (sw) + Telugu (te, excluded)
uv run fetch/fetch_ood_latin.py          # vi / id / fi / cy / yo
```

### 2. Run the cells (GPU). Common args + per-cell flags are tabulated in `MANIFEST.md`.

```sh
COMMON="--model EleutherAI/pythia-410m --ctx 512 --eval-bytes 2000000 --refit-bytes 8000000 --refit-tok 400000"

# tab:pythia-quant (ID); the 16-bit refit is the table's reference (0.861)
for b in 16 8 4 3 2; do uv run y1_pretrained_por.py $COMMON --mode quant --bits $b --out results/quant_b$b.json; done

# tab:pythia-ood — in-mixture splits use --split; OOD-curve languages use --textfile
uv run y1_pretrained_por.py $COMMON --mode shift --split q1 --out results/shift_q1.json   # code
uv run y1_pretrained_por.py $COMMON --mode shift --split q2 --out results/shift_q2.json   # Japanese
uv run y1_pretrained_por.py $COMMON --mode shift --split sw --out results/shift_sw.json   # Swahili
uv run y1_pretrained_por.py $COMMON --mode shift --split te --out results/shift_te.json   # Telugu (excluded)
for lc in vi:lang_vi id:lang_id fi:lang_fi cy:cy yo:yo; do
  code=${lc%%:*}; tag=${lc##*:}
  uv run y1_pretrained_por.py $COMMON --mode shift \
      --textfile $DATA_ROOT/ood_${code}_wikipedia/data.txt --out results/shift_${tag}.json
done

# Y5 activation-quant footnote
for a in 16 8 6 4; do uv run y5_pretrained_actquant.py $COMMON --abits $a --out results/actq_a$a.json; done
```

### 3. Verify (CPU; regenerates both tables from `results/` and asserts vs expected)

```sh
uv run verify.py            # check committed results/ within tolerance (stdlib only)
uv run verify.py --smoke    # ALSO re-run two cells end-to-end on a GPU and check
```

Ends with `VERIFY: PASS — all cells reproduce within tolerance`.

### Environment / dependencies

- Python ≥ 3.10 (run env 3.12.3). Runner deps are declared in each script's PEP-723
  header: `torch` (PyTorch cu130 index), `numpy`, `transformers>=4.40`, `safetensors`,
  `huggingface_hub`. Fetch scripts: `huggingface_hub>=0.23`, `pyarrow>=15`.
- `uv run` installs these per-script automatically. Exact torch/transformers build
  used for the committed numbers was not recorded in the logs (uv-managed); pin the
  floors above. `verify.py` itself needs **no third-party packages**.
- **Reproduction is within an epsilon, not bit-exact**: head-refit uses bf16
  autocast, so cells reproduce to ~2–3 decimals on the same GPU class but are not
  guaranteed identical across hardware. Tolerance: |Δ bits/byte| ≤ 0.01 per cell,
  |Δ correlation| ≤ 0.01 (see `EXPECTED_NUMBERS.md`). Same posture the repo records
  for Experiment 3.

## Packaging note (path portability + the `--textfile` flag)

The original runner hard-coded corpus paths under `/vault/datasets/text/...`. For
this package the corpus root is overridable:

- `DATA_ROOT=/your/path` (env var; default `/vault/datasets/text`) — rebases the
  whole `TEXT` dict, so `quant`/`shift`/Y5 all read from your fetched corpora;
- `--idfile PATH` — overrides just the ID (WikiText-103 test) path for `quant` and Y5;
- `--textfile PATH` — overrides a shift corpus path (already present in the
  working-tree runner; **required** for the five OOD-curve languages).

The `--textfile` flag and these overrides change **no science** (`path =
override or TEXT[split]`; identical compute). The committed Y3b language results
were produced by the `--textfile` version, which is why those JSONs carry the
vestigial `"split": "q1"` — language identity lives in the filename, not the JSON
(see `EXPECTED_NUMBERS.md` and `MANIFEST.md`).

## Provenance

Post-hoc; **no pre-registration exists** (by design — see `POSTHOC_NOTE.md`).
Original location: `mlr-proof-program/experiments/expD_quantization/pilot/`
(runners) + `/gfs/curv-campaign/scripts/` (fetch) + `expY{1,3,3b,5}_run/` (raw data).
Monorepo *results* commits: Y1 `b78770f…`, Y3 `91244b0…`, Y3b `ceb19bf…`,
Y5 `fc545de…`; paper v17.6 fold-in `e18054a…`; methodological-lineage prereg
(expD weight-quant, a *different* experiment) `ae03b1a…`. Full hashes in
`POSTHOC_NOTE.md` and `../../evidence/PROVENANCE.md`.
