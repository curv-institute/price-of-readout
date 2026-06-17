# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Verify the Pythia external-replication tables against the paper's v17.6 Discussion.

Default mode (no GPU, no deps beyond the stdlib): reads the committed raw per-cell
JSONs in results/, recomputes the two paper tables (tab:pythia-quant, tab:pythia-ood)
plus the Pearson/Spearman correlations and the Y5 activation-quant footnote, and ASSERTS
every cell matches EXPECTED_NUMBERS.md to the recorded tolerance. This regenerates the
tables from the raw data the same way the paper's figure pipeline reads committed results
(no measured value is transcribed by hand here).

    python3 verify.py                 # check committed results/ against expected (CPU, stdlib only)
    python3 verify.py --smoke         # ALSO re-run two cells end-to-end on a GPU and check them
    python3 verify.py --regen-only    # just print the tables, no assertions

--smoke needs a CUDA GPU + the runner's deps (torch, transformers>=4.40, ...); it re-runs
the 16-bit quant reference and the ID-corpus shift cell, writing to a temp dir, and checks
the regenerated numbers against the committed JSONs within EPS_BPB. It will download the
Pythia-410m weights from HF if not cached, but it does NOT fetch any corpus: the WikiText-103
test corpus must already be present at <root>/wikitext103_raw/test.txt, where <root> is
--data-root if given, else $DATA_ROOT, else the runner default (/vault/datasets/text). Use
the fetch/ scripts to obtain it first (see README.md / MANIFEST.md).

Exit status 0 = all checks pass; non-zero = a deviation exceeded tolerance.
"""
from __future__ import annotations

import argparse, json, math, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# ---- tolerances (see EXPECTED_NUMBERS.md) --------------------------------------
# The runner is single-run + deterministic-by-seed, but bf16 autocast makes exact
# bit-reproduction hardware/library-version dependent, so we check within an epsilon
# rather than for an exact match (same posture as the repo's Experiment-3 record).
EPS_BPB  = 0.01    # |delta| per bits/byte cell (table values are shown to <=3 dp)
EPS_CORR = 0.01    # |delta| per correlation (paper shows 2 dp)

# ---- reference & expected (the v17.6 paper tables, transcribed) -----------------
REF_QUANT = 0.861          # 16-bit head-refit refit_id, the quant-table reference (raw 0.860535)

EXPECTED_QUANT = {  # bits -> (total, recoverable, irrecoverable, pct_interface)  (None = "---"/"free")
    8: (0.001, None,  0.001, None),   # "free": refit fully recovers; irrecoverable ~= 0.001
    4: (0.389, 0.222, 0.166, 57),
    3: (1.990, 0.601, 1.390, 30),
    2: (2.451, 0.894, 1.557, 36),
}
EXPECTED_OOD = {  # display name -> (frozen, recovery)   (order = paper table order)
    "code":       (0.68, 0.003),
    "Japanese":   (1.10, 0.021),
    "Vietnamese": (1.24, 0.042),
    "Indonesian": (1.54, 0.063),
    "Finnish":    (1.57, 0.031),
    "Yoruba":     (2.22, 0.295),
    "Swahili":    (2.34, 0.252),
    "Welsh":      (2.46, 0.274),
}
EXPECTED_CORR = {"pearson": 0.93, "spearman": 0.86, "n": 8}
EXPECTED_Y5_MEAN_PCT = 40  # "~40% interface-borne" footnote (a8/a6/a4 -> 45/40/39, mean ~41)

# ---- raw-file map: which JSON feeds which paper cell ----------------------------
# NOTE: the Y3b language JSONs all carry "split":"q1" (vestigial argparse default,
# because the language was selected via --textfile, not --split). Language identity
# is carried ONLY by the filename, never by the JSON 'split' field. See MANIFEST.md.
QUANT_FILES = {16: "quant_b16.json", 8: "quant_b8.json", 4: "quant_b4.json",
               3: "quant_b3.json", 2: "quant_b2.json"}
OOD_FILES = {  # display name -> (raw json, runner invocation that produced it)
    "code":       "shift_q1.json",
    "Japanese":   "shift_q2.json",
    "Vietnamese": "shift_lang_vi.json",
    "Indonesian": "shift_lang_id.json",
    "Finnish":    "shift_lang_fi.json",
    "Yoruba":     "shift_yo.json",
    "Swahili":    "shift_sw.json",
    "Welsh":      "shift_cy.json",
}
Y5_FILES = {16: "actq_a16.json", 8: "actq_a8.json", 6: "actq_a6.json", 4: "actq_a4.json"}


def load(name):
    return json.loads((RES / name).read_text())


def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy)


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for rk, i in enumerate(s):
            r[i] = rk
        return r
    return pearson(rank(x), rank(y))


def regen():
    """Recompute the two tables + correlations + Y5 from the committed raw JSONs."""
    ref = load(QUANT_FILES[16])["refit_id"]
    quant = {}
    for b in (8, 4, 3, 2):
        d = load(QUANT_FILES[b]); fz = d["frozen_id"]; rf = d["refit_id"]
        total = fz - ref; recov = fz - rf; irrec = rf - ref
        pct = recov / total * 100 if total else float("nan")
        quant[b] = dict(frozen=fz, refit=rf, total=total, recoverable=recov,
                        irrecoverable=irrec, pct_interface=pct)
    ood = {}; froz = []; rec = []
    for nm, fn in OOD_FILES.items():
        d = load(fn); fz = d["frozen"]; rf = d["refit"]; rc = fz - rf
        ood[nm] = dict(frozen=fz, refit=rf, recovery=rc)
        froz.append(fz); rec.append(rc)
    corr = dict(pearson=pearson(froz, rec), spearman=spearman(froz, rec), n=len(froz))
    aref = load(Y5_FILES[16])["refit_id"]; y5 = {}
    for a in (8, 6, 4):
        d = load(Y5_FILES[a]); fz = d["frozen_id"]; rf = d["refit_id"]
        total = fz - aref; recov = fz - rf
        y5[a] = dict(total=total, recoverable=recov, pct_interface=recov / total * 100)
    return dict(ref=ref, quant=quant, ood=ood, corr=corr, y5=y5,
                y5_mean_pct=sum(y5[a]["pct_interface"] for a in (8, 6, 4)) / 3)


def print_tables(r):
    print(f"\nReference (16-bit head-refit refit_id) = {r['ref']:.4f}  (paper rounds to {REF_QUANT})")
    print("\n=== Table tab:pythia-quant (bits/byte, ID; vs 16-bit refit reference) ===")
    print(f"{'bits':>4} {'total':>8} {'recover':>8} {'irrecov':>8} {'%interface':>11}")
    for b in (8, 4, 3, 2):
        q = r["quant"][b]
        pct = "free" if b == 8 else f"{q['pct_interface']:.0f}%"
        rec = "---" if b == 8 else f"{q['recoverable']:.3f}"
        tot = q["irrecoverable"] if b == 8 else q["total"]   # paper's 8-bit row shows the irrecoverable value
        print(f"{b:>4} {tot:>8.3f} {rec:>8} {q['irrecoverable']:>8.3f} {pct:>11}")
    print("\n=== Table tab:pythia-ood (bits/byte; frozen | +refit recovery) ===")
    print(f"{'shift':>11} {'frozen':>7} {'recovery':>9}")
    for nm in OOD_FILES:
        o = r["ood"][nm]; print(f"{nm:>11} {o['frozen']:>7.2f}  +{o['recovery']:.3f}")
    c = r["corr"]
    print(f"\nPearson(frozen,recovery)  = {c['pearson']:.4f}  (paper {EXPECTED_CORR['pearson']})")
    print(f"Spearman(frozen,recovery) = {c['spearman']:.4f}  (paper {EXPECTED_CORR['spearman']})  n={c['n']}")
    print("\n=== Y5 activation-quant footnote (~40% interface-borne) ===")
    for a in (8, 6, 4):
        y = r["y5"][a]; print(f"a{a}: total={y['total']:.3f} recoverable={y['recoverable']:.3f} "
                              f"%interface={y['pct_interface']:.0f}%")
    print(f"mean %interface (a8/a6/a4) = {r['y5_mean_pct']:.0f}%  (paper '~40%')")


def check(r):
    fails = []

    def near(label, got, exp, eps):
        if got is None or exp is None:
            return
        if abs(got - exp) > eps:
            fails.append(f"{label}: got {got:.4f}, expected {exp:.4f} (|d|={abs(got-exp):.4f} > {eps})")

    # quant table
    for b, (tot, rec, irr, pct) in EXPECTED_QUANT.items():
        q = r["quant"][b]
        if b == 8:
            near(f"quant b8 irrecoverable", q["irrecoverable"], irr, EPS_BPB)
        else:
            near(f"quant b{b} total", q["total"], tot, EPS_BPB)
            near(f"quant b{b} recoverable", q["recoverable"], rec, EPS_BPB)
            near(f"quant b{b} irrecoverable", q["irrecoverable"], irr, EPS_BPB)
            # %interface: 1 pct point tolerance (it is a ratio of two ~0.01-tolerance numbers)
            if abs(q["pct_interface"] - pct) > 1.5:
                fails.append(f"quant b{b} %interface: got {q['pct_interface']:.1f}, expected {pct}")
    # ood table
    for nm, (fz, rc) in EXPECTED_OOD.items():
        o = r["ood"][nm]
        near(f"ood {nm} frozen", o["frozen"], fz, EPS_BPB)
        near(f"ood {nm} recovery", o["recovery"], rc, EPS_BPB)
    # correlations
    near("pearson", r["corr"]["pearson"], EXPECTED_CORR["pearson"], EPS_CORR)
    near("spearman", r["corr"]["spearman"], EXPECTED_CORR["spearman"], EPS_CORR)
    if r["corr"]["n"] != EXPECTED_CORR["n"]:
        fails.append(f"ood n: got {r['corr']['n']}, expected {EXPECTED_CORR['n']}")
    # Y5 footnote (~40%, allow +/-5 pts on the mean)
    if abs(r["y5_mean_pct"] - EXPECTED_Y5_MEAN_PCT) > 5:
        fails.append(f"Y5 mean %interface: got {r['y5_mean_pct']:.0f}, expected ~{EXPECTED_Y5_MEAN_PCT}")
    return fails


def smoke(args):
    """Re-run two cells end-to-end on a GPU and check them against the committed JSONs."""
    runner = HERE / "y1_pretrained_por.py"
    common = ["--model", args.model, "--ctx", "512", "--eval-bytes", "2000000",
              "--refit-bytes", "8000000", "--refit-tok", "400000"]
    # --data-root is honored consistently with the runner: it sets DATA_ROOT for the
    # re-run subprocess (which rebases the runner's TEXT dict). If not given, the
    # subprocess inherits whatever DATA_ROOT is already in the environment, else the
    # runner default (/vault/datasets/text). The required corpus is NOT fetched here.
    env = dict(os.environ)
    if args.data_root:
        env["DATA_ROOT"] = args.data_root
    root = env.get("DATA_ROOT", "/vault/datasets/text")
    idcorpus = Path(root) / "wikitext103_raw" / "test.txt"
    print(f"[smoke] corpus root = {root}  (ID corpus: {idcorpus})", flush=True)
    if not idcorpus.exists():
        print(f"[smoke] ERROR: required corpus not found at {idcorpus}.\n"
              f"        --smoke does not fetch; run fetch/fetch_wikitext103.py "
              f"(see README.md) and/or pass --data-root.", flush=True)
        return [f"smoke: missing corpus {idcorpus}"]
    fails = []
    with tempfile.TemporaryDirectory() as td:
        cells = [
            (["--mode", "quant", "--bits", "16"], "quant_b16.json",
             ("frozen_id", "refit_id")),
            (["--mode", "shift", "--split", "id"], "shift_id.json",
             ("frozen", "refit")),  # cheapest shift cell, ID corpus, no --textfile needed
        ]
        for extra, committed, keys in cells:
            out = Path(td) / committed
            cmd = [sys.executable, str(runner)] + common + extra + ["--out", str(out)]
            print(f"\n[smoke] running: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, env=env)
            got = json.loads(out.read_text()); want = load(committed)
            for k in keys:
                d = abs(got[k] - want[k])
                ok = "OK" if d <= EPS_BPB else "FAIL"
                print(f"[smoke] {committed} {k}: regen={got[k]:.4f} committed={want[k]:.4f} |d|={d:.4f} {ok}")
                if d > EPS_BPB:
                    fails.append(f"smoke {committed} {k}: |d|={d:.4f} > {EPS_BPB}")
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regen-only", action="store_true", help="print the tables, no assertions")
    ap.add_argument("--smoke", action="store_true",
                    help="also re-run two cells end-to-end on a GPU and check within tolerance")
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--data-root", default=None,
                    help="--smoke: corpus root (sets DATA_ROOT for the re-run); else uses the runner default")
    args = ap.parse_args()

    r = regen()
    print_tables(r)
    if args.regen_only:
        return 0

    fails = check(r)
    if args.smoke:
        fails += smoke(args)

    print("\n" + "=" * 60)
    if fails:
        print(f"VERIFY: FAIL ({len(fails)} deviation(s) exceeded tolerance):")
        for f in fails:
            print("  - " + f)
        return 1
    print("VERIFY: PASS — all cells reproduce within tolerance "
          f"(|d bits/byte| <= {EPS_BPB}, |d corr| <= {EPS_CORR}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
