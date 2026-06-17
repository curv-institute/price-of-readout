#!/usr/bin/env python3
"""Recompute the two Pythia tables + correlations + Y5 from the DISJOINT-split
result JSONs, using the exact formulas of experimentY_pythia/verify.py:regen()."""
import json, math
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
QUANT = {16: "quant_b16.json", 8: "quant_b8.json", 4: "quant_b4.json",
         3: "quant_b3.json", 2: "quant_b2.json"}
OOD = {"code": "shift_q1.json", "Japanese": "shift_q2.json",
       "Vietnamese": "shift_lang_vi.json", "Indonesian": "shift_lang_id.json",
       "Finnish": "shift_lang_fi.json", "Yoruba": "shift_yo.json",
       "Swahili": "shift_sw.json", "Welsh": "shift_cy.json"}
Y5 = {16: "actq_a16.json", 8: "actq_a8.json", 6: "actq_a6.json", 4: "actq_a4.json"}


def load(n):
    return json.loads((RES / n).read_text())


def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy)


def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
    for pos, i in enumerate(s):
        r[i] = pos
    return r


def spearman(x, y):
    return pearson(rank(x), rank(y))


def main():
    ref = load(QUANT[16])["refit_id"]
    print(f"16-bit head-refit reference (refit_id) = {ref:.4f}")
    print(f"  16-bit frozen_id = {load(QUANT[16])['frozen_id']:.4f}")
    print("\n=== tab:pythia-quant (above 16-bit refit reference) ===")
    print(f"{'bits':>4} {'frozen':>8} {'refit':>8} {'total':>8} {'recover':>8} {'irrecov':>8} {'%iface':>7}")
    for b in (8, 4, 3, 2):
        d = load(QUANT[b]); fz, rf = d["frozen_id"], d["refit_id"]
        total, recov, irrec = fz - ref, fz - rf, rf - ref
        pct = recov / total * 100 if total else float("nan")
        print(f"{b:>4} {fz:>8.3f} {rf:>8.3f} {total:>8.3f} {recov:>8.3f} {irrec:>8.3f} {pct:>6.0f}%")
    print("\n=== tab:pythia-ood (frozen | +refit recovery) ===")
    froz, rec = [], []
    print(f"{'shift':>11} {'frozen':>7} {'refit':>7} {'recovery':>9}")
    for nm, fn in OOD.items():
        d = load(fn); fz, rf = d["frozen"], d["refit"]; rc = fz - rf
        froz.append(fz); rec.append(rc)
        print(f"{nm:>11} {fz:>7.3f} {rf:>7.3f} {rc:>+9.3f}")
    print(f"\nPearson(frozen,recovery)  = {pearson(froz, rec):.4f}")
    print(f"Spearman(frozen,recovery) = {spearman(froz, rec):.4f}   n={len(froz)}")
    # Telugu (excluded from n=8) for the record:
    try:
        dte = load("shift_te.json"); print(f"[excluded] Telugu frozen={dte['frozen']:.3f} recovery={dte['frozen']-dte['refit']:+.3f}")
    except FileNotFoundError:
        pass
    print("\n=== Y5 activation-quant ===")
    aref = load(Y5[16])["refit_id"]; pcts = []
    print(f"16-bit-activation refit reference = {aref:.4f}")
    print(f"{'abits':>5} {'total':>8} {'recover':>8} {'%iface':>7}")
    for a in (8, 6, 4):
        d = load(Y5[a]); fz, rf = d["frozen_id"], d["refit_id"]
        total, recov = fz - aref, fz - rf; pct = recov / total * 100
        pcts.append(pct)
        print(f"{a:>5} {total:>8.3f} {recov:>8.3f} {pct:>6.0f}%")
    print(f"mean %interface (a8/a6/a4) = {sum(pcts)/3:.0f}%")


if __name__ == "__main__":
    main()
