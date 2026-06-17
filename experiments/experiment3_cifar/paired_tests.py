#!/usr/bin/env python3
"""Distribution-free corroboration of the Experiment-3 masking gates.

The pre-registered gates (G-B1..G-B4) are evaluated on seed-means with a
committed per-cell slack delta = max(2*SE_seed, 0.10). This script adds two
assumption-light per-seed checks that do not rely on the normal-SE bound:

  (1) Sign test (exact binomial, H0 p=0.5) that the naive arm's per-seed
      delta_transfer exceeds the 0.5-bit G-B2 threshold under the decorrelate
      shift, at each rho in {0.9, 0.95, 0.99}.
  (2) Exact paired permutation test (all 2^10 sign flips) of
      naive-minus-lawful delta_transfer, paired by seed, at each rho.

Reads the committed results.csv in this directory. Prints a table; exits 0.
"""
import csv, itertools, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = list(csv.DictReader(open(os.path.join(HERE, "results.csv"))))
V = {}
for r in ROWS:
    V[(r["arm"], r["rho"], int(r["seed"]), r["shift"], r["measurement"])] = float(r["value_bits"])
SEEDS = list(range(10))
RHOS = ["0.9", "0.95", "0.99"]


def sign_test_gt(vals, thr):
    k = sum(1 for v in vals if v > thr)
    n = len(vals)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n  # one-sided
    return k, n, p


def perm_paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    obs = sum(d) / n
    cnt = sum(1 for sg in itertools.product([1, -1], repeat=n)
              if abs(sum(s * di for s, di in zip(sg, d)) / n) >= abs(obs) - 1e-12)
    return obs, cnt / 2 ** n  # two-sided


def main():
    print("G-B2 naive delta_transfer > 0.5 bits (decorrelate), exact sign test")
    for rho in RHOS:
        vals = [V[("naive", rho, s, "decorrelate", "delta_transfer")] for s in SEEDS]
        k, n, p = sign_test_gt(vals, 0.5)
        print(f"  rho={rho}: {k}/{n} seeds>0.5  mean={sum(vals)/n:.3f}  sign-p={p:.4g}")
    print("\nnaive vs lawful delta_transfer, exact paired permutation (decorrelate)")
    for rho in RHOS:
        a = [V[("naive", rho, s, "decorrelate", "delta_transfer")] for s in SEEDS]
        b = [V[("lawful", rho, s, "decorrelate", "delta_transfer")] for s in SEEDS]
        obs, p = perm_paired(a, b)
        print(f"  rho={rho}: mean(naive-lawful)={obs:.3f}  perm-p={p:.4g}")


if __name__ == "__main__":
    main()
