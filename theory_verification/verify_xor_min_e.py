#!/usr/bin/env python3
"""Machine verification of every finite enumeration claimed in the paper
"Lawful Compression and the Price of Readout" (papers/
lawful-compression-readout-geometry/main.tex), in particular the
exhaustive E <= 11 circuit enumeration behind Proposition prop:minE
(the XOR family's minimal-E edge at exactly E = 12).

Sections (each ends in hard assertions; any failure raises):

  1. Linear-threshold-function (LTF) census for fan-in f = 0..3 by
     exhaustive integer weight/threshold grid, validated against the
     known census counts 2 / 4 / 14 / 104 (OEIS A000609).  Muroga's
     bound (integer weights of magnitude <= 2 realize every threshold
     function of <= 3 variables) guarantees the grid |w_i| <= 4,
     half-integer thresholds, is complete; the count match is the check.
  2. Closure/reduction lemmas used to prune the circuit search, each
     machine-checked against the census tables:
       (a) complementation closure at equal fan-in (equal cost);
       (b) repeated-input absorption: a gate reading a repeated signal
           computes an LTF of its distinct signals (smaller fan-in,
           strictly cheaper gate);
       (c) constant-input absorption: a gate reading a constant signal
           computes an LTF of its remaining signals (same).
     Together (b)-(c) justify restricting the main enumeration to gates
     with distinct, non-constant inputs.
  3. Exhaustive enumeration of all affine-threshold composition circuits
     on {0,1}^2 (Definition def:acomp; fan-in-f gate costs 2f + 1 FLOPs)
     with total cost E <= 11: all gate-count/fan-in shapes, all wirings,
     all LTFs per gate.  Asserts that NO circuit computes XOR or its
     complement, i.e. no circuit at E <= 11 induces the diagonal
     partition -- the minimality half of Proposition prop:minE.  Prints
     the achieved function set at each budget E <= 11, verifies equality
     with the 14 halfspace-achievable functions (the paper's halfspace
     table, Table tab:app-xor) at every E >= 5, and asserts absence of
     XOR and its complement at every E.
  4. The E = 12 witness of Proposition prop:minE computes XOR exactly,
     at K = 7 stored reals unshared and K = 3 under value sharing.
  5. Proposition prop:budget-trivial machine check: circuits with
     E <= E(A_lin(2)) = 5 realize exactly the 14 halfspace-achievable
     Boolean functions of Appendix app:enum-xor and nothing else.
  6. Appendix app:enum-xor: halfspace census over the four points of
     {0,1}^2 -- exactly 14 achievable subsets / 7 achievable partitions,
     diagonal pair non-achievable, floor (3/4) h2(1/3), best error 1/4.
     Proposition prop:quadratic: 1{(z1 - z2)^2 >= 1/2} computes XOR.
  7. Appendix app:enum-sup and Theorem thm:superposition (ii)-(iv):
     prefix/suffix entropy table, the 21-contiguous-block width-2
     enumeration (cut-pair check included) with floor (2/3) h2(1/4),
     the width-3 exact decode, the quadratic-threshold tie on the line
     (Remark rem:costing-artifact), and the sine decoder sign pattern
     (Proposition prop:sine).

Dependencies: Python standard library only.  Runtime: well under a
minute on any machine.

Usage: python3 verify_xor_min_e.py
"""

import itertools
import math
from fractions import Fraction

LOG2_3 = math.log2(3)
XOR_FLOOR = 0.75 * LOG2_3 - 0.5          # (3/4) h2(1/3) ~ 0.688722 bits
SUP_FLOOR = 4.0 / 3.0 - 0.5 * LOG2_3      # (2/3) h2(1/4) ~ 0.540852 bits
TOL = 1e-12


def h2(p):
    if p in (0, 1):
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def gate_cost(f):
    """FLOP cost of a fan-in-f affine-threshold gate (Def. def:acomp)."""
    return 2 * f + 1


# ----------------------------------------------------------------------
# Section 1: LTF census, validated against the known counts 2/4/14/104
# ----------------------------------------------------------------------
def ltf_census(max_fanin=3, wmax=4):
    """census[f] = sorted set of truth tables (tuples over the 2^f input
    points in itertools.product order) realizable as 1{w.x >= theta}."""
    census = {}
    for f in range(max_fanin + 1):
        pts = list(itertools.product((0, 1), repeat=f))
        tables = set()
        # Half-integer thresholds cover all strict/non-strict cases for
        # integer weights; range generously past any attainable w.x.
        thetas = [t / 2 for t in range(-2 * wmax * (f + 1) - 1,
                                       2 * wmax * (f + 1) + 2)]
        for w in itertools.product(range(-wmax, wmax + 1), repeat=f):
            for th in thetas:
                tables.add(tuple(
                    1 if sum(wi * xi for wi, xi in zip(w, x)) >= th else 0
                    for x in pts))
        census[f] = tables
    return census


CENSUS = ltf_census()
KNOWN_COUNTS = {0: 2, 1: 4, 2: 14, 3: 104}  # OEIS A000609
for f, n in KNOWN_COUNTS.items():
    assert len(CENSUS[f]) == n, (f, len(CENSUS[f]))
print("[1] LTF census validated: counts per fan-in 0/1/2/3 =",
      [len(CENSUS[f]) for f in range(4)], "== [2, 4, 14, 104] (A000609)")


# ----------------------------------------------------------------------
# Section 2: closure / reduction lemmas, machine-checked
# ----------------------------------------------------------------------
def apply_table(table, f, args):
    """Evaluate truth table of fan-in f at bit-tuple args."""
    idx = 0
    for b in args:
        idx = (idx << 1) | b
    return table[idx]


# (a) complementation closure
for f in range(4):
    for t in CENSUS[f]:
        assert tuple(1 - v for v in t) in CENSUS[f]

# (b) repeated-input absorption: substitute any duplication pattern of
# g distinct signals into a fan-in-f LTF -> LTF of g signals.
for f in (2, 3):
    for g in range(1, f):
        for assign in itertools.product(range(g), repeat=f):
            if len(set(assign)) != g:
                continue
            pts_g = list(itertools.product((0, 1), repeat=g))
            for t in CENSUS[f]:
                induced = tuple(
                    apply_table(t, f, tuple(x[a] for a in assign))
                    for x in pts_g)
                assert induced in CENSUS[g], (f, g, assign, t)

# (c) constant-input absorption: fix any single input to 0/1 -> LTF of
# the remaining f-1 inputs.
for f in (1, 2, 3):
    pts_m = list(itertools.product((0, 1), repeat=f - 1))
    for pos in range(f):
        for const in (0, 1):
            for t in CENSUS[f]:
                induced = tuple(
                    apply_table(t, f,
                                x[:pos] + (const,) + x[pos:])
                    for x in pts_m)
                assert induced in CENSUS[f - 1], (f, pos, const, t)

print("[2] Reduction lemmas verified: complement closure; repeated-input"
      " and constant-input absorption into smaller-fan-in LTFs")


# ----------------------------------------------------------------------
# Section 3: exhaustive circuit enumeration at E <= budget
# ----------------------------------------------------------------------
DOMAIN = list(itertools.product((0, 1), repeat=2))  # (0,0),(0,1),(1,0),(1,1)
XOR_T = tuple(z1 ^ z2 for z1, z2 in DOMAIN)         # (0, 1, 1, 0)
NXOR_T = tuple(1 - v for v in XOR_T)                # (1, 0, 0, 1)


def achievable_functions(budget):
    """Set of (cost, output truth table over DOMAIN) for ALL circuits of
    affine-threshold gates with distinct non-constant inputs (justified
    exhaustive by Section 2 reductions), total cost <= budget.  Gates are
    enumerated in topological order; the last gate is the output (every
    prefix is itself enumerated, so non-final outputs are covered).
    Returns dict: table -> min cost."""
    base_signals = [tuple(z[0] for z in DOMAIN), tuple(z[1] for z in DOMAIN)]
    best = {}

    def rec(signals, cost_so_far):
        n_avail = len(signals)
        for f in range(1, n_avail + 1):
            c = cost_so_far + gate_cost(f)
            if c > budget:
                continue
            # fan-in census must cover f
            assert f in CENSUS, f"needed census fan-in {f} not computed"
            for combo in itertools.combinations(range(n_avail), f):
                ins = [signals[i] for i in combo]
                for t in CENSUS[f]:
                    out = tuple(
                        apply_table(t, f, tuple(s[p] for s in ins))
                        for p in range(4))
                    if out not in best or c < best[out]:
                        best[out] = c
                    rec(signals + [out], c)

    rec(base_signals, 0)
    return best


# Cost argument making fan-in <= 3 sufficient (asserted, not assumed):
# every reading gate costs >= 3, so at E <= 11 at most 3 gates; gate k
# sees 2 + (k - 1) distinct signals, so fan-in 4 occurs only at gate 3,
# costing 9 + 3 + 3 = 15 > 11.  The recursion above never requests
# census fan-in > 3 under budget 11 (the assert inside rec guards this).
BEST11 = achievable_functions(11)
assert XOR_T not in BEST11, "XOR computed at E <= 11 -- prop:minE FALSIFIED"
assert NXOR_T not in BEST11, "NXOR computed at E <= 11 -- prop:minE FALSIFIED"
n_funcs_11 = len(BEST11)
print(f"[3] Exhaustive E <= 11 enumeration: {n_funcs_11} distinct Boolean"
      " functions achievable; XOR (0,1,1,0) and its complement are NOT"
      " among them -- minimality of E = 12 verified")

# Per-budget audit trail: print the achieved function set at each E <= 11,
# verify equality with the 14 halfspace-achievable functions of the paper's
# halfspace-enumeration table (Table tab:app-xor) once the linear budget
# E = 5 is reached, and assert absence of XOR and its complement at every E.
HALFSPACE_T6 = {tuple(apply_table(t, 2, z) for z in DOMAIN)
                for t in CENSUS[2]}  # the 14 functions of Table tab:app-xor
assert len(HALFSPACE_T6) == 14
print("[3a] Achieved function set per budget E (truth tables over"
      " (0,0),(0,1),(1,0),(1,1)):")
for E in range(0, 12):
    achieved = {tab for tab, c in BEST11.items() if c <= E}
    assert XOR_T not in achieved, f"XOR achieved at E = {E}"
    assert NXOR_T not in achieved, f"NXOR achieved at E = {E}"
    assert achieved <= HALFSPACE_T6, f"non-halfspace function at E = {E}"
    if E >= 5:
        assert achieved == HALFSPACE_T6, (
            f"E = {E} set != the 14 halfspace functions of Table tab:app-xor")
    print(f"  E <= {E:2d}: {len(achieved):2d} functions:",
          sorted(''.join(map(str, t)) for t in achieved))
print("[3a] Per-E sets verified: every set is XOR/NXOR-free; from E = 5"
      " through E = 11 each equals the 14 halfspace functions of the"
      " halfspace-enumeration table exactly")


# ----------------------------------------------------------------------
# Section 4: the E = 12 witness of prop:minE
# ----------------------------------------------------------------------
def witness(z1, z2):
    g1 = 1 if z1 + z2 - 2 >= 0 else 0            # fan-in 2: 5 FLOPs
    return 1 if z1 + z2 - 2 * g1 - 1 >= 0 else 0  # fan-in 3: 7 FLOPs


assert tuple(witness(*z) for z in DOMAIN) == XOR_T
E_witness = gate_cost(2) + gate_cost(3)
assert E_witness == 12
K_unshared = (2 + 1) + (3 + 1)                    # f + 1 reals per gate
assert K_unshared == 7
stored_values = {1, -2} | {1, 1, -2, -1}          # g1: (1,1,-2); r: (1,1,-2,-1)
assert stored_values == {1, -2, -1} and len(stored_values) == 3
print("[4] E = 12 witness verified: computes XOR exactly; E = 5 + 7 = 12,"
      " K = 7 unshared, K = 3 value-shared")


# ----------------------------------------------------------------------
# Section 5: prop:budget-trivial -- E <= 5 gives exactly the 14
# halfspace-achievable functions
# ----------------------------------------------------------------------
BEST5 = achievable_functions(5)
halfspace_fns = set()
for t in CENSUS[2]:  # all LTFs of (z1, z2), as tables over DOMAIN
    halfspace_fns.add(tuple(apply_table(t, 2, z) for z in DOMAIN))
assert len(halfspace_fns) == 14
assert set(BEST5) == halfspace_fns, "E <= 5 functions != halfspace set"
# Stronger observed fact: even at E <= 11 nothing beyond the 14 appears.
assert set(BEST11) == halfspace_fns
print("[5] prop:budget-trivial verified: E <= 5 circuits realize exactly"
      " the 14 halfspace-achievable functions, nothing else"
      " (and the E <= 11 set is the same 14)")


# ----------------------------------------------------------------------
# Section 6: app:enum-xor halfspace census, floor, error; prop:quadratic
# ----------------------------------------------------------------------
def cond_entropy(cells, labels, masses):
    """H(Y | partition) for cells = list of index lists."""
    H = 0.0
    for cell in cells:
        m = sum(masses[i] for i in cell)
        if m == 0:
            continue
        p1 = sum(masses[i] for i in cell if labels[i] == 1) / m
        H += float(m) * h2(p1)
    return H


pts4 = DOMAIN
labels4 = list(XOR_T)
mass4 = [Fraction(1, 4)] * 4
subsets = set()
for w1 in range(-4, 5):
    for w2 in range(-4, 5):
        for th2 in range(-17, 18):  # theta in half-integers
            th = th2 / 2
            B = frozenset(i for i, (a, b) in enumerate(pts4)
                          if w1 * a + w2 * b >= th)
            subsets.add(B)
assert len(subsets) == 14
diag1 = frozenset({1, 2})  # {(0,1),(1,0)}
diag0 = frozenset({0, 3})  # {(0,0),(1,1)}
assert diag1 not in subsets and diag0 not in subsets
partitions = {frozenset({B, frozenset(set(range(4)) - B)}) for B in subsets}
assert len(partitions) == 7
entropies, errors = [], []
for part in partitions:
    cells = [sorted(c) for c in part]
    entropies.append(cond_entropy(cells, labels4, mass4))
    err = sum(min(sum(mass4[i] for i in c if labels4[i] == 1),
                  sum(mass4[i] for i in c if labels4[i] == 0))
              for c in cells)
    errors.append(float(err))
assert abs(min(entropies) - XOR_FLOOR) < TOL
assert abs(min(errors) - 0.25) < TOL
# prop:quadratic
assert tuple(1 if (z1 - z2) ** 2 >= 0.5 else 0 for z1, z2 in pts4) == XOR_T
print("[6] app:enum-xor verified: 14 achievable subsets / 7 partitions,"
      " diagonal pair excluded; floor = (3/4) h2(1/3) ="
      f" {XOR_FLOOR:.6f} bits; best error 1/4;"
      " quadratic decoder computes XOR (prop:quadratic)")


# ----------------------------------------------------------------------
# Section 7: superposition instance enumerations
# ----------------------------------------------------------------------
atoms = [3, 5, 6, 9, 10, 12]
labels6 = [1, 1, 0, 1, 0, 0]
mass6 = [Fraction(1, 6)] * 6

# (ii) prefix/suffix table
prefix_H = [cond_entropy([list(range(m)), list(range(m, 6))],
                         labels6, mass6) for m in range(7)]
expected = [1.0, 5 / 6 * h2(2 / 5), SUP_FLOOR, h2(1 / 3) * 1.0,
            SUP_FLOOR, 5 / 6 * h2(3 / 5), 1.0]
# note: m = 3 gives (1/2)h2(1/3) + (1/2)h2(1/3) = h2(1/3)
for got, exp in zip(prefix_H, expected):
    assert abs(got - exp) < TOL, (got, exp)
assert abs(min(prefix_H) - SUP_FLOOR) < TOL
assert prefix_H.index(min(prefix_H)) == 2 and abs(prefix_H[4] - SUP_FLOOR) < TOL

# Only the exact label partition beats the floor among ALL 2^6 partitions
# (the analogue of the XOR diagonal-pair argument).
beaters = []
for bits in range(64):
    cellA = [i for i in range(6) if (bits >> i) & 1]
    cellB = [i for i in range(6) if not (bits >> i) & 1]
    if cond_entropy([cellA, cellB], labels6, mass6) < SUP_FLOOR - 1e-9:
        beaters.append(frozenset(cellA))
assert set(beaters) == {frozenset({0, 1, 3}), frozenset({2, 4, 5})}

# (iii) width-2: all contiguous blocks (21) and explicit cut-pair check
block_H = []
for i in range(6):
    for j in range(i, 6):
        block = list(range(i, j + 1))
        rest = [k for k in range(6) if k not in block]
        block_H.append(cond_entropy([block, rest], labels6, mass6))
assert len(block_H) == 21
vals = sorted({round(v, 4) for v in block_H})
assert vals == [0.5409, 0.8091, 0.9183, 1.0], vals
assert abs(min(block_H) - SUP_FLOOR) < TOL
# cut-pair + output-labeling enumeration reaches no lower value
cuts = [2.0, 4.0, 5.5, 7.5, 9.5, 11.0, 13.0]
best_pair = 1.0
for c1, c2 in itertools.product(cuts, repeat=2):
    for lab in itertools.product((0, 1), repeat=3):
        region = [lab[(a >= c1) + (a >= c2)] for a in atoms]
        cellA = [i for i in range(6) if region[i] == 1]
        cellB = [i for i in range(6) if region[i] == 0]
        best_pair = min(best_pair,
                        cond_entropy([cellA, cellB], labels6, mass6))
assert abs(best_pair - SUP_FLOOR) < TOL

# (iv) width-3 exact decode
def width3(z):
    g1 = 1 if -z + 5.5 >= 0 else 0
    g2 = 1 if z - 7 >= 0 else 0
    g3 = 1 if z - 9.5 >= 0 else 0
    return 1 if g1 + g2 - g3 - 1 >= 0 else 0


assert [width3(z) for z in atoms] == labels6

# rem:costing-artifact: quadratic threshold on the line ties, never beats
best_quad = 1.0
for c4 in range(0, 61):       # center c in [0, 15] step 0.25
    c = c4 / 4
    for t4 in range(0, 161):  # radius^2 threshold in [0, 40] step 0.25
        t = t4 / 4
        cellA = [i for i, a in enumerate(atoms) if (a - c) ** 2 >= t]
        cellB = [i for i in range(6) if i not in cellA]
        best_quad = min(best_quad,
                        cond_entropy([cellA, cellB], labels6, mass6))
assert abs(best_quad - SUP_FLOOR) < TOL  # ties the floor, does not beat it

# prop:sine: exact decode with margin
omega, phi = 2.8616, 0.5236
sins = [math.sin(omega * z + phi) for z in atoms]
assert all((s >= 0.30) if y == 1 else (s <= -0.30)
           for s, y in zip(sins, labels6))

print("[7] Superposition enumerations verified: prefix/suffix floor"
      f" (2/3) h2(1/4) = {SUP_FLOOR:.6f} bits at m = 2, 4; only the exact"
      " label partition beats it; 21-block width-2 enumeration and"
      " cut-pair check tie the floor (thm:superposition(iii));"
      " width-3 decode exact (iv); quadratic-on-line ties"
      " (rem:costing-artifact); sine decoder margin 0.30 (prop:sine)")

print()
print("ALL CHECKS PASSED")
print(f"  XOR floor   (3/4) h2(1/3) = {XOR_FLOOR:.12f} bits")
print(f"  sup. floor  (2/3) h2(1/4) = {SUP_FLOOR:.12f} bits")
print(f"  minimal E to beat the XOR floor in A_comp(2): 12"
      f" (no circuit at E <= 11 among {n_funcs_11} achievable functions)")
