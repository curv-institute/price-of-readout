# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
# ]
# ///
"""
test_fourq.py — validation suite for the fourq four-quantity harness.

Run:  uv run test_fourq.py        (plain asserts; exit code 0 = all green)

Anchors:
  1. Track-3 toy regeneration: the generative model of
     ../track3_gate2/PREREGISTRATION.md §2-§3 is regenerated exactly
     (same construction, same rng streams default_rng(10000+seed), same
     draw order, same n_train/n_test), the harness battery is run on it,
     and the results are compared against the COMMITTED per-seed values in
     ../track3_gate2/results.csv and the committed analytic predictions.
     The data is bit-identical to the committed run; the readout optimizer
     is not (committed: custom damped-Newton IRLS, ridge 1e-8 on the mean
     loss; harness: sklearn lbfgs, effective mean-loss ridge ~2e-13). Both
     converge to the (essentially identical) class optimum, so the honest
     comparison is per-seed agreement within an optimizer-equivalence
     tolerance, declared below with its rationale — NOT exact float
     equality.
  2. Analytic closed-form checks: (a) binary symmetric Gaussian with
     logistic-linear Bayes posterior, Bayes CE by Gauss-Hermite quadrature;
     (b) 4-class planted-softmax posterior, Bayes CE by exact-posterior
     Monte Carlo. Harness estimates must converge to the closed forms
     (rate documented: MLE excess ~ n_params/(2 n ln 2) bits).
  3. Schema discipline: the serializer refuses undeclared quantities,
     unlicensed CE* declarations, and entropy declarations above log2|Y|
     (the §6.5 tell); frozen-shift rows carry not_an_entropy=True.

All seeds fixed; the suite is deterministic.
"""

import json
import math
import os
import sys
import tempfile
import time
import traceback

import numpy as np

LIB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LIB_DIR)
import fourq  # noqa: E402

TRACK3_DIR = os.path.join(LIB_DIR, "..", "track3_gate2")
LOG2 = math.log(2.0)


# ===========================================================================
# Anchor 1 — Track-3 toy (committed seeds; exact data regeneration)
# ===========================================================================
# Generative model: verbatim from track3_gate2 PREREGISTRATION.md §2 /
# lawful_compression_experiment.py (same draw order, same rng streams).
D_S = 4
W = np.array([1.0, 1.0, 1.0, 1.0])
WNORM = 2.0
RHO_TRAIN = 0.9
RHO_SHIFT = -0.9
N_TRAIN = 50_000
N_TEST = 100_000
SEEDS = list(range(10))

# --- Declared tolerances, with rationale ---
# Per-seed: identical data (bit-identical rng streams), two near-exact
# optimizers of the same strictly convex objective (ridge difference 1e-8
# vs ~2e-13 on the mean loss; committed run converged to grad inf-norm
# <= 2.3e-17, sklearn to lbfgs ftol/gtol limits). Observed max deviations
# on this machine: 9.7e-9 bits ID, 4.3e-6 bits frozen-shift (the shift
# side is off-distribution, hence more weight-sensitive; worst at
# d_n = 256). Tolerances set at ~100x/~10x the observed maxima to absorb
# BLAS/platform variation — still 4-5 orders of magnitude below the
# program's delta = 0.010 bits proxy resolution. (These were tightened
# from provisional 1e-3/5e-3 after the first run; tightening is
# permitted, loosening would require escalation.)
TOL_PER_SEED_ID = 1e-6       # bits, per-seed ID CE vs committed
TOL_PER_SEED_SHIFT = 5e-5    # bits, per-seed frozen-shift CE vs committed
# Seed-mean (10 seeds) statistical agreement, program conventions:
TOL_MEAN_VS_ANALYTIC = 0.010   # bits — the pre-registered G2-c tolerance
TOL_DRIFT = 0.010              # bits — the pre-registered delta
TOL_REFIT_VS_ANALYTIC = 0.010  # bits — Q-refit vs analytic H_A(Y|N)=CE*(Q)
TOL_ELEVATION = 0.02           # bits — frozen elevation vs committed 1.6666


def generate(rng, n, d_n, rho):
    """Verbatim re-implementation of the committed toy's generator."""
    S = rng.standard_normal((n, D_S))
    u = S @ W / WNORM
    if d_n > 0:
        N = rho * u[:, None] + np.sqrt(1.0 - rho**2) * rng.standard_normal((n, d_n))
    else:
        N = np.zeros((n, 0))
    z = S @ W
    p = np.empty_like(z)
    pos = z >= 0
    p[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    p[~pos] = ez / (1.0 + ez)
    Y = (rng.random(n) < p).astype(float)
    return S, N, Y


def load_committed():
    rows = {}
    import csv as _csv
    with open(os.path.join(TRACK3_DIR, "results.csv")) as f:
        for r in _csv.DictReader(f):
            rows[(int(r["seed"]), int(r["d_n"]), r["arm"])] = (
                float(r["H_hat_id_bits"]), float(r["H_hat_shift_bits"]))
    with open(os.path.join(TRACK3_DIR, "analytic_predictions.json")) as f:
        analytic = json.load(f)
    return rows, analytic


def test_anchor1_track3_toy():
    committed, analytic = load_committed()
    H_S_analytic = analytic["H_Y_given_S_bits"]           # 0.66654
    an8 = analytic["per_dn"]["8"]
    # H_A(Y|N) at d_n=8 (= CE*_log(P); by the sign-flip symmetry of the
    # construction it equals CE*_log(Q) too — RESULTS.md terminology block:
    # "a readout retrained under Q would land back at the ID numbers"):
    HA_N8 = an8["H_A_logistic_Y_given_N_bits"]            # 0.67885
    CE_shift_N8 = an8["CE_shift_unlawful_bits"]           # 2.3414

    per_seed_id_err = []
    per_seed_shift_err = []
    bat_S, bat_N = [], []
    id_X, id_S, id_N = [], [], []

    for seed in SEEDS:
        d_n = 8
        rng = np.random.default_rng(10_000 + seed)
        S_tr, N_tr, y_tr = generate(rng, N_TRAIN, d_n, RHO_TRAIN)
        S_id, N_id, y_id = generate(rng, N_TEST, d_n, RHO_TRAIN)
        S_sh, N_sh, y_sh = generate(rng, N_TEST, d_n, RHO_SHIFT)

        # X arm: fit + frozen eval only (battery on S and N arms below)
        rX = fourq.fit_logistic(np.hstack([S_tr, N_tr]), y_tr)
        ceX_id = fourq.ce_bits(rX, np.hstack([S_id, N_id]), y_id)
        ceX_sh = fourq.ce_bits(rX, np.hstack([S_sh, N_sh]), y_sh)
        cid, csh = committed[(seed, d_n, "X")]
        per_seed_id_err.append(abs(ceX_id - cid))
        per_seed_shift_err.append(abs(ceX_sh - csh))
        id_X.append(ceX_id)

        # S arm (lawful) and N arm (unlawful): full four-quantity battery
        bS = fourq.battery(S_tr, y_tr, S_id, y_id, S_sh, y_sh,
                           k_refit_folds=5, seed=0,
                           proxy_validated=True,
                           license_ref="track3_gate2 G2-c proxy validation "
                                       "(RESULTS.md); logistic-linear "
                                       "construction")
        cid, csh = committed[(seed, d_n, "S")]
        per_seed_id_err.append(abs(bS.CEhat_ID - cid))
        per_seed_shift_err.append(abs(bS.CE_Q_frozen - csh))
        bat_S.append(bS)
        id_S.append(bS.CEhat_ID)

        bN = fourq.battery(N_tr, y_tr, N_id, y_id, N_sh, y_sh,
                           k_refit_folds=5, seed=0,
                           proxy_validated=True,
                           license_ref="track3_gate2 G2-c proxy validation "
                                       "(RESULTS.md)")
        cid, csh = committed[(seed, d_n, "N")]
        per_seed_id_err.append(abs(bN.CEhat_ID - cid))
        per_seed_shift_err.append(abs(bN.CE_Q_frozen - csh))
        bat_N.append(bN)
        id_N.append(bN.CEhat_ID)

    max_id = max(per_seed_id_err)
    max_sh = max(per_seed_shift_err)
    print(f"  d_n=8, 10 seeds x 3 arms: max per-seed |CE - committed|: "
          f"ID {max_id:.2e}, frozen-shift {max_sh:.2e} bits")
    assert max_id < TOL_PER_SEED_ID, \
        f"per-seed ID CE mismatch vs committed results.csv: {max_id:.3e}"
    assert max_sh < TOL_PER_SEED_SHIFT, \
        f"per-seed frozen-shift CE mismatch vs committed: {max_sh:.3e}"

    # Headline 1: ID CE values vs analytic ground truth (G2-c style)
    for name, vals, ref in [("X", id_X, H_S_analytic),
                            ("S", id_S, H_S_analytic),
                            ("N", id_N, HA_N8)]:
        dev = abs(np.mean(vals) - ref)
        print(f"  ID CE seed-mean, arm {name}: {np.mean(vals):.5f} "
              f"(analytic {ref:.5f}, |dev| {dev:.5f})")
        assert dev < TOL_MEAN_VS_ANALYTIC, f"arm {name} ID vs analytic: {dev}"

    # Headline 2: near-zero S-arm drift (lawful arm, frozen under shift)
    s_drift = np.mean([b.CE_Q_frozen - b.CEhat_ID for b in bat_S])
    s_dinfo = np.mean([b.delta_info for b in bat_S])
    s_dtrans = np.mean([b.delta_transfer for b in bat_S])
    print(f"  S arm: frozen drift {s_drift:+.5f}, delta_info {s_dinfo:+.5f}, "
          f"delta_transfer {s_dtrans:+.5f} bits")
    assert abs(s_drift) < TOL_DRIFT
    assert abs(s_dinfo) < TOL_DRIFT
    assert abs(s_dtrans) < TOL_DRIFT

    # Headline 3: the ~1.7-bit FROZEN elevation on the N arm, which the
    # Q-refit pair decomposes into delta_info ~ 0 + delta_transfer ~ 1.66
    # (transfer risk, not information loss — the §2-bis separation)
    n_frozen = np.mean([b.CE_Q_frozen for b in bat_N])
    n_elev = np.mean([b.CE_Q_frozen - b.CEhat_ID for b in bat_N])
    n_refit = np.mean([b.CEhat_Q_refit for b in bat_N])
    n_dinfo = np.mean([b.delta_info for b in bat_N])
    n_dtrans = np.mean([b.delta_transfer for b in bat_N])
    committed_frozen = np.mean([committed[(s, 8, "N")][1] for s in SEEDS])
    committed_elev = committed_frozen - np.mean(
        [committed[(s, 8, "N")][0] for s in SEEDS])
    print(f"  N arm: CE_Q(r_P) {n_frozen:.5f} (committed {committed_frozen:.5f},"
          f" analytic {CE_shift_N8:.5f}); elevation {n_elev:.5f} "
          f"(committed {committed_elev:.5f})")
    print(f"  N arm: CEhat_Q_refit {n_refit:.5f} (analytic CE*(Q) {HA_N8:.5f});"
          f" delta_info {n_dinfo:+.5f}, delta_transfer {n_dtrans:+.5f}")
    assert abs(n_elev - committed_elev) < TOL_ELEVATION, \
        f"frozen elevation {n_elev} vs committed {committed_elev}"
    assert n_elev > 0.8313, "pre-registered G3-b binding threshold"
    assert abs(n_refit - HA_N8) < TOL_REFIT_VS_ANALYTIC, \
        f"Q-refit {n_refit} vs analytic {HA_N8}"
    assert abs(n_dinfo) < TOL_DRIFT, \
        f"delta_info should be ~0 (sign-flip preserves information): {n_dinfo}"
    assert abs(n_dtrans - (CE_shift_N8 - HA_N8)) < TOL_ELEVATION, \
        f"delta_transfer {n_dtrans} vs analytic {CE_shift_N8 - HA_N8}"

    # Per-seed spot checks at the larger d_n cells (seed 0): identical data,
    # optimizer-equivalence tolerance as above.
    for d_n in (64, 256):
        rng = np.random.default_rng(10_000 + 0)
        S_tr, N_tr, y_tr = generate(rng, N_TRAIN, d_n, RHO_TRAIN)
        S_id, N_id, y_id = generate(rng, N_TEST, d_n, RHO_TRAIN)
        S_sh, N_sh, y_sh = generate(rng, N_TEST, d_n, RHO_SHIFT)
        arms = {"X": (np.hstack([S_tr, N_tr]), np.hstack([S_id, N_id]),
                      np.hstack([S_sh, N_sh])),
                "S": (S_tr, S_id, S_sh), "N": (N_tr, N_id, N_sh)}
        for arm, (Ztr, Zid, Zsh) in arms.items():
            r = fourq.fit_logistic(Ztr, y_tr)
            ce_id = fourq.ce_bits(r, Zid, y_id)
            ce_sh = fourq.ce_bits(r, Zsh, y_sh)
            cid, csh = committed[(0, d_n, arm)]
            print(f"  d_n={d_n} arm={arm} seed=0: |dID| {abs(ce_id-cid):.2e} "
                  f"|dShift| {abs(ce_sh-csh):.2e} bits")
            assert abs(ce_id - cid) < TOL_PER_SEED_ID, (d_n, arm, ce_id, cid)
            assert abs(ce_sh - csh) < TOL_PER_SEED_SHIFT, (d_n, arm, ce_sh, csh)


# ===========================================================================
# Anchor 2 — analytic closed forms
# ===========================================================================
def _h2_bits(p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / LOG2


def test_anchor2_analytic_gaussian():
    # (a) Binary symmetric Gaussian: Z|Y=±1 ~ N(±mu, I_d), priors 1/2.
    # Bayes posterior p(Y=+1|z) = sigma(2 mu.z) — logistic-linear, so the
    # declared class contains Bayes and CE*_log(P) = H(Y|Z) exactly.
    # Closed form via 64-node probabilists' Gauss-Hermite:
    #   H(Y|Z) = E_{zeta~N(0,1)}[ h2(sigma(2 m (m + zeta))) ],  m = ||mu||.
    d = 4
    mu = np.full(d, 0.5)               # m = 1
    m = float(np.linalg.norm(mu))
    x, w = np.polynomial.hermite_e.hermegauss(64)
    w = w / w.sum()
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    bayes_ce = float(np.sum(w * _h2_bits(sig(2.0 * m * (m + x)))))

    rng = np.random.default_rng(424242)

    def draw(n):
        y = (rng.random(n) < 0.5).astype(float)
        Z = (2 * y - 1)[:, None] * mu[None, :] + rng.standard_normal((n, d))
        return Z, y

    Z_eval, y_eval = draw(400_000)
    n_grid = [500, 2000, 8000, 32000]
    gaps = []
    print(f"  binary Gaussian: Bayes CE = H(Y|Z) = {bayes_ce:.5f} bits")
    for n in n_grid:
        Z, y = draw(n)
        r = fourq.fit_logistic(Z, y)
        ce, det = fourq.ce_bits_detail(r, Z_eval, y_eval)
        expected_excess = (d + 1) / (2 * n * LOG2)
        gaps.append(ce - bayes_ce)
        print(f"    n={n:6d}: CEhat {ce:.5f}, gap {ce - bayes_ce:+.5f} "
              f"(asymptotic MLE excess ~ {expected_excess:.5f})")
    # eval-set noise scale (shared across the n grid; same eval set)
    se_eval = 0.0015  # ~sd(per-sample loss)/sqrt(400k), conservative
    assert all(g > -3 * se_eval for g in gaps), \
        "estimate beat the Bayes CE beyond eval noise — impossible"
    assert gaps[-1] < 0.005, f"n=32000 gap {gaps[-1]} not converged"
    assert gaps[0] > gaps[-1], "no convergence from n=500 to n=32000"

    # (b) Multi-class planted softmax: Z ~ N(0, I_6), p(y|z) = softmax(Wz),
    # K = 4. Bayes CE = E[H(softmax(WZ))] by exact-posterior Monte Carlo
    # (2e6 fresh draws; MC SE ~ 3e-4 bits). Exercises the multinomial path.
    K, dz = 4, 6
    Wm = np.random.default_rng(7).normal(size=(K, dz)) * 1.2

    def draw_mc(n):
        Z = rng.standard_normal((n, dz))
        logits = Z @ Wm.T
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        cum = np.cumsum(P, axis=1)
        y = (rng.random((n, 1)) > cum).sum(axis=1)
        return Z, y, P

    Zb, _, Pb = draw_mc(2_000_000)
    bayes_mc = float(np.mean(-(Pb * np.log(np.clip(Pb, 1e-300, 1))).sum(axis=1)) / LOG2)
    Ztr, ytr, _ = draw_mc(40_000)
    Zev, yev, _ = draw_mc(400_000)
    r = fourq.fit_logistic(Ztr, ytr)
    assert r.settings["class_output_cardinality"] == K
    ce = fourq.ce_bits(r, Zev, yev)
    gap = ce - bayes_mc
    floors = fourq.class_floors(yev)
    print(f"  4-class softmax: Bayes CE {bayes_mc:.5f} (MC), CEhat {ce:.5f}, "
          f"gap {gap:+.5f} (expected excess ~ "
          f"{(K - 1) * (dz + 1) / (2 * 40_000 * LOG2):.5f}); "
          f"H(Y) = {floors['H_Y_bits']:.4f}, log2|Y| = "
          f"{floors['log2_cardinality_bits']:.4f}")
    assert abs(gap) < 0.008, f"multiclass gap {gap}"
    assert ce < floors["H_Y_bits"], "readout no better than marginal floor"


# ===========================================================================
# Anchor 3 — schema discipline
# ===========================================================================
def test_anchor3_schema_discipline():
    # A real (tiny) battery output must serialize cleanly...
    rng = np.random.default_rng(1)
    n, d = 1200, 3
    y = (rng.random(n) < 0.5).astype(int)
    Z = (2 * y - 1)[:, None] * 0.8 + rng.standard_normal((n, d))
    yq = (rng.random(n) < 0.5).astype(int)
    Zq = -(2 * yq - 1)[:, None] * 0.8 + rng.standard_normal((n, d))  # shifted
    b = fourq.battery(Z[:800], y[:800], Z[800:], y[800:], Zq, yq,
                      k_refit_folds=3, proxy_validated=True,
                      license_ref="anchor-2 analytic validation (this suite)")
    rows = b.measurements(context={"seed": 1, "arm": "demo"})
    assert fourq.declaration_problems(rows) == []
    frozen = [r for r in rows if r["quantity"] == fourq.Q_CEQ]
    assert len(frozen) == 1 and frozen[0]["not_an_entropy"] is True, \
        "frozen-shift row must carry not_an_entropy=True"
    assert frozen[0]["fit_distribution"] == "P"
    with tempfile.TemporaryDirectory() as td:
        fourq.write_results_csv(rows, os.path.join(td, "ok.csv"))
        fourq.write_results_json(rows, os.path.join(td, "ok.json"))
        assert os.path.exists(os.path.join(td, "ok.csv"))

        def refuses(bad_rows, why):
            try:
                fourq.write_results_csv(bad_rows, os.path.join(td, "bad.csv"))
            except fourq.QuantityDeclarationError:
                print(f"  serializer refused: {why} -- OK")
                return
            raise AssertionError(f"serializer accepted a table with {why}")

        # 1. missing quantity declaration (§0-bis criterion 5)
        bad = [dict(r) for r in rows]
        del bad[0]["quantity"]
        refuses(bad, "a missing quantity declaration")
        # 2. missing distribution
        bad = [dict(r) for r in rows]
        bad[0]["distribution"] = ""
        refuses(bad, "an empty distribution declaration")
        # 3. frozen-shift row without not_an_entropy=True
        bad = [dict(r) for r in rows]
        for r in bad:
            if r["quantity"] == fourq.Q_CEQ:
                r["not_an_entropy"] = False
        refuses(bad, "a CE_Q(r_P) row not marked not_an_entropy")
        # 4. unlicensed CE* declaration (RULE item 2)
        bad = [dict(r) for r in rows]
        for r in bad:
            if r["quantity"] == fourq.Q_CESTAR:
                r["licensed"] = False
                r["license_ref"] = ""
        refuses(bad, "an unlicensed CE*_A declaration")
        # 5. the §6.5 tell: an 'entropy' above log2|Y|
        bad = [dict(rows[0])]
        bad[0].update(measurement="H_conflated", quantity=fourq.Q_H,
                      value_bits=2.34, not_an_entropy=False, licensed=True,
                      license_ref="x", log2_cardinality_bits=1.0)
        refuses(bad, "a declared entropy above log2|Y| (the §6.5 tell)")
        # 6. undeclared quantity name
        bad = [dict(rows[0])]
        bad[0]["quantity"] = "H proxy"
        refuses(bad, "the inadmissible bare 'H proxy' declaration")

    # battery itself refuses a license claim without a reference
    try:
        fourq.battery(Z[:800], y[:800], Z[800:], y[800:], Zq, yq,
                      k_refit_folds=3, proxy_validated=True, license_ref="")
        raise AssertionError("battery accepted proxy_validated without ref")
    except ValueError:
        print("  battery refused proxy_validated=True with empty "
              "license_ref -- OK")

    # unvalidated battery rows are declared raw_heldout_CE, and still pass
    b_raw = fourq.battery(Z[:800], y[:800], Z[800:], y[800:], Zq, yq,
                          k_refit_folds=3)
    rows_raw = b_raw.measurements()
    assert fourq.declaration_problems(rows_raw) == []
    assert all(r["quantity"] == fourq.Q_RAW for r in rows_raw
               if r["measurement"] in ("CEhat_ID", "CEhat_Q_refit")), \
        "unvalidated CEs must be declared raw_heldout_CE (RULE item 2)"


# ===========================================================================
# Small unit checks (floors, PQ helpers, noise floor)
# ===========================================================================
def test_units():
    f = fourq.class_floors([0, 0, 0, 1])
    assert abs(f["H_Y_bits"] - 0.81128) < 1e-4
    assert f["log2_cardinality_bits"] == 1.0
    assert f["majority_prior"] == 0.75
    assert abs(f["majority_class_ce_bits"] - (-math.log2(0.75))) < 1e-12
    assert f["perfect_prediction_floor_bits"] == 0.0

    assert abs(fourq.pq2_cv([1.0, 2.0, 3.0]) - 0.5) < 1e-12
    assert fourq.pq2_check([1.0, 2.0, 3.0])["passed"] is True
    assert fourq.pq2_check([1.0, 1.01, 0.99])["passed"] is False

    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.5).astype(int)
    Z = (2 * y - 1)[:, None] * 0.7 + rng.standard_normal((2000, 2))
    nf = fourq.noise_floor(Z, y, n_splits=5, seed=0)
    assert len(nf["fold_ces_bits"]) == 5
    assert nf["se_bits"] > 0 and nf["delta_bits"] == 2.0 * nf["se_bits"]
    assert 0.3 < nf["mean_bits"] < 1.0  # sane range for this construction
    print(f"  noise_floor demo: mean {nf['mean_bits']:.4f}, "
          f"SE {nf['se_bits']:.5f}, delta(2xSE) {nf['delta_bits']:.5f} bits")


# ===========================================================================
def main():
    tests = [
        ("units (floors, PQ helpers, noise floor)", test_units),
        ("anchor 3: schema discipline", test_anchor3_schema_discipline),
        ("anchor 2: analytic closed forms", test_anchor2_analytic_gaussian),
        ("anchor 1: track3 toy (committed seeds)", test_anchor1_track3_toy),
    ]
    failures = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        print(f"[ RUN  ] {name}")
        try:
            fn()
            print(f"[ PASS ] {name} ({time.perf_counter() - t0:.1f}s)")
        except Exception:
            failures += 1
            print(f"[ FAIL ] {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} test groups passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
