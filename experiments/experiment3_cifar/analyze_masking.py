# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "scikit-learn>=1.4"]
# ///
r"""
analyze_masking.py — the ONCE-ONLY analysis for `expB-masking-real-v1`.

Runs EXACTLY ONCE on the committed protocol (PREREGISTRATION.md §6). It loads
all 80 trained runs' penultimate features (the four §4 sets per cell) from
/vault/datasets/features/expB/, runs the §5.1 planted-posterior proxy-validation
selftest (τ = 0.05 bits) FIRST, then for every (ρ, seed, arm) cell runs the §4
fourq battery under BOTH shifts (decorrelate gated, anticorrelate descriptive),
aggregates seed-means with per-cell δ = max(2·SE_seed, 0.10), evaluates the §5.3
gates G-B1..G-B4 + the C-B5 coincidence check (HALT on violation) + the §5.6
PQ checklist + §5.7 declaration checklist + §5.8 §6.5 self-audit, and writes
results.csv / results_summary.json / a machine summary + RESULTS.md skeleton.

CONTRACT: `--selftest` runs the §5.1 planted-posterior validation ALONE and
touches NO trained features (the pre-run gate). The full once-only run requires
`--run` and is the analysis agent's job, NOT run-B's.

Pinned dependencies (imported by path, sha256-checked at startup):
  ../lib/fourq.py        8b66bc7d07c2f0d1cc11180a5c629b156c014c413cb1b5644a6e1539abdd255a
  ./spurious_cifar.py    ab8f6e61d54be4ea13b23aee326aabccb2f272c4bfffa4431ace45b70c9050ba
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.normpath(os.path.join(HERE, "..", "lib"))
sys.path.insert(0, LIB)
sys.path.insert(0, HERE)

import fourq  # noqa: E402

# ---------------------------------------------------------------------------
# Pins (PREREGISTRATION §2, §6.4)
# ---------------------------------------------------------------------------
FOURQ_SHA256 = "8b66bc7d07c2f0d1cc11180a5c629b156c014c413cb1b5644a6e1539abdd255a"
SPURIOUS_SHA256 = "ab8f6e61d54be4ea13b23aee326aabccb2f272c4bfffa4431ace45b70c9050ba"

# Committed grid / readout / thresholds
RHO_GRID = [0.7, 0.9, 0.95, 0.99]
SEEDS = list(range(10))
ARMS = ["naive", "lawful"]
SHIFTS = ["decorrelate", "anticorrelate"]   # decorrelate gated; anti descriptive
GATED_SHIFT = "decorrelate"

READOUT_C = 1.0                 # §4 recorded-finite C (override of fourq default)
STANDARDIZE = True              # §4
READOUT_TOL = 1e-10
READOUT_MAX_ITER = 10000
FIT_OPTS = {"C": READOUT_C, "tol": READOUT_TOL, "max_iter": READOUT_MAX_ITER}
K_REFIT_FOLDS = 5               # §4

MATERIALITY_FLOOR = 0.10        # §5.2
TAU_SELFTEST = 0.05             # §5.1 planted-posterior tolerance (bits)
G_B2_MIN = 0.5                  # §5.3 G-B2 minimum naive Δ_transfer (bits)
G_B2_RHOS = [0.9, 0.95, 0.99]   # §5.3 G-B2 gated ρ
UNDERPOWERED_DELTA = 0.25       # §5.5 cap
H_Y_BITS = math.log2(10)        # 3.3219...; uniform labels
LICENSE_REF = "expB-masking-real-v1 §5.1 planted-posterior PASS"
N_BOOTSTRAP = 1000              # §4 masking-curve CIs (descriptive)

FEAT_DIR_DEFAULT = "/vault/datasets/features/expB"
K = 10
D = 256
K_READOUT = K * (D + 1)          # 2570

# §1.1 / pilot cue-only semi-analytic elevations under decorrelate (construction
# math, NOT measurements; used only for the §1.1 fraction-of-envelope statistic
# and the G-B2 0.5 justification — never as a gate target).
ELEV_DEC_CUEONLY = {0.7: 2.635390453667264, 0.9: 5.071880002307709,
                    0.95: 6.305174637653008, 0.99: 8.721360643154528}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pins():
    got_f = sha256_file(os.path.join(LIB, "fourq.py"))
    got_s = sha256_file(os.path.join(HERE, "spurious_cifar.py"))
    if got_f != FOURQ_SHA256:
        sys.exit(f"FATAL: fourq.py sha256 {got_f} != pinned {FOURQ_SHA256}")
    if got_s != SPURIOUS_SHA256:
        sys.exit(f"FATAL: spurious_cifar.py sha256 {got_s} != pinned {SPURIOUS_SHA256}")
    return got_f, got_s


# ===========================================================================
# §5.1 — planted-posterior proxy validation (the CE* license)
#
# AMENDED 2026-06-11 (§5.9, the experiment's ONE permitted amendment) per ruling
# queue/decisions/answered/runB-expB-s51-selftest-excess.md (Option 2). The PASS
# criterion is the recovery of CE* DIFFERENCES (the estimand every gate consumes),
# not ABSOLUTE CE* recovery. Rationale (ruling + escalation
# queue/decisions/done/runB-expB-s51-selftest-excess.md): at n=5000, D=256, K=10
# the held-out CE of the C=1.0 multinomial-logistic MLE carries an intrinsic
# finite-sample excess ≈ K_readout/(2 n ln2) = 2570/(2*5000*ln2) ≈ 0.371 bits
# above the true CE*_log. This common-mode excess defeats ABSOLUTE recovery at
# τ=0.05 (the original FAIL, preserved below) but CANCELS to first order in the
# DIFFERENCES (Δ_info, Δ_transfer, arm gaps) the gates G-B1..G-B4 / C-B5 actually
# evaluate at matched (n, D, K). The amended selftest validates that cancellation
# directly. ABSOLUTE CEhat values therefore serialize as raw_heldout_CE with the
# bias disclosed; DIFFERENCE quantities carry the CE*-difference license.
# ===========================================================================

# Disclosed absolute-CE bias (ruling condition 2): the intrinsic finite-sample
# excess of the held-out CE of the K×(D+1) multinomial-logistic MLE at the §4
# settings. Every raw_heldout_CE absolute sits ~this far ABOVE the true CE*_log;
# it is common-mode and cancels in differences.
ABS_CE_BIAS_BITS = K_READOUT / (2.0 * 5000 * math.log(2.0))   # ≈ 0.371 bits


def _planted_softmax_dataset(n, d, k, *, rng, r_signal=8, margin=1.0,
                             W=None, b=None, label_noise=0.0):
    """Generate (Z, y) in R^d whose Bayes posterior is EXACTLY (a label-noise
    blend of) a softmax-linear law, so its population CE*_log = H(Y|Z) is
    computable from the planted weights.

    The class signal lives in a LOW-RANK informative subspace (rank `r_signal`,
    embedded in R^d via a random orthonormal basis); the remaining d - r_signal
    feature directions are pure noise. This mirrors learned CNN penultimate
    features, where class information concentrates in a low-dimensional subspace
    while the ambient dimension is large. `margin` scales the planted logit
    weights, setting how peaked the base posterior is.

    `W`, `b` (optional): reuse a FIXED planted softmax law instead of redrawing
    one. The difference-recovery selftest (§5.1 amended) holds (W, b) — hence the
    Z-geometry and the readout-fit regime, hence the finite-sample excess — FIXED
    across the members of a pair, and moves CE* ONLY via `label_noise`, so the
    common-mode excess cancels in the recovered difference.

    `label_noise` (lambda in [0,1]): the sampling posterior is blended toward
    uniform, P_eff = (1-lambda)*softmax + lambda/K. This is an explicit label-noise
    channel: it RAISES CE* by a known amount while leaving the feature geometry and
    posterior peakedness of the signal component (hence the MLE excess) essentially
    unchanged — the controlled, excess-matched way to plant a known CE* DIFFERENCE.

    Returns (Z, y, ce_star_bits, W, b). ce_star is the population soft-readout
    optimum of P_eff, estimated on a 200k companion sample (MC SE ~ a few e-4 bits,
    well inside τ).
    """
    if W is None or b is None:
        # orthonormal embedding of the r_signal-dim signal subspace into R^d
        A = rng.standard_normal((d, r_signal))
        Q, _ = np.linalg.qr(A)                   # (d, r_signal), orthonormal cols
        Ws = rng.standard_normal((k, r_signal)) * margin
        b = rng.standard_normal(k) * (0.3 * margin)
        W = Ws @ Q.T                             # (k, d): weights nonzero only in subspace

    def posterior(Z):
        logits = Z @ W.T + b
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p = p / p.sum(axis=1, keepdims=True)
        if label_noise:
            p = (1.0 - label_noise) * p + label_noise / k
        return p

    Z_pop = rng.standard_normal((200_000, d))
    P_pop = posterior(Z_pop)
    ce_star = float(np.mean(-np.sum(P_pop * np.log2(np.clip(P_pop, 1e-300, 1.0)),
                                    axis=1)))

    Z = rng.standard_normal((n, d))
    P = posterior(Z)
    cum = np.cumsum(P, axis=1)
    y = (rng.random((n, 1)) > cum).sum(axis=1).astype(int)
    return Z.astype(np.float32), y, ce_star, W, b


# --- amended difference-recovery design (ruling condition 1) ----------------
# Planted pairs at matched (n=5000, D=256, K=10). Each pair shares ONE fixed
# softmax law (W, b); the two members differ only in label_noise (lambda), so
# CE* differs by a known amount while the MLE excess stays common-mode. Each
# member's CE* and recovered CEhat are AVERAGED over SELFTEST_NDRAW independent
# planted draws — exactly the seed-averaging the gates apply over the 10 seeds,
# which suppresses the per-realization excess wobble that is NOT part of the
# estimand. PASS iff every planted difference is recovered within τ = 0.05 bits.
SELFTEST_NDRAW = 6
SELFTEST_PAIRS = [
    # (pair name, member-A spec, member-B spec); members carry (label_name,
    # label_noise, seed_bank). One ~0.5-bit gate-relevant pair, one ~0 null pair.
    ("gate-relevant (~0.5 bit, Delta-recovery)",
     ("lam0.30", 0.30, 10000), ("lam0.10", 0.10, 20000)),
    ("null (~0 bit, calibration)",
     ("lam0.25", 0.25, 30000), ("lam0.25b", 0.25, 40000)),
]
# Fixed shared planted law for every pair (drawn once from SELFTEST_WSEED so the
# excess regime is identical across members and the difference is excess-matched).
SELFTEST_WSEED = 2026
SELFTEST_MARGIN = 1.2


def _planted_W(rng, *, margin=SELFTEST_MARGIN, r_signal=8):
    A = rng.standard_normal((D, r_signal))
    Q, _ = np.linalg.qr(A)
    Ws = rng.standard_normal((K, r_signal)) * margin
    b = rng.standard_normal(K) * (0.3 * margin)
    return Ws @ Q.T, b


def _member_family(W, b, label_noise, seed0, ndraw):
    """Mean CE* and mean recovered CEhat over `ndraw` independent planted draws of
    ONE pair member (fixed law W,b; fixed label_noise; disjoint seed bank)."""
    css, chs, convs, unseen = [], [], [], 0
    for j in range(ndraw):
        rng = np.random.default_rng(seed0 + j)
        Z, y, ce_star, _, _ = _planted_softmax_dataset(
            10000, D, K, rng=rng, W=W, b=b, label_noise=label_noise)
        r = fourq.fit_logistic(Z[:5000], y[:5000], standardize=STANDARDIZE, **FIT_OPTS)
        ce_hat, det = fourq.ce_bits_detail(r, Z[5000:], y[5000:])
        css.append(ce_star); chs.append(ce_hat)
        convs.append(r.diagnostics["converged"]); unseen += det["n_unseen_labels"]
    return {"ce_star_mean": float(np.mean(css)),
            "ce_hat_mean": float(np.mean(chs)),
            "excess_mean": float(np.mean(chs) - np.mean(css)),
            "all_converged": bool(all(convs)),
            "n_unseen_labels": int(unseen), "ndraw": ndraw}


# Original ABSOLUTE-recovery selftest result, preserved per ruling condition 4
# (the amendment supersedes the criterion, not the history). This is the FAIL
# table from the escalation / PRERUN_NOTE_2026-06-11.md, recorded verbatim so the
# selftest record always carries the superseded criterion's outcome alongside the
# amended one. ABSOLUTE recovery is unreachable at τ=0.05 because of ABS_CE_BIAS_BITS.
ABSOLUTE_RECOVERY_HISTORY = {
    "criterion": "SUPERSEDED 2026-06-11 (§5.9): absolute CE*_log recovery within "
                 "τ=0.05 bits at n=5000, D=256, K=10",
    "result": "FAIL — intrinsic finite-sample MLE excess ≈ ABS_CE_BIAS_BITS "
              "(~0.371 bits) defeats absolute recovery; this is the §5.9 trigger",
    "tau_bits": TAU_SELFTEST,
    "cases": [
        {"case": "planted-sufficient", "ce_star_bits": 1.52,
         "ce_hat_recovered_bits": 2.07, "abs_err_bits": 0.54, "passed": False},
        {"case": "planted-sufficient-mild", "ce_star_bits": 2.31,
         "ce_hat_recovered_bits": 2.73, "abs_err_bits": 0.42, "passed": False},
        {"case": "planted-insufficient-control", "ce_star_bits": 3.32,
         "ce_hat_recovered_bits": 3.72, "abs_err_bits": 0.39, "passed": False},
    ],
    "dimension_sweep_excess_vs_analytic_K_Dp1_over_2n_ln2": {
        "D=256": {"excess": 0.54, "analytic": 0.371},
        "D=64": {"excess": 0.081, "analytic": 0.094},
        "D=16": {"excess": 0.054, "analytic": 0.025},
        "D=8": {"excess": 0.007, "analytic": 0.013},
    },
    "ref_ruling": "queue/decisions/answered/runB-expB-s51-selftest-excess.md",
    "ref_escalation": "queue/decisions/done/runB-expB-s51-selftest-excess.md",
}


def run_selftest(rng_seed=0, verbose=True):
    """§5.1 (AMENDED 2026-06-11, §5.9): the CE*-DIFFERENCE recovery selftest.

    Fits the declared A_log pipeline (C=1.0, standardize=True, the §4 split sizes
    5000 fit / 5000 eval, D=256, K=10) on planted softmax-linear posteriors whose
    CE*_log is analytically known, and validates that planted CE* DIFFERENCES are
    recovered within τ = 0.05 bits — the estimand the gates consume, where the
    common-mode finite-sample MLE excess (~ABS_CE_BIAS_BITS) cancels. Includes one
    ~0.5-bit gate-relevant pair and one near-zero null pair (ruling condition 1).
    The superseded ABSOLUTE-recovery FAIL table is preserved in the record (ruling
    condition 4). ABSOLUTE CEs are reported as raw_heldout_CE with the bias
    disclosed (ruling condition 2). Touches NO trained features.

    `rng_seed` perturbs the per-member draw seed banks (kept disjoint), so the
    selftest is reproducible and its margin can be probed; the fixed planted law
    (SELFTEST_WSEED) and the pair structure are committed constants.
    """
    Wrng = np.random.default_rng(SELFTEST_WSEED)
    W, b = _planted_W(Wrng)

    members = {}
    pairs = []
    for pair_name, (na, lna, sa), (nb, lnb, sb) in SELFTEST_PAIRS:
        if na not in members:
            members[na] = _member_family(W, b, lna, sa + rng_seed, SELFTEST_NDRAW)
        if nb not in members:
            members[nb] = _member_family(W, b, lnb, sb + rng_seed, SELFTEST_NDRAW)
        A, B = members[na], members[nb]
        d_star = A["ce_star_mean"] - B["ce_star_mean"]
        d_hat = A["ce_hat_mean"] - B["ce_hat_mean"]
        err = abs(d_hat - d_star)
        passed = err <= TAU_SELFTEST
        pairs.append({
            "pair": pair_name, "member_A": na, "member_B": nb,
            "ce_star_A_bits": A["ce_star_mean"], "ce_star_B_bits": B["ce_star_mean"],
            "ce_hat_A_bits": A["ce_hat_mean"], "ce_hat_B_bits": B["ce_hat_mean"],
            "excess_A_bits": A["excess_mean"], "excess_B_bits": B["excess_mean"],
            "planted_diff_bits": d_star, "recovered_diff_bits": d_hat,
            "diff_abs_err_bits": err, "tolerance_bits": TAU_SELFTEST,
            "passed": bool(passed), "ndraw": SELFTEST_NDRAW,
            "n_fit": 5000, "n_eval": 5000, "D": D, "K": K,
            "C": READOUT_C, "standardize": STANDARDIZE,
            "all_converged": A["all_converged"] and B["all_converged"],
            "n_unseen_labels": A["n_unseen_labels"] + B["n_unseen_labels"],
        })
        if verbose:
            print(f"  [{pair_name}] planted ΔCE*={d_star:+.4f} "
                  f"recovered ΔCEhat={d_hat:+.4f} |err|={err:.4f} "
                  f"(<= {TAU_SELFTEST}); excess_A={A['excess_mean']:+.3f} "
                  f"excess_B={B['excess_mean']:+.3f} -> "
                  f"{'PASS' if passed else 'FAIL'}")
    overall = all(p["passed"] for p in pairs)
    if verbose:
        print(f"  (disclosed absolute-CE bias ABS_CE_BIAS_BITS = "
              f"K_readout/(2n ln2) = {K_READOUT}/(2*5000*ln2) = "
              f"{ABS_CE_BIAS_BITS:.4f} bits; common-mode, cancels in differences)")
    return {"selftest": "expB §5.1 CE*-DIFFERENCE recovery (AMENDED 2026-06-11, §5.9)",
            "criterion": "recover planted CE* DIFFERENCES within τ=0.05 bits "
                         "(the gate estimand; common-mode MLE excess cancels)",
            "tau_bits": TAU_SELFTEST, "license_ref": LICENSE_REF,
            "abs_ce_bias_bits_disclosed": ABS_CE_BIAS_BITS,
            "abs_ce_bias_formula": "K_readout/(2 n ln2) = "
                                   f"{K_READOUT}/(2*5000*ln(2))",
            "licensing_split": {
                "differences_licensed": ["delta_info", "delta_transfer",
                                         "arm CEhat_ID gaps"],
                "absolutes_raw": ["CEhat_ID", "CEhat_Q_refit"],
                "note": "differences carry the CE*-difference license citing this "
                        "amended selftest; absolute CEhat serialize as "
                        "raw_heldout_CE with abs_ce_bias_bits_disclosed",
            },
            "member_families": members,
            "pairs": pairs,
            "absolute_recovery_history_PRESERVED": ABSOLUTE_RECOVERY_HISTORY,
            "ref_ruling": "queue/decisions/answered/runB-expB-s51-selftest-excess.md",
            "ref_escalation": "queue/decisions/done/runB-expB-s51-selftest-excess.md",
            "PASS": bool(overall)}


# ===========================================================================
# Feature loading
# ===========================================================================
def cell_tag(arm, rho, seed):
    return f"{arm}_rho{rho}_seed{seed}"


def load_cell(feat_dir, arm, rho, seed):
    tag = cell_tag(arm, rho, seed)
    fpath = os.path.join(feat_dir, f"features_{tag}.npz")
    mpath = os.path.join(feat_dir, f"run_{tag}.json")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"missing feature file for cell {tag}: {fpath}")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"missing run-marker for cell {tag}: {mpath}")
    d = np.load(fpath)
    with open(mpath) as f:
        marker = json.load(f)
    return d, marker


def inventory(feat_dir):
    """Return (present, missing) cell lists — what the analysis agent verifies."""
    present, missing = [], []
    for arm, rho, seed in product(ARMS, RHO_GRID, SEEDS):
        tag = cell_tag(arm, rho, seed)
        fpath = os.path.join(feat_dir, f"features_{tag}.npz")
        mpath = os.path.join(feat_dir, f"run_{tag}.json")
        (present if (os.path.exists(fpath) and os.path.exists(mpath))
         else missing).append(tag)
    return present, missing


# ===========================================================================
# Per-cell battery
# ===========================================================================
def cell_battery(d, shift, proxy_validated, license_ref):
    """Run the §4 battery for one cell under one shift.
    P-fit set = ZA_ID (half-A, ID coupling); P-held-out = ZB_ID; Q set = ZB_<shift>.
    The frozen readout is never evaluated on re-tinted versions of fit images
    (A vs B are disjoint by construction)."""
    ZA, yA = d["ZA_ID"], d["yA"]
    ZB, yB = d["ZB_ID"], d["yB"]
    ZQ = d[f"ZB_{'dec' if shift == 'decorrelate' else 'anti'}"]
    yQ = yB  # the Q set is half-B re-tinted; labels y are unchanged by the shift
    return fourq.battery(ZA, yA, ZB, yB, ZQ, yQ,
                         k_refit_folds=K_REFIT_FOLDS,
                         proxy_validated=proxy_validated, license_ref=license_ref,
                         standardize=STANDARDIZE, seed=0, fit_opts=FIT_OPTS)


def bootstrap_ceq(d, shift, n_boot=N_BOOTSTRAP, seed=0):
    """1000-resample nonparametric bootstrap of CE_Q(r_P) over the Q samples
    (§4 masking-curve CIs, descriptive). Refits r_P once on ZA_ID, then resamples
    the Q set."""
    ZA, yA = d["ZA_ID"], d["yA"]
    ZQ = d[f"ZB_{'dec' if shift == 'decorrelate' else 'anti'}"]
    yQ = d["yB"]
    r = fourq.fit_logistic(ZA, yA, standardize=STANDARDIZE, **FIT_OPTS)
    rng = np.random.default_rng(seed)
    n = len(yQ)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = fourq.ce_bits(r, ZQ[idx], yQ[idx])
    return {"mean": float(vals.mean()),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5))}


# ===========================================================================
# Aggregation + gates
# ===========================================================================
def seed_mean_se(values):
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return mean, se


def delta_cell(se_seed):
    return max(2.0 * se_seed, MATERIALITY_FLOOR)


def bootstrap_ci_mean(values, n_boot=2000, seed=0):
    """Bootstrap CI of a seed-mean (cross-seed aggregation CIs)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_full(feat_dir, outdir, selftest_record):
    os.makedirs(outdir, exist_ok=True)
    present, missing = inventory(feat_dir)
    if missing:
        sys.exit(f"FATAL: {len(missing)} cells missing; cannot run the once-only "
                 f"analysis. Missing: {missing}")

    proxy_ok = selftest_record["PASS"]
    lic = LICENSE_REF if proxy_ok else ""

    # §5.9 AMENDMENT (2026-06-11) LICENSING SPLIT (ruling condition 2):
    # the amended §5.1 selftest validates CE* DIFFERENCE recovery, NOT absolute
    # recovery. So even on PASS, ABSOLUTE CEhat (CEhat_ID, CEhat_Q_refit) carry NO
    # CE*-point license — they serialize as raw_heldout_CE with the disclosed bias
    # (~ABS_CE_BIAS_BITS, common-mode). The pinned fourq battery licenses absolutes
    # and differences together via one flag, so we run the battery UNLICENSED
    # (proxy_validated=False -> absolutes raw) and then RE-LICENSE only the
    # difference rows (delta_info, delta_transfer) below. Gates read numeric agg
    # values, not row license flags, so gate evaluation is unaffected by the split.
    DIFF_MEASUREMENTS = ("delta_info", "delta_transfer")
    DIFF_LICENSE_REF = (LICENSE_REF + " (CE*-DIFFERENCE recovery; absolutes are "
                        "raw_heldout_CE, bias≈%.3f bits disclosed)"
                        % ABS_CE_BIAS_BITS)

    # ---- per (cell × shift) batteries + bootstraps -------------------------
    # store[arm][rho][shift] = list over seeds of dict(per-cell quantities)
    store = {a: {r: {s: [] for s in SHIFTS} for r in RHO_GRID} for a in ARMS}
    rows = []          # declaration-complete per-measurement rows
    lawful_noise = {}  # per (rho, seed) lawful in-run noise floor on ZB_ID
    convergence = []   # convergence diagnostics
    sanity = {}        # std_test_acc per cell

    for arm, rho, seed in product(ARMS, RHO_GRID, SEEDS):
        d, marker = load_cell(feat_dir, arm, rho, seed)
        sanity[cell_tag(arm, rho, seed)] = marker.get("std_test_acc")
        for shift in SHIFTS:
            # battery run UNLICENSED so ABSOLUTE CEhat rows serialize raw (split).
            br = cell_battery(d, shift, False, "")
            boot = bootstrap_ceq(d, shift)
            convergence.append({"cell": cell_tag(arm, rho, seed), "shift": shift,
                                "settings": br.settings})
            store[arm][rho][shift].append({
                "seed": seed,
                "CEhat_ID": br.CEhat_ID, "CE_Q_frozen": br.CE_Q_frozen,
                "CEhat_Q_refit": br.CEhat_Q_refit,
                "delta_info": br.delta_info, "delta_transfer": br.delta_transfer,
                "boot": boot, "n_unseen_Q": br.n_unseen_labels_Q,
            })
            ctx = {"arm": arm, "rho": rho, "seed": seed, "shift": shift}
            cell_rows = br.measurements(context=ctx)
            if proxy_ok:
                # RE-LICENSE only the difference rows with the amended CE*-DIFFERENCE
                # license (ruling condition 2); absolute rows stay raw_heldout_CE.
                for rw in cell_rows:
                    if rw.get("measurement") == "delta_info":
                        rw["licensed"] = True
                        rw["license_ref"] = DIFF_LICENSE_REF
                        rw["definition"] = ("CEhat_Q_refit - CEhat_ID; estimates "
                                            "CE*_A(Q) - CE*_A(P) (information-"
                                            "accessibility change under shift). "
                                            "Licensed as a CE* DIFFERENCE by the "
                                            "§5.9 amended selftest; the absolute "
                                            "operands are raw_heldout_CE (common-"
                                            "mode bias≈%.3f bits cancels)."
                                            % ABS_CE_BIAS_BITS)
                    elif rw.get("measurement") == "delta_transfer":
                        rw["licensed"] = True
                        rw["license_ref"] = DIFF_LICENSE_REF
                        # delta_transfer is a frozen-transfer property (CE_Q_frozen
                        # is definitionally exact); its definition is unchanged.
                    elif rw.get("measurement") in ("CEhat_ID", "CEhat_Q_refit"):
                        rw["abs_ce_bias_bits_disclosed"] = ABS_CE_BIAS_BITS
            rows.extend(cell_rows)
        # lawful in-run noise floor (§4): recomputed on each lawful ZB_ID
        if arm == "lawful":
            nf = fourq.noise_floor(d["ZB_ID"], d["yB"], n_splits=10, seed=0,
                                   standardize=STANDARDIZE, fit_opts=FIT_OPTS)
            lawful_noise[cell_tag(arm, rho, seed)] = nf

    # ---- seed-mean aggregation per (arm, rho, shift, quantity) -------------
    agg = {}
    QUANTS = ["CEhat_ID", "CE_Q_frozen", "CEhat_Q_refit", "delta_info", "delta_transfer"]
    for arm, rho, shift in product(ARMS, RHO_GRID, SHIFTS):
        cells = store[arm][rho][shift]
        agg[(arm, rho, shift)] = {}
        for q in QUANTS:
            vals = [c[q] for c in cells]
            mean, se = seed_mean_se(vals)
            ci = bootstrap_ci_mean(vals)
            agg[(arm, rho, shift)][q] = {
                "seed_mean": mean, "se_seed": se, "delta_cell": delta_cell(se),
                "ci95": ci, "per_seed": vals,
            }

    # ---- §5.5 underpowered flags (gated cells only) ------------------------
    def underpowered(arm, rho, q, shift=GATED_SHIFT):
        return agg[(arm, rho, shift)][q]["delta_cell"] > UNDERPOWERED_DELTA

    # =====================================================================
    # GATES (§5.3) — all on seed-means; shift gates attach to decorrelate only
    # =====================================================================
    gates = {}

    # G-B1: masking-ID premise. PASS iff at every ρ:
    #   mean CEhat_ID(naive) - mean CEhat_ID(lawful) <= +δ
    g1 = {"name": "masking-ID premise", "per_rho": [], "requires_license": True,
          "licensed": proxy_ok,
          "license_note": "G-B1 consumes the naive−lawful CEhat_ID GAP (a CE* "
                          "DIFFERENCE at matched n,D,K): licensed by the §5.9 "
                          "amended difference-recovery selftest. The absolute "
                          "CEhat_ID per arm remain raw_heldout_CE (bias≈%.3f bits "
                          "disclosed, common-mode, cancels in the gap)."
                          % ABS_CE_BIAS_BITS}
    g1_pass = proxy_ok
    for rho in RHO_GRID:
        n_id = agg[("naive", rho, GATED_SHIFT)]["CEhat_ID"]
        l_id = agg[("lawful", rho, GATED_SHIFT)]["CEhat_ID"]
        gap = n_id["seed_mean"] - l_id["seed_mean"]
        delta = max(n_id["delta_cell"], l_id["delta_cell"])
        up = underpowered("naive", rho, "CEhat_ID") or underpowered("lawful", rho, "CEhat_ID")
        ok = gap <= delta
        g1["per_rho"].append({
            "rho": rho, "naive_CEhat_ID": n_id["seed_mean"],
            "lawful_CEhat_ID": l_id["seed_mean"], "gap": gap, "delta": delta,
            "band_lower": -l_id["seed_mean"], "band_upper": delta,
            "underpowered": up, "pass": (None if up else bool(ok))})
        if up:
            g1_pass = None if g1_pass else g1_pass
        elif not ok:
            g1_pass = False
    g1["PASS"] = (None if any(p["underpowered"] for p in g1["per_rho"]) and g1_pass is not False
                  else bool(g1_pass) if g1_pass is not None else None)
    if not proxy_ok:
        g1["PASS"] = None
        g1["note"] = ("§5.1 difference-recovery license absent: the naive−lawful "
                      "CEhat_ID gap is a raw CE difference; G-B1 not evaluable as "
                      "masking-ID")
    gates["G-B1"] = g1

    # G-B2: naive transfer brittleness. PASS iff naive mean Δ_transfer(decorr) >=
    #   0.5 bits at each ρ in {0.9,0.95,0.99}; underpowered if δ_cell > 0.25.
    g2 = {"name": "naive transfer brittleness", "min_bits": G_B2_MIN,
          "rhos": G_B2_RHOS, "per_rho": []}
    g2_pass = True
    for rho in G_B2_RHOS:
        cell = agg[("naive", rho, GATED_SHIFT)]["delta_transfer"]
        up = cell["delta_cell"] > UNDERPOWERED_DELTA  # §5.5: G-B2 needs δ<=0.25
        ok = cell["seed_mean"] >= G_B2_MIN
        frac = (cell["seed_mean"] / ELEV_DEC_CUEONLY[rho]) if ELEV_DEC_CUEONLY[rho] else None
        g2["per_rho"].append({
            "rho": rho, "naive_delta_transfer": cell["seed_mean"],
            "se": cell["se_seed"], "ci95": cell["ci95"],
            "delta_cell": cell["delta_cell"], "min": G_B2_MIN,
            "frac_of_envelope": frac, "cueonly_elev": ELEV_DEC_CUEONLY[rho],
            "underpowered": up, "pass": (None if up else bool(ok))})
        if up:
            g2_pass = None if g2_pass is not False else False
        elif not ok:
            g2_pass = False
    g2["PASS"] = (None if g2_pass is None else bool(g2_pass))
    gates["G-B2"] = g2

    # G-B3: lawful arm flat. PASS iff at every ρ:
    #   |mean Δ_info(lawful,decorr)| <= δ AND |mean Δ_transfer(lawful,decorr)| <= δ
    g3 = {"name": "lawful arm flat", "per_rho": []}
    g3_pass = True
    for rho in RHO_GRID:
        di = agg[("lawful", rho, GATED_SHIFT)]["delta_info"]
        dt = agg[("lawful", rho, GATED_SHIFT)]["delta_transfer"]
        delta = max(di["delta_cell"], dt["delta_cell"])
        up = (di["delta_cell"] > UNDERPOWERED_DELTA) or (dt["delta_cell"] > UNDERPOWERED_DELTA)
        ok = (abs(di["seed_mean"]) <= delta) and (abs(dt["seed_mean"]) <= delta)
        # anticorrelate reported descriptively beside
        di_a = agg[("lawful", rho, "anticorrelate")]["delta_info"]["seed_mean"]
        dt_a = agg[("lawful", rho, "anticorrelate")]["delta_transfer"]["seed_mean"]
        g3["per_rho"].append({
            "rho": rho, "lawful_delta_info_dec": di["seed_mean"],
            "lawful_delta_transfer_dec": dt["seed_mean"], "delta": delta,
            "lawful_delta_info_anti": di_a, "lawful_delta_transfer_anti": dt_a,
            "underpowered": up, "pass": (None if up else bool(ok))})
        if up:
            g3_pass = None if g3_pass is not False else False
        elif not ok:
            g3_pass = False
    g3["PASS"] = (None if g3_pass is None else bool(g3_pass))
    if proxy_ok:
        g3["license_note"] = ("G-B3 consumes lawful |Δ_info|, |Δ_transfer| (CE* "
                              "DIFFERENCES): licensed by the §5.9 amended "
                              "difference-recovery selftest.")
    else:
        g3["note"] = ("§5.1 difference-recovery license absent: Δ_info is a raw CE "
                      "difference; G-B3 info-side not licensed as CE* difference")
    gates["G-B3"] = g3

    # G-B4: masking-curve monotonicity of naive mean Δ_transfer(decorr) in ρ:
    #   Δ_transfer(ρ_{i+1}) >= Δ_transfer(ρ_i) - δ for every adjacent pair.
    g4 = {"name": "masking-curve monotonicity", "pairs": []}
    g4_pass = True
    seq = [(rho, agg[("naive", rho, GATED_SHIFT)]["delta_transfer"]) for rho in RHO_GRID]
    for (r0, c0), (r1, c1) in zip(seq[:-1], seq[1:]):
        delta = max(c0["delta_cell"], c1["delta_cell"])
        up = (c0["delta_cell"] > UNDERPOWERED_DELTA) or (c1["delta_cell"] > UNDERPOWERED_DELTA)
        ok = c1["seed_mean"] >= c0["seed_mean"] - delta
        g4["pairs"].append({"rho_lo": r0, "rho_hi": r1,
                            "dt_lo": c0["seed_mean"], "dt_hi": c1["seed_mean"],
                            "delta": delta, "underpowered": up,
                            "pass": (None if up else bool(ok))})
        if up:
            g4_pass = None if g4_pass is not False else False
        elif not ok:
            g4_pass = False
    g4["PASS"] = (None if g4_pass is None else bool(g4_pass))
    gates["G-B4"] = g4

    # C-B5: ρ=0.9 dec/anti coincidence per arm. |mean CE_Q^dec - mean CE_Q^anti| <= δ.
    #   Violation => pipeline bug: HALT (raise), disclose.
    c5 = {"name": "rho=0.9 dec/anti coincidence", "is_gate": False, "per_arm": []}
    c5_violated = False
    for arm in ARMS:
        dec = agg[(arm, 0.9, "decorrelate")]["CE_Q_frozen"]
        anti = agg[(arm, 0.9, "anticorrelate")]["CE_Q_frozen"]
        delta = max(dec["delta_cell"], anti["delta_cell"])
        diff = abs(dec["seed_mean"] - anti["seed_mean"])
        ok = diff <= delta
        c5["per_arm"].append({"arm": arm, "ce_q_dec": dec["seed_mean"],
                              "ce_q_anti": anti["seed_mean"], "abs_diff": diff,
                              "delta": delta, "consistent": bool(ok)})
        if not ok:
            c5_violated = True
    c5["VIOLATED"] = bool(c5_violated)
    gates["C-B5"] = c5

    # descriptive: Δ_transfer >> Δ_info at ρ>=0.9 (ratio >= 2), reported side by side
    descriptive_separation = []
    for rho in [0.9, 0.95, 0.99]:
        dt = agg[("naive", rho, GATED_SHIFT)]["delta_transfer"]["seed_mean"]
        di = agg[("naive", rho, GATED_SHIFT)]["delta_info"]["seed_mean"]
        ratio = (dt / di) if abs(di) > 1e-9 else float("inf")
        descriptive_separation.append({"rho": rho, "naive_delta_transfer": dt,
                                       "naive_delta_info": di, "ratio": ratio,
                                       "prediction": "ratio >= 2"})

    # =====================================================================
    # PQ2 — CV across the cell-mean grid {ρ}×{arm}×{CEhat_ID, CE_Q_dec, CEhat_Q_refit}
    # =====================================================================
    pq2_vals = []
    for arm, rho in product(ARMS, RHO_GRID):
        a = agg[(arm, rho, GATED_SHIFT)]
        pq2_vals += [a["CEhat_ID"]["seed_mean"], a["CE_Q_frozen"]["seed_mean"],
                     a["CEhat_Q_refit"]["seed_mean"]]
    pq2 = fourq.pq2_check(pq2_vals, threshold=0.20)

    # =====================================================================
    # serialize
    # =====================================================================
    fourq.check_quantity_declarations(rows)  # §5.7 — refuses undeclared tables
    fourq.write_results_csv(rows, os.path.join(outdir, "results.csv"))
    fourq.write_results_json(rows, os.path.join(outdir, "results_rows.json"),
                             meta={"experiment": "expB-masking-real-v1",
                                   "license_ref": lic, "proxy_validated": proxy_ok,
                                   "amendment_2026_06_11": {
                                       "ruling": "queue/decisions/answered/"
                                                 "runB-expB-s51-selftest-excess.md",
                                       "selftest_criterion": "CE*-DIFFERENCE recovery",
                                       "differences_licensed": proxy_ok,
                                       "absolutes": "raw_heldout_CE",
                                       "abs_ce_bias_bits_disclosed": ABS_CE_BIAS_BITS}})

    def jsonable_agg():
        out = {}
        for (arm, rho, shift), qd in agg.items():
            out[f"{arm}|rho{rho}|{shift}"] = {
                q: {kk: vv for kk, vv in d.items()} for q, d in qd.items()}
        return out

    summary = {
        "experiment": "expB-masking-real-v1", "schema": "expB-summary-v1",
        "pins": {"fourq_sha256": FOURQ_SHA256, "spurious_sha256": SPURIOUS_SHA256},
        "readout": {"class": "A_log multinomial-logistic", "C": READOUT_C,
                    "standardize": STANDARDIZE, "tol": READOUT_TOL,
                    "max_iter": READOUT_MAX_ITER, "k_refit_folds": K_REFIT_FOLDS,
                    "D": D, "K_readout": K_READOUT, "cnn_params": 1739210},
        "thresholds": {"materiality_floor": MATERIALITY_FLOOR,
                       "delta_formula": "max(2*SE_seed, 0.10)",
                       "underpowered_delta": UNDERPOWERED_DELTA,
                       "G_B2_min": G_B2_MIN, "tau_selftest": TAU_SELFTEST},
        "selftest_5_1": selftest_record,
        "gates": gates,
        "descriptive_transfer_vs_info": descriptive_separation,
        "pq": {"PQ2_cv": pq2,
               "PQ1": {"unit": "held-out CE bits of A_log (C=1.0, standardize=True) "
                              "on 256-d penultimate Z", "K_readout": K_READOUT,
                       "cnn_params": 1739210},
               "PQ4": "identical units/splits/readout/recipe across arms, rho, shifts",
               "PQ5": "no R-objective optimized; lambda_var in no criterion"},
        "aggregate": jsonable_agg(),
        "lawful_noise_floors": lawful_noise,
        "std_test_acc": sanity,
        "convergence_sample": convergence[:8],
        "n_cells": len(present),
        "H_Y_bits": H_Y_BITS, "log2_K_bits": H_Y_BITS,
    }
    with open(os.path.join(outdir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # machine summary (one-line PASS/FAIL per gate + verdict)
    def gv(g):
        v = g.get("PASS", None)
        return "UNDERPOWERED" if v is None else ("PASS" if v else "FAIL")
    verdict_all = all(gates[g].get("PASS") is True for g in ["G-B1", "G-B2", "G-B3", "G-B4"])
    machine = {
        "selftest_5_1_PASS": selftest_record["PASS"],
        "C_B5_violated": c5["VIOLATED"],
        "G-B1": gv(gates["G-B1"]), "G-B2": gv(gates["G-B2"]),
        "G-B3": gv(gates["G-B3"]), "G-B4": gv(gates["G-B4"]),
        "masking_at_scale_reproduced": bool(verdict_all and not c5["VIOLATED"]),
        "PQ2_cv": pq2["cv"],
    }
    with open(os.path.join(outdir, "machine_summary.json"), "w") as f:
        json.dump(machine, f, indent=2)

    # C-B5 HALT (after writing artifacts so the bug is inspectable)
    if c5["VIOLATED"]:
        sys.exit("C-B5 VIOLATED (rho=0.9 dec/anti disagreement > delta): pipeline "
                 "bug per §5.4 — analysis HALTS, no result stands until C-B5 holds. "
                 "See results_summary.json gates.C-B5.")

    print("\n=== machine summary ===")
    print(json.dumps(machine, indent=2))
    print(f"\nartifacts -> {outdir}: results.csv, results_rows.json, "
          f"results_summary.json, machine_summary.json")
    return summary


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true",
                   help="§5.1 planted-posterior validation ONLY (touches no trained features)")
    g.add_argument("--run", action="store_true",
                   help="the ONCE-ONLY full analysis (analysis agent only)")
    ap.add_argument("--feat-dir", default=FEAT_DIR_DEFAULT)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--selftest-seed", type=int, default=0)
    args = ap.parse_args()

    fsha, ssha = verify_pins()
    print(f"pins OK: fourq={fsha[:12]}.. spurious={ssha[:12]}..")

    if args.selftest:
        print(f"\n§5.1 CE*-DIFFERENCE recovery selftest (AMENDED 2026-06-11, §5.9; "
              f"tau={TAU_SELFTEST} bits, C={READOUT_C}, standardize={STANDARDIZE}, "
              f"n=5000/5000, D={D}, K={K}):")
        rec = run_selftest(args.selftest_seed)
        print(f"\n§5.1 selftest (difference-recovery): "
              f"{'PASS' if rec['PASS'] else 'FAIL'}  "
              f"(license_ref='{rec['license_ref']}')")
        sys.exit(0 if rec["PASS"] else 1)

    # --run: the once-only analysis. Re-run the §5.1 selftest first (license).
    print("\nONCE-ONLY ANALYSIS (--run). Establishing §5.1 difference-recovery "
          "license first.")
    rec = run_selftest(0, verbose=True)
    if not rec["PASS"]:
        print("WARNING: §5.1 difference-recovery selftest FAILED — proceeding with "
              "NO CE*-difference license; every CEhat AND every Δ serializes as raw "
              "(§5.1/§5.9; halt-and-escalate condition per the amendment).")
    run_full(args.feat_dir, args.outdir, rec)


if __name__ == "__main__":
    main()
