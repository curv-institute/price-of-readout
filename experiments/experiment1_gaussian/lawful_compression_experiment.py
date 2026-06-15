#!/usr/bin/env python3
"""
Track 3 (Lawful-Compression Theorem) — empirical acceptance gates 2 and 3.

Pre-registered protocol: see PREREGISTRATION.md in this directory.
All parameters below are fixed by the pre-registration. No parameter may be
changed after the first experimental run (single-amendment policy in the
pre-registration, proxy-quality reasons only).

Modes:
  --analytic   compute ground-truth quantities from the generative model by
               Gauss-Hermite quadrature (no data is drawn, no readout trained);
               writes analytic_predictions.json. Run at pre-registration time.
  (default)    run the full pre-registered experiment grid; writes
               results.csv and results_summary.json.

numpy-only. Fixed seeds. Deterministic given the seed list.
"""

import argparse
import csv
import json
import os
import platform
import time

import numpy as np

# ----------------------------------------------------------------------------
# Pre-registered fixed parameters (see PREREGISTRATION.md §3)
# ----------------------------------------------------------------------------
D_S = 4                                  # signal dimension
W = np.array([1.0, 1.0, 1.0, 1.0])       # task weights; ||W|| = 2
WNORM = 2.0                              # ||W||
RHO_TRAIN = 0.9                          # in-distribution S-N coupling
RHO_SHIFT = -0.9                         # shifted S-N coupling (sign flip)
DN_GRID = [0, 8, 64, 256]                # nuisance dimensions (K/E axis)
SEEDS = list(range(10))                  # data seeds 0..9
N_TRAIN = 50_000
N_TEST = 100_000                         # per evaluation condition (ID, shift)
IRLS_ITERS = 25                          # fixed Newton/IRLS iteration budget
RIDGE = 1e-8                             # numerical damping (stated in prereg)
GH_NODES = 64                            # Gauss-Hermite nodes (probabilists')
LOG2 = np.log(2.0)
EPS = 1e-12

OUTDIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Generative model (constructed sufficiency; PREREGISTRATION.md §2)
# ----------------------------------------------------------------------------
def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def generate(rng, n, d_n, rho):
    """X = (S, N); Y | S ~ Bernoulli(sigma(W.S)); N_j = rho*u + sqrt(1-rho^2)*eps_j
    with u = W.S/||W||.  Y depends on X only through S (sufficiency of S is a
    theorem of this construction)."""
    S = rng.standard_normal((n, D_S))
    u = S @ W / WNORM
    if d_n > 0:
        N = rho * u[:, None] + np.sqrt(1.0 - rho**2) * rng.standard_normal((n, d_n))
    else:
        N = np.zeros((n, 0))
    p = sigmoid(S @ W)
    Y = (rng.random(n) < p).astype(float)
    return S, N, Y


# ----------------------------------------------------------------------------
# Readout: logistic regression fit by damped Newton (IRLS), numpy only
# ----------------------------------------------------------------------------
def _mean_logloss_nats(Xb, y, beta):
    z = Xb @ beta
    # stable log(1+exp(-|z|)) form
    return float(np.mean(np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1 - y)))


def fit_logistic_irls(X, y, iters=IRLS_ITERS, ridge=RIDGE):
    """Damped Newton on mean logistic loss + (ridge/2)||beta||^2.
    Returns beta, diagnostics dict (incl. analytic FLOP count per the
    pre-registered convention) ."""
    n, d = X.shape
    D = d + 1
    Xb = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(D)
    flops = 0
    halvings_total = 0
    loss = _mean_logloss_nats(Xb, y, beta)
    for _ in range(iters):
        z = Xb @ beta                       # 2nD
        p = sigmoid(z)                      # ~5n
        g = Xb.T @ (p - y) / n + ridge * beta          # 2nD + n + 2D
        w = p * (1 - p)                                # 2n
        H = (Xb * w[:, None]).T @ Xb / n + ridge * np.eye(D)  # nD + 2nD^2 + D^2
        step = np.linalg.solve(H, g)        # (2/3)D^3 + 2D^2
        flops += 2 * n * D + 5 * n + 2 * n * D + n + 2 * D + 2 * n \
            + n * D + 2 * n * D**2 + D**2 + (2 * D**3) // 3 + 2 * D**2
        # deterministic step-halving line search on the damped Newton step
        t = 1.0
        for _h in range(30):
            cand = beta - t * step
            cand_loss = _mean_logloss_nats(Xb, y, cand) + 0.5 * ridge * float(cand @ cand)
            flops += 2 * n * D + 6 * n + 2 * D       # one extra loss eval
            if cand_loss <= loss + 1e-15:
                beta = cand
                loss = cand_loss
                break
            t *= 0.5
            halvings_total += 1
    # final gradient inf-norm (convergence diagnostic, recorded)
    z = Xb @ beta
    p = sigmoid(z)
    g = Xb.T @ (p - y) / n + ridge * beta
    return beta, {
        "flops_train": int(flops),
        "grad_inf_norm": float(np.max(np.abs(g))),
        "halvings": int(halvings_total),
        "final_loss_nats": float(loss),
    }


def eval_ce_bits(X, y, beta):
    """Held-out cross-entropy of the trained readout, in bits/sample.
    Inference FLOPs per pre-registered convention: 2*n*D + 5*n."""
    n, d = X.shape
    D = d + 1
    Xb = np.hstack([X, np.ones((n, 1))])
    z = Xb @ beta
    ce_nats = float(np.mean(np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1 - y)))
    return ce_nats / LOG2, int(2 * n * D + 5 * n)


# ----------------------------------------------------------------------------
# Analytic ground truth (Gauss-Hermite quadrature on the generative model)
# ----------------------------------------------------------------------------
def gh_nodes(n=GH_NODES):
    x, w = np.polynomial.hermite_e.hermegauss(n)  # probabilists': E over N(0,1)
    return x, w / w.sum()


def h2_bits(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / LOG2


def ce_bits(p_true, q_pred):
    q = np.clip(q_pred, EPS, 1 - EPS)
    return -(p_true * np.log(q) + (1 - p_true) * np.log(1 - q)) / LOG2


def v_mu(d_n, rho=RHO_TRAIN):
    """Var of E[u|N] under the construction (u, N jointly Gaussian)."""
    r2 = rho**2
    return r2 / (r2 + (1 - r2) / d_n)


def analytic_quantities():
    x, wq = gh_nodes()
    # H(Y|S) = H(Y|X) = E_{u~N(0,1)} h2(sigma(||W|| u))   [bits]
    H0 = float(np.sum(wq * h2_bits(sigmoid(WNORM * x))))

    out = {
        "H_Y_given_S_bits": H0,
        "per_dn": {},
        "conventions": {
            "K": "K = (d+1) + d = 2d+1 (readout params incl. bias + input dim)",
            "flops_train": "per IRLS iter: 2nD^2 + 7nD + 14n + (2/3)D^3 + 3D^2 + 4D "
                           "(D = d+1) + one loss eval (2nD+6n+2D) per halving step",
            "flops_infer": "2*n_test*D + 5*n_test",
        },
    }
    for d_n in DN_GRID:
        if d_n == 0:
            continue
        v = v_mu(d_n)
        sd_m = np.sqrt(v)
        m = sd_m * x                      # outer nodes: m = E[u|N] ~ N(0, v)
        # inner smoothing: p(Y=1|N) = E[sigma(2(m + sqrt(1-v) zeta))]
        z_in, w_in = gh_nodes()
        G = sigmoid(2.0 * m[:, None] + 2.0 * np.sqrt(1 - v) * z_in[None, :]) @ w_in
        H_Y_given_N = float(np.sum(wq * h2_bits(G)))

        # readout-constrained value: best logistic-in-m predictor sigma(c*m)
        def risk(c):
            return float(np.sum(wq * ce_bits(G, sigmoid(c * m))))
        lo, hi = 1e-3, 50.0
        for _ in range(200):              # ternary search (risk convex in c)
            c1, c2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if risk(c1) < risk(c2):
                hi = c2
            else:
                lo = c1
        c_star = 0.5 * (lo + hi)
        H_A = risk(c_star)

        # post-shift CE of the in-distribution-optimal logistic readout:
        # under rho -> -rho the true p(Y=1|N=n) becomes g(-m) = 1 - g(m),
        # while the trained readout still predicts sigma(c* m).
        CE_shift = float(np.sum(wq * ce_bits(1.0 - G, sigmoid(c_star * m))))

        out["per_dn"][str(d_n)] = {
            "v_mu": float(v),
            "H_Y_given_N_bits": H_Y_given_N,
            "H_A_logistic_Y_given_N_bits": float(H_A),
            "c_star": float(c_star),
            "CE_shift_unlawful_bits": CE_shift,
            "predicted_shift_increase_bits": float(CE_shift - H_A),
        }
    return out


# ----------------------------------------------------------------------------
# Experiment grid
# ----------------------------------------------------------------------------
def run_experiment():
    rows = []
    for seed in SEEDS:
        for d_n in DN_GRID:
            rng = np.random.default_rng(10_000 + seed)
            S_tr, N_tr, y_tr = generate(rng, N_TRAIN, d_n, RHO_TRAIN)
            S_id, N_id, y_id = generate(rng, N_TEST, d_n, RHO_TRAIN)
            S_sh, N_sh, y_sh = generate(rng, N_TEST, d_n, RHO_SHIFT)

            arms = {"X": (np.hstack([S_tr, N_tr]), np.hstack([S_id, N_id]),
                          np.hstack([S_sh, N_sh])),
                    "S": (S_tr, S_id, S_sh)}
            if d_n > 0:
                arms["N"] = (N_tr, N_id, N_sh)

            for arm, (Xtr, Xid, Xsh) in arms.items():
                d = Xtr.shape[1]
                t0 = time.perf_counter()
                beta, diag = fit_logistic_irls(Xtr, y_tr)
                t_train = time.perf_counter() - t0

                t0 = time.perf_counter()
                h_id, fl_inf_id = eval_ce_bits(Xid, y_id, beta)
                h_sh, fl_inf_sh = eval_ce_bits(Xsh, y_sh, beta)
                t_infer = time.perf_counter() - t0

                rows.append({
                    "seed": seed, "d_n": d_n, "arm": arm, "dim": d,
                    "K": 2 * d + 1,
                    "H_hat_id_bits": h_id, "H_hat_shift_bits": h_sh,
                    "flops_train": diag["flops_train"],
                    "flops_infer": fl_inf_id + fl_inf_sh,
                    "wall_train_s": t_train, "wall_infer_s": t_infer,
                    "grad_inf_norm": diag["grad_inf_norm"],
                    "halvings": diag["halvings"],
                })
                print(f"seed={seed} d_n={d_n:3d} arm={arm} dim={d:3d} "
                      f"H_id={h_id:.4f} H_shift={h_sh:.4f} "
                      f"grad={diag['grad_inf_norm']:.2e} t={t_train:.2f}s",
                      flush=True)
    return rows


def summarize(rows):
    """Seed-mean/std per (d_n, arm) cell + PQ2 CV computations."""
    cells = {}
    for r in rows:
        cells.setdefault((r["d_n"], r["arm"]), []).append(r)
    summary = {}
    for (d_n, arm), rs in sorted(cells.items()):
        def stat(k):
            v = np.array([r[k] for r in rs], dtype=float)
            return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
        summary[f"dn{d_n}_{arm}"] = {
            "d_n": d_n, "arm": arm, "n_seeds": len(rs),
            "K": rs[0]["K"], "dim": rs[0]["dim"],
            "H_hat_id_bits": stat("H_hat_id_bits"),
            "H_hat_shift_bits": stat("H_hat_shift_bits"),
            "flops_train": stat("flops_train"),
            "flops_infer": stat("flops_infer"),
            "wall_train_s": stat("wall_train_s"),
            "wall_infer_s": stat("wall_infer_s"),
            "grad_inf_norm_max": float(max(r["grad_inf_norm"] for r in rs)),
        }
    # PQ2: CV across the grid of cell means (H pooled over ID+shift cells;
    # E and K over arm/d_n cells), per PREREGISTRATION.md §6.
    h_cells = []
    for c in summary.values():
        h_cells.append(c["H_hat_id_bits"]["mean"])
        h_cells.append(c["H_hat_shift_bits"]["mean"])
    e_flops = [c["flops_train"]["mean"] + c["flops_infer"]["mean"]
               for c in summary.values()]
    e_wall = [c["wall_train_s"]["mean"] + c["wall_infer_s"]["mean"]
              for c in summary.values()]
    k_cells = [float(c["K"]) for c in summary.values()]

    def cv(v):
        v = np.array(v, dtype=float)
        return float(v.std(ddof=1) / v.mean())

    summary["_PQ2_CV"] = {
        "H_cell_means_cv": cv(h_cells),
        "E_flops_cell_means_cv": cv(e_flops),
        "E_wallclock_cell_means_cv": cv(e_wall),
        "K_cell_means_cv": cv(k_cells),
    }
    summary["_env"] = {
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "n_train": N_TRAIN, "n_test": N_TEST,
        "irls_iters": IRLS_ITERS, "ridge": RIDGE,
        "seeds": SEEDS, "dn_grid": DN_GRID,
        "rho_train": RHO_TRAIN, "rho_shift": RHO_SHIFT,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytic", action="store_true",
                    help="compute analytic predictions only (pre-registration step)")
    args = ap.parse_args()

    if args.analytic:
        out = analytic_quantities()
        path = os.path.join(OUTDIR, "analytic_predictions.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        print(f"\nwritten: {path}")
        return

    t0 = time.perf_counter()
    rows = run_experiment()
    total = time.perf_counter() - t0

    csv_path = os.path.join(OUTDIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    summary = summarize(rows)
    summary["_total_wall_s"] = total
    with open(os.path.join(OUTDIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {total:.1f}s. Wrote results.csv and results_summary.json")


if __name__ == "__main__":
    main()
