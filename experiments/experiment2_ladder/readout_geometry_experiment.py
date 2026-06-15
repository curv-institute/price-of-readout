#!/usr/bin/env python3
"""
Track 4 (Readout-Geometry Theorem) — empirical acceptance gates 2 and 3.
Experiment B: sparse superposed features, declared readout ladder.

Pre-registered protocol: see PREREGISTRATION.md in this directory.
All parameters below are fixed by the pre-registration. No parameter may be
changed after the first experimental run (single-amendment policy in the
pre-registration, proxy-quality reasons only).

Modes:
  --analytic   compute ground-truth quantities from the construction with NO
               sampling and NO trained readout: exact atom enumeration,
               injectivity check, and the exact population optimum of the
               linear-logistic readout class on the finite atom distribution
               (damped Newton on the exactly weighted atoms). Writes
               analytic_predictions.json. Run at pre-registration time.
  (default)    run the full pre-registered experiment grid; writes
               results.csv and results_summary.json.

numpy-only. Fixed seeds. Deterministic given the seed lists.
"""

import argparse
import csv
import itertools
import json
import os
import time

import numpy as np

# ----------------------------------------------------------------------------
# Pre-registered fixed parameters (see PREREGISTRATION.md section 3)
# ----------------------------------------------------------------------------
D_FEAT = 16                 # number of binary features
K_ACT = 4                   # exactly-k active features per pattern
DZ_GRID = [4, 6, 8, 12]     # representation dimensions (superposition load axis)
SEEDS = list(range(10))     # seeds 0..9
N_TRAIN = 20_000
N_TEST = 100_000
IRLS_ITERS = 25             # rung (i) fixed Newton/IRLS budget (track-3 protocol)
RIDGE = 1e-8                # rung (i) numerical damping
ANALYTIC_NEWTON_ITERS = 300 # analytic population-floor Newton budget
ANALYTIC_RIDGE = 1e-10
M_HID = 64                  # rung (ii) hidden width
N_INITS = 3                 # rung (ii) restarts; selection by final TRAIN loss only
BATCH = 2_000               # rung (ii) minibatch size
EPOCHS = 1_000              # rung (ii) epochs (10 steps/epoch)
ADAM_LR = 3e-3
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
EPS_CLIP = 1e-6             # rung (iii) predictive-probability clip
LOG2 = np.log(2.0)

OUTDIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Construction (PREREGISTRATION.md section 2)
# ----------------------------------------------------------------------------
def build_atoms():
    """All C(D_FEAT, K_ACT) exactly-k feature patterns, uniform; Y = F_1."""
    supports = list(itertools.combinations(range(D_FEAT), K_ACT))
    C = len(supports)
    F = np.zeros((C, D_FEAT))
    for i, s in enumerate(supports):
        F[i, list(s)] = 1.0
    y = np.array([1.0 if 0 in s else 0.0 for s in supports])
    return F, y


def projection(seed, d_z):
    """Fixed random projection for (seed, d_z): N(0,1) entries, unit-norm columns."""
    rng = np.random.default_rng(20_000 + seed)
    A = rng.standard_normal((d_z, D_FEAT))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A


def min_pairwise_dist(Zat, chunk=512):
    """Minimum pairwise Euclidean distance among atoms (injectivity check)."""
    n = Zat.shape[0]
    best = np.inf
    sq = (Zat ** 2).sum(axis=1)
    for i0 in range(0, n, chunk):
        blk = Zat[i0:i0 + chunk]
        d2 = sq[i0:i0 + chunk][:, None] + sq[None, :] - 2.0 * blk @ Zat.T
        for r in range(blk.shape[0]):
            d2[r, i0 + r] = np.inf
        best = min(best, float(np.sqrt(max(d2.min(), 0.0))))
    return best


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ----------------------------------------------------------------------------
# Analytic population floor of the linear-logistic class (exact atom weights)
# ----------------------------------------------------------------------------
def linear_population_floor(Zat, yat, wts, iters=ANALYTIC_NEWTON_ITERS,
                            ridge=ANALYTIC_RIDGE):
    """Population cross-entropy optimum (bits) of q(y|z)=sigma(w.z+b) on the
    exact finite atom distribution. Damped Newton with step-halving."""
    n, d = Zat.shape
    Xb = np.hstack([Zat, np.ones((n, 1))])
    D = d + 1
    beta = np.zeros(D)

    def loss(b):
        z = Xb @ b
        return float(np.sum(wts * (np.logaddexp(0.0, -z) * yat
                                   + np.logaddexp(0.0, z) * (1.0 - yat))))

    L = loss(beta)
    for _ in range(iters):
        z = Xb @ beta
        p = sigmoid(z)
        g = Xb.T @ (wts * (p - yat)) + ridge * beta
        W = wts * p * (1.0 - p)
        H = (Xb * W[:, None]).T @ Xb + ridge * np.eye(D)
        step = np.linalg.solve(H, g)
        t = 1.0
        for _h in range(50):
            cand = beta - t * step
            cl = loss(cand)
            if cl <= L + 1e-15:
                beta = cand
                L = cl
                break
            t *= 0.5
    return L / LOG2, beta


# ----------------------------------------------------------------------------
# Rung (i): linear logistic by damped Newton/IRLS (track-3 trainer + FLOPs)
# ----------------------------------------------------------------------------
def _mean_logloss_nats(Xb, y, beta):
    z = Xb @ beta
    return float(np.mean(np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1 - y)))


def fit_logistic_irls(X, y, iters=IRLS_ITERS, ridge=RIDGE):
    n, d = X.shape
    D = d + 1
    Xb = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(D)
    flops = 0
    halvings_total = 0
    loss = _mean_logloss_nats(Xb, y, beta)
    for _ in range(iters):
        z = Xb @ beta
        p = sigmoid(z)
        g = Xb.T @ (p - y) / n + ridge * beta
        w = p * (1 - p)
        H = (Xb * w[:, None]).T @ Xb / n + ridge * np.eye(D)
        step = np.linalg.solve(H, g)
        flops += 2 * n * D + 5 * n + 2 * n * D + n + 2 * D + 2 * n \
            + n * D + 2 * n * D**2 + D**2 + (2 * D**3) // 3 + 2 * D**2
        t = 1.0
        for _h in range(30):
            cand = beta - t * step
            cand_loss = _mean_logloss_nats(Xb, y, cand) + 0.5 * ridge * float(cand @ cand)
            flops += 2 * n * D + 6 * n + 2 * D
            if cand_loss <= loss + 1e-15:
                beta = cand
                loss = cand_loss
                break
            t *= 0.5
            halvings_total += 1
    z = Xb @ beta
    p = sigmoid(z)
    g = Xb.T @ (p - y) / n + ridge * beta
    return beta, {"train_flops": flops, "halvings": halvings_total,
                  "grad_inf_norm": float(np.abs(g).max())}


def eval_logistic(X, y, beta):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return _mean_logloss_nats(Xb, y, beta) / LOG2


# ----------------------------------------------------------------------------
# Rung (ii): two-layer ReLU net trained by minibatch Adam (numpy)
# ----------------------------------------------------------------------------
def fit_two_layer(Xtr, ytr, seed, d_z, init_idx):
    rng = np.random.default_rng(40_000 + 1_000 * seed + 10 * d_z + init_idx)
    n = Xtr.shape[0]
    W1 = rng.standard_normal((d_z, M_HID)) * np.sqrt(2.0 / d_z)
    b1 = np.zeros(M_HID)
    W2 = rng.standard_normal(M_HID) / np.sqrt(M_HID)
    b2 = float(np.log(K_ACT / (D_FEAT - K_ACT)))  # logit of the prior P(Y=1)
    params = [W1, b1, W2, np.array([b2])]
    mom = [np.zeros_like(p) for p in params]
    vel = [np.zeros_like(p) for p in params]
    t = 0
    steps_per_epoch = n // BATCH
    for ep in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            X, y = Xtr[idx], ytr[idx]
            Hpre = X @ params[0] + params[1]
            A = np.maximum(Hpre, 0.0)
            z = A @ params[2] + params[3][0]
            p = sigmoid(z)
            dz = (p - y) / BATCH
            g = [None] * 4
            g[2] = A.T @ dz
            g[3] = np.array([dz.sum()])
            dA = np.outer(dz, params[2])
            dH = dA * (Hpre > 0)
            g[0] = X.T @ dH
            g[1] = dH.sum(axis=0)
            t += 1
            for j in range(4):
                mom[j] = ADAM_B1 * mom[j] + (1 - ADAM_B1) * g[j]
                vel[j] = ADAM_B2 * vel[j] + (1 - ADAM_B2) * g[j] ** 2
                mhat = mom[j] / (1 - ADAM_B1 ** t)
                vhat = vel[j] / (1 - ADAM_B2 ** t)
                params[j] -= ADAM_LR * mhat / (np.sqrt(vhat) + ADAM_EPS)
    return params


def eval_two_layer(X, y, params):
    Hpre = X @ params[0] + params[1]
    A = np.maximum(Hpre, 0.0)
    z = A @ params[2] + params[3][0]
    ll = np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1 - y)
    return float(ll.mean()) / LOG2


# ----------------------------------------------------------------------------
# Rung (iii): oracle nearest-atom decoder (codebook = constructed atoms)
# ----------------------------------------------------------------------------
def eval_nearest_atom(Ztest, ytest, Zat, yat, chunk=2_000):
    n = Ztest.shape[0]
    sq_at = (Zat ** 2).sum(axis=1)
    correct = 0
    ll_sum = 0.0
    for i0 in range(0, n, chunk):
        blk = Ztest[i0:i0 + chunk]
        d2 = (blk ** 2).sum(axis=1)[:, None] + sq_at[None, :] - 2.0 * blk @ Zat.T
        pred = yat[np.argmin(d2, axis=1)]
        yb = ytest[i0:i0 + chunk]
        hit = pred == yb
        correct += int(hit.sum())
        # predictive prob 1-EPS_CLIP on predicted label
        ll_sum += float(np.where(hit, -np.log(1.0 - EPS_CLIP),
                                 -np.log(EPS_CLIP)).sum())
    return ll_sum / n / LOG2, correct / n


# ----------------------------------------------------------------------------
# Declared K / E accounting (PREREGISTRATION.md section 4)
# ----------------------------------------------------------------------------
def accounting(d_z, C):
    D = d_z + 1
    K_i = D
    K_ii = M_HID * D + (M_HID + 1)
    K_iii = C * D
    E_i = 2 * D + 5
    E_ii = 2 * M_HID * D + M_HID + 2 * (M_HID + 1) + 5
    E_iii = 3 * C * d_z + C + 1
    steps = EPOCHS * (N_TRAIN // BATCH)
    E_train_ii = N_INITS * (steps * BATCH * 3 * E_ii + steps * 10 * K_ii)
    return {"K_i": K_i, "K_ii": K_ii, "K_iii": K_iii,
            "E_inf_i": E_i, "E_inf_ii": E_ii, "E_inf_iii": E_iii,
            "E_train_ii": E_train_ii}


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def run_analytic():
    F, yat = build_atoms()
    C = F.shape[0]
    prior = float(yat.mean())
    out = {"_construction": {"d_feat": D_FEAT, "k_act": K_ACT, "n_atoms": C,
                             "prior_P(Y=1)": prior,
                             "H(Y)_bits": float(-prior * np.log2(prior)
                                                - (1 - prior) * np.log2(1 - prior))},
           "cells": {}}
    wts = np.full(C, 1.0 / C)
    for d_z in DZ_GRID:
        for seed in SEEDS:
            A = projection(seed, d_z)
            Zat = F @ A.T
            md = min_pairwise_dist(Zat)
            floor_bits, _ = linear_population_floor(Zat, yat, wts)
            out["cells"][f"dz{d_z}_seed{seed}"] = {
                "min_atom_dist": md,
                "H(Y|Z)_bits": 0.0 if md > 1e-9 else None,
                "CE_lin_star_bits": floor_bits,
            }
        vals = [out["cells"][f"dz{d_z}_seed{s}"]["CE_lin_star_bits"] for s in SEEDS]
        out[f"dz{d_z}_CE_lin_star_mean"] = float(np.mean(vals))
        out[f"dz{d_z}_CE_lin_star_min"] = float(np.min(vals))
        out[f"dz{d_z}_CE_lin_star_max"] = float(np.max(vals))
        out[f"dz{d_z}_accounting"] = accounting(d_z, C)
    with open(os.path.join(OUTDIR, "analytic_predictions.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))


def run_experiment():
    F, yat = build_atoms()
    C = F.shape[0]
    with open(os.path.join(OUTDIR, "analytic_predictions.json")) as f:
        analytic = json.load(f)
    rows = []
    t_total = time.time()
    for d_z in DZ_GRID:
        acc = accounting(d_z, C)
        for seed in SEEDS:
            A = projection(seed, d_z)
            Zat = F @ A.T
            rng = np.random.default_rng(30_000 + 1_000 * seed + d_z)
            idx_tr = rng.integers(0, C, N_TRAIN)
            idx_te = rng.integers(0, C, N_TEST)
            Xtr, ytr = Zat[idx_tr], yat[idx_tr]
            Xte, yte = Zat[idx_te], yat[idx_te]
            ana = analytic["cells"][f"dz{d_z}_seed{seed}"]

            # rung (i)
            t0 = time.time()
            beta, diag = fit_logistic_irls(Xtr, ytr)
            wt_i = time.time() - t0
            t0 = time.time()
            ce_i = eval_logistic(Xte, yte, beta)
            wi_i = time.time() - t0
            rows.append(dict(seed=seed, d_z=d_z, rung="i_linear",
                             K=acc["K_i"], E_inf=acc["E_inf_i"],
                             E_train=diag["train_flops"], ce_bits=ce_i,
                             wall_train=wt_i, wall_infer=wi_i,
                             diag_grad=diag["grad_inf_norm"],
                             diag_halvings=diag["halvings"],
                             analytic_floor=ana["CE_lin_star_bits"],
                             min_atom_dist=ana["min_atom_dist"]))

            # rung (ii): N_INITS restarts, select by final TRAIN loss only
            t0 = time.time()
            best = None
            for init_idx in range(N_INITS):
                params = fit_two_layer(Xtr, ytr, seed, d_z, init_idx)
                tr_loss = eval_two_layer(Xtr, ytr, params)
                if best is None or tr_loss < best[0]:
                    best = (tr_loss, params, init_idx)
            wt_ii = time.time() - t0
            t0 = time.time()
            ce_ii = eval_two_layer(Xte, yte, best[1])
            wi_ii = time.time() - t0
            rows.append(dict(seed=seed, d_z=d_z, rung="ii_twolayer",
                             K=acc["K_ii"], E_inf=acc["E_inf_ii"],
                             E_train=acc["E_train_ii"], ce_bits=ce_ii,
                             wall_train=wt_ii, wall_infer=wi_ii,
                             diag_grad=best[0], diag_halvings=best[2],
                             analytic_floor=ana["CE_lin_star_bits"],
                             min_atom_dist=ana["min_atom_dist"]))

            # rung (iii): oracle nearest-atom decoder (no training)
            t0 = time.time()
            ce_iii, acc_rate = eval_nearest_atom(Xte, yte, Zat, yat)
            wi_iii = time.time() - t0
            rows.append(dict(seed=seed, d_z=d_z, rung="iii_oracle",
                             K=acc["K_iii"], E_inf=acc["E_inf_iii"],
                             E_train=0, ce_bits=ce_iii,
                             wall_train=0.0, wall_infer=wi_iii,
                             diag_grad=acc_rate, diag_halvings=0,
                             analytic_floor=ana["CE_lin_star_bits"],
                             min_atom_dist=ana["min_atom_dist"]))
            print(f"d_z={d_z} seed={seed}: i={ce_i:.4f} ii={ce_ii:.4f} "
                  f"iii={ce_iii:.6f} floor*={ana['CE_lin_star_bits']:.4f}",
                  flush=True)

    fields = list(rows[0].keys())
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # summary: cell means/stds + PQ2 CVs over the grid of cell means
    summary = {"_runtime_s": time.time() - t_total}
    cell_means = {"H": [], "K": [], "E_inf": [], "E_train": [], "wall": []}
    for d_z in DZ_GRID:
        for rung in ["i_linear", "ii_twolayer", "iii_oracle"]:
            sel = [r for r in rows if r["d_z"] == d_z and r["rung"] == rung]
            ce = np.array([r["ce_bits"] for r in sel])
            summary[f"dz{d_z}_{rung}"] = {
                "ce_mean": float(ce.mean()), "ce_std": float(ce.std()),
                "ce_se": float(ce.std(ddof=1) / np.sqrt(len(ce))),
                "K": sel[0]["K"], "E_inf": sel[0]["E_inf"],
                "E_train_mean": float(np.mean([r["E_train"] for r in sel])),
                "wall_train_mean": float(np.mean([r["wall_train"] for r in sel])),
                "wall_infer_mean": float(np.mean([r["wall_infer"] for r in sel])),
            }
            cell_means["H"].append(float(ce.mean()))
            cell_means["K"].append(sel[0]["K"])
            cell_means["E_inf"].append(sel[0]["E_inf"])
            cell_means["E_train"].append(float(np.mean([r["E_train"] for r in sel])))
            cell_means["wall"].append(
                float(np.mean([r["wall_train"] + r["wall_infer"] for r in sel])))
        floors = [r["analytic_floor"] for r in rows
                  if r["d_z"] == d_z and r["rung"] == "i_linear"]
        summary[f"dz{d_z}_analytic_floor_mean"] = float(np.mean(floors))
    summary["_PQ2_CV"] = {
        k: float(np.std(v) / np.mean(v)) for k, v in cell_means.items()}
    with open(os.path.join(OUTDIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytic", action="store_true")
    args = ap.parse_args()
    if args.analytic:
        run_analytic()
    else:
        run_experiment()
