#!/usr/bin/env python3
"""Generate the data-driven figures for the lawful-compression/readout-geometry paper.

Reads ONLY the committed experiment artifacts (results.csv of each
campaign), by relative path within the repository; no measured number is
transcribed by hand.  Analytic reference values are recomputed from the
paper's exact formulas where needed (the 1-bit binary entropy bound); the
Experiment 2 per-seed exact floors are read from the committed
``analytic_floor`` column of the results file itself.

Outputs (vector PDF, colorblind-safe Okabe--Ito palette):
  fig_exp1_shift.pdf    -- Experiment 1 (lawful compression) masking/shift headline result
  fig_exp2_ladder.pdf   -- Experiment 2 (readout ladder): measured CE vs. load, floors overlaid
  fig_exp3_masking.pdf  -- Experiment 3 (masking at scale): the masking curve over rho
  fig_exp3_stimuli.pdf  -- Experiment 3 stimulus strip (pinned construction module + CIFAR-10)
  fig_exp3_cannibal.pdf -- Experiment 3 cue-cannibalization panel (accuracy + delta_info vs rho)

Usage: python3 make_figures.py   (from any directory)
"""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
# paper/figures -> repository root
PROGRAM_ROOT = HERE.parent.parent
T3_CSV = PROGRAM_ROOT / "experiments" / "experiment1_gaussian" / "results.csv"
T4_CSV = PROGRAM_ROOT / "experiments" / "experiment2_ladder" / "results.csv"
EXPB_CSV = PROGRAM_ROOT / "experiments" / "experiment3_cifar" / "results.csv"
EXPB_SUMMARY = PROGRAM_ROOT / "experiments" / "experiment3_cifar" / "results_summary.json"
# The pinned Experiment 3 construction module (Appendix: details-e3); its
# SHA-256 is asserted before import so the stimulus figure can only ever be
# produced by the exact construction code that produced the experiment.
SPURIOUS_MODULE = PROGRAM_ROOT / "experiments" / "experiment3_cifar" / "spurious_cifar.py"
SPURIOUS_SHA256 = "ab8f6e61d54be4ea13b23aee326aabccb2f272c4bfffa4431ace45b70c9050ba"

# Okabe--Ito colorblind-safe palette
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"

# Two-column layout (v12): figures are generated at single-column width
# (3.25 in, the \columnwidth of the 10pt two-column article with 1 in
# margins and 0.25 in column separation) so fonts render at true size.
FIG_W = 3.25

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 6.4,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
    }
)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------
# Figure D: Experiment 1 (lawful compression) masking/shift result
# ----------------------------------------------------------------------
def fig_track3():
    rows = read_csv(T3_CSV)
    # index: (d_n, arm, seed) -> (ce_id, ce_shift)
    ce = {}
    for r in rows:
        key = (int(r["d_n"]), r["arm"], int(r["seed"]))
        ce[key] = (float(r["H_hat_id_bits"]), float(r["H_hat_shift_bits"]))
    seeds = sorted({int(r["seed"]) for r in rows})
    dns = [8, 64, 256]  # cells where the S vs N comparison exists

    # paired per-seed quantities (S and N arms share seed streams)
    id_excess, elevation, lawful_drift = {}, {}, {}
    for dn in dns:
        id_excess[dn] = np.array(
            [ce[(dn, "N", s)][0] - ce[(dn, "S", s)][0] for s in seeds]
        )
        elevation[dn] = np.array(
            [ce[(dn, "N", s)][1] - ce[(dn, "N", s)][0] for s in seeds]
        )
        lawful_drift[dn] = np.array(
            [ce[(dn, "S", s)][1] - ce[(dn, "S", s)][0] for s in seeds]
        )

    series = [
        (id_excess, "insufficient arm $N$: ID excess over lawful arm", SKY),
        (
            elevation,
            "insufficient arm $N$: post-shift elevation of\n"
            r"$\mathrm{CE}_Q(r_P)$ over its ID value",
            VERMILLION,
        ),
        (lawful_drift, "lawful arm $S$: post-shift drift", GREEN),
    ]

    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    x = np.arange(len(dns))
    width = 0.26
    rng = np.random.default_rng(0)  # jitter only (presentation)
    for i, (data, label, color) in enumerate(series):
        pos = x + (i - 1) * width
        means = [data[dn].mean() for dn in dns]
        sds = [data[dn].std(ddof=1) for dn in dns]
        ax.bar(
            pos,
            means,
            width,
            yerr=sds,
            capsize=2.5,
            label=label,
            color=color,
            error_kw={"lw": 0.9},
            zorder=3,
        )
        # per-seed scatter
        for j, dn in enumerate(dns):
            jit = (rng.random(len(data[dn])) - 0.5) * width * 0.6
            ax.plot(
                pos[j] + jit,
                data[dn],
                ".",
                color="black",
                ms=2.5,
                alpha=0.55,
                zorder=4,
            )
        # numeric annotations (4 decimals: the tiny bars are the point);
        # white backing + top z-order so neighboring bars cannot occlude
        # them at column width
        for j, dn in enumerate(dns):
            m = means[j]
            ax.annotate(
                f"{m:+.4f}",
                (pos[j], max(m, 0) + sds[j] + 0.045),
                ha="center",
                fontsize="x-small" if abs(m) < 0.1 else "small",
                color="black",
                zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=0.6),
            )

    # The 1-bit bound is a legend entry, not an in-axes annotation: the
    # annotation collided with the tall center bars (v7 review, item 6a).
    ax.axhline(
        1.0,
        color="black",
        ls="--",
        lw=1.0,
        zorder=2,
        label="1-bit binary entropy bound (no entropy\n"
        "difference of this task can exceed it)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"$d_n = {dn}$" for dn in dns])
    ax.set_ylabel("bits (seed-mean over 10 seeds)")
    ax.set_ylim(-0.12, 2.05)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        frameon=False,
        borderaxespad=0,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(HERE / "fig_exp1_shift.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure E: Experiment 2 (readout) ladder, measured CE vs. superposition load,
# committed exact floors overlaid
# ----------------------------------------------------------------------
def fig_track4():
    rows = read_csv(T4_CSV)
    d_feat = 16  # construction constant (d_feat = 16, k = 4)
    dzs = sorted({int(r["d_z"]) for r in rows})
    loads = {dz: d_feat / dz for dz in dzs}

    def collect(rung, col):
        out = {}
        for dz in dzs:
            out[dz] = np.array(
                [
                    float(r[col])
                    for r in rows
                    if int(r["d_z"]) == dz and r["rung"] == rung
                ]
            )
        return out

    meas = {rung: collect(rung, "ce_bits") for rung in
            ["i_linear", "ii_twolayer", "iii_oracle"]}
    floors = collect("i_linear", "analytic_floor")  # per-seed exact floors

    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    xs = np.array([loads[dz] for dz in dzs])
    order = np.argsort(xs)
    xs_o = xs[order]
    dz_o = [dzs[i] for i in order]

    # committed exact floor (prediction)
    floor_means = np.array([floors[dz].mean() for dz in dz_o])
    ax.plot(
        xs_o,
        floor_means,
        "--",
        color="black",
        lw=1.2,
        marker="s",
        mfc="white",
        ms=6,
        zorder=3,
        label=r"exact floor $\mathrm{CE}^{\star}_{\mathrm{lin}}$"
        " (analytic prediction)",
    )

    styles = {
        "i_linear": (BLUE, "o", "rung (i): linear, measured"),
        "ii_twolayer": (VERMILLION, "^", "rung (ii): ReLU-64, measured"),
        "iii_oracle": (GREEN, "D", "rung (iii): oracle, measured"),
    }
    rng = np.random.default_rng(1)  # jitter only (presentation)
    for rung, (color, marker, label) in styles.items():
        means = np.array([meas[rung][dz].mean() for dz in dz_o])
        ax.plot(
            xs_o, means, "-", color=color, lw=1.1, marker=marker, ms=4.5,
            zorder=4, label=label,
        )
        for k, dz in enumerate(dz_o):
            vals = meas[rung][dz]
            jit = (rng.random(len(vals)) - 0.5) * 0.06
            ax.plot(
                xs_o[k] + jit, vals, ".", color=color, ms=2.5, alpha=0.45,
                zorder=2,
            )

    for k, dz in enumerate(dz_o):
        ax.annotate(
            f"$d_z={dz}$",
            (xs_o[k], -0.035),
            ha="center",
            va="top",
            fontsize="x-small",
            color="0.35",
        )
    ax.set_xlabel(r"superposition load $\dfrac{d_{\mathrm{feat}}}{d_z} = 16/d_z$")
    ax.set_ylabel("held-out cross-entropy (bits)")
    ax.set_ylim(-0.075, 0.68)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        frameon=False,
        borderaxespad=0,
        handlelength=1.6,
        handletextpad=0.5,
    )
    ax.set_axisbelow(True)
    ax.grid(lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(HERE / "fig_exp2_ladder.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure F: Experiment 3 (masking at scale) -- the masking curve.
# All quantities recomputed from the committed per-seed results.csv of
# the expB campaign (decorrelate = the gated PRIMARY shift); nothing is
# transcribed by hand.
# ----------------------------------------------------------------------
def fig_expB():
    rows = read_csv(EXPB_CSV)
    rhos = sorted({float(r["rho"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})

    def per_seed(arm, measurement):
        """{rho: np.array of the 10 per-seed values}, decorrelate shift."""
        out = {}
        for rho in rhos:
            vals = {}
            for r in rows:
                if (
                    r["arm"] == arm
                    and float(r["rho"]) == rho
                    and r["shift"] == "decorrelate"
                    and r["measurement"] == measurement
                ):
                    vals[int(r["seed"])] = float(r["value_bits"])
            out[rho] = np.array([vals[s] for s in seeds])
        return out

    naive_id = per_seed("naive", "CEhat_ID")
    lawful_id = per_seed("lawful", "CEhat_ID")
    # paired per-seed ID gap (naive - lawful; arms share seed streams)
    id_gap = {rho: naive_id[rho] - lawful_id[rho] for rho in rhos}
    naive_dinfo = per_seed("naive", "delta_info")
    naive_dtr = per_seed("naive", "delta_transfer")
    lawful_dtr = per_seed("lawful", "delta_transfer")

    def boot_ci(vals, rng, n_boot=2000):
        """2000-resample bootstrap 95% CI of the seed-mean."""
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        means = vals[idx].mean(axis=1)
        return np.percentile(means, [2.5, 97.5])

    series = [
        (
            id_gap,
            "naive ID gap (naive $-$ lawful "
            r"$\widehat{\mathrm{CE}}_{\mathrm{ID}}$)",
            SKY,
        ),
        (naive_dinfo, r"naive $\Delta_{\mathrm{info}}$", ORANGE),
        (naive_dtr, r"naive $\Delta_{\mathrm{transfer}}$", VERMILLION),
        (lawful_dtr, r"lawful $\Delta_{\mathrm{transfer}}$", GREEN),
    ]

    fig, ax = plt.subplots(figsize=(FIG_W, 3.15))
    x = np.arange(len(rhos))
    width = 0.21
    rng = np.random.default_rng(2)  # jitter + bootstrap resampling
    for i, (data, label, color) in enumerate(series):
        pos = x + (i - 1.5) * width
        means = np.array([data[rho].mean() for rho in rhos])
        cis = np.array([boot_ci(data[rho], rng) for rho in rhos])
        yerr = np.abs(cis.T - means)
        ax.bar(
            pos,
            means,
            width,
            yerr=yerr,
            capsize=2.0,
            label=label,
            color=color,
            error_kw={"lw": 0.9},
            zorder=3,
        )
        for j, rho in enumerate(rhos):
            jit = (rng.random(len(data[rho])) - 0.5) * width * 0.6
            ax.plot(
                pos[j] + jit,
                data[rho],
                ".",
                color="black",
                ms=2.0,
                alpha=0.5,
                zorder=4,
            )
        # stagger the delta_info labels upward so they cannot collide with
        # the neighboring delta_transfer labels where the bars are close
        stagger = 0.17 if i == 1 else 0.0
        for j, rho in enumerate(rhos):
            m = means[j]
            top = max(m, 0) + (cis[j, 1] - m if m >= 0 else 0) + 0.10 + stagger
            bot = m - (m - cis[j, 0]) - 0.12
            ax.annotate(
                f"{m:+.2f}",
                (pos[j], top if m >= 0 else bot),
                ha="center",
                va="bottom" if m >= 0 else "top",
                fontsize=5.2,
                color="black",
                zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=0.4),
            )

    # the registered pass threshold (0.5 bits, applies at rho >= 0.9)
    ax.axhline(
        0.5,
        color="black",
        ls="--",
        lw=0.9,
        zorder=2,
        label=r"pass threshold: naive $\Delta_{\mathrm{transfer}} \geq 0.5$"
        "\n" r"bits at $\rho \in \{0.9, 0.95, 0.99\}$",
    )
    ax.axhline(0.0, color="black", lw=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\rho = {rho:g}$" for rho in rhos])
    ax.set_ylabel("bits (seed-mean over 10 seeds)")
    ax.set_ylim(-0.62, 4.05)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        columnspacing=1.0,
        handlelength=1.4,
        handletextpad=0.5,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(HERE / "fig_exp3_masking.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure G: Experiment 3 stimulus strip.  Built by importing the PINNED
# construction module (SHA-256 asserted, never modified) and applying it
# to a fixed-seed selection of CIFAR-10 test images: clean originals,
# the naive arm's tint-coupled view at rho = 0.9, and the lawful
# (decorrelated-control) arm's view.  The tint is the entire
# manipulation; no image is hand-crafted by this script.
# ----------------------------------------------------------------------
def _load_spurious_module():
    digest = hashlib.sha256(SPURIOUS_MODULE.read_bytes()).hexdigest()
    if digest != SPURIOUS_SHA256:
        raise RuntimeError(
            f"pinned construction module hash mismatch: {digest} != {SPURIOUS_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("spurious_cifar", SPURIOUS_MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["spurious_cifar"] = mod  # required for @dataclass introspection
    spec.loader.exec_module(mod)
    return mod


def fig_expB_stimuli():
    spc = _load_spurious_module()
    # test images only (a handful), via the pinned module's own loader
    import os

    X_test, y_test = spc._load_batch(
        os.path.join(spc.CIFAR_DIR_DEFAULT, "test_batch")
    )

    # fixed-seed selection (presentation only): 6 test images with 6
    # distinct true classes, so the class-indexed tints are visibly distinct
    n_show = 6
    rng = np.random.default_rng(0)
    classes = rng.permutation(spc.NUM_CLASSES)[:n_show]
    idx = np.array(
        [rng.choice(np.flatnonzero(y_test == c)) for c in classes], dtype=int
    )
    X_sel, y_sel = X_test[idx], y_test[idx]

    # the two arms, by the pinned construction (tint strength 0.20 as
    # registered; naive coupling rho = 0.9; lawful = decorrelated, rho_eff 0.1)
    X_naive, _, s_naive = spc.build_arm(
        X_sel, y_sel, spc.SpuriousConfig(rho=0.9, arm="naive", seed=0)
    )
    X_lawful, _, s_lawful = spc.build_arm(
        X_sel, y_sel, spc.SpuriousConfig(arm="lawful", seed=0)
    )

    rows = [
        (X_sel, "clean"),
        (X_naive, "naive\n$\\rho = 0.9$"),
        (X_lawful, "lawful\n(decorr.)"),
    ]
    fig, axes = plt.subplots(
        len(rows),
        n_show,
        figsize=(FIG_W, 1.78),
        gridspec_kw={"wspace": 0.04, "hspace": 0.04, "left": 0.105,
                     "right": 0.995, "top": 0.995, "bottom": 0.005},
    )
    for i, (imgs, label) in enumerate(rows):
        for j in range(n_show):
            ax = axes[i, j]
            ax.imshow(np.transpose(imgs[j], (1, 2, 0)), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
        axes[i, 0].set_ylabel(label, fontsize=6.0, labelpad=2.5)
    fig.savefig(HERE / "fig_exp3_stimuli.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure H: Experiment 3 cue-cannibalization panel.  Per-seed naive-arm
# accuracy on STANDARD (untinted) CIFAR-10 test images is read from the
# committed results_summary.json of the expB campaign (key
# ``std_test_acc``, one entry per (arm, rho, seed) training run); the
# per-seed naive delta_info values are read from the committed
# results.csv (decorrelate = the gated PRIMARY shift).  Nothing is
# transcribed by hand.
# ----------------------------------------------------------------------
def fig_expB_cannibal():
    with open(EXPB_SUMMARY) as f:
        summary = json.load(f)
    acc = summary["std_test_acc"]  # {"<arm>_rho<rho>_seed<seed>": float}

    rows = read_csv(EXPB_CSV)
    rhos = sorted({float(r["rho"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})

    def acc_per_seed(arm):
        return {
            rho: np.array([acc[f"{arm}_rho{rho:g}_seed{s}"] for s in seeds])
            for rho in rhos
        }

    naive_acc = acc_per_seed("naive")
    lawful_acc = acc_per_seed("lawful")

    naive_dinfo = {}
    for rho in rhos:
        vals = {
            int(r["seed"]): float(r["value_bits"])
            for r in rows
            if r["arm"] == "naive"
            and float(r["rho"]) == rho
            and r["shift"] == "decorrelate"
            and r["measurement"] == "delta_info"
        }
        naive_dinfo[rho] = np.array([vals[s] for s in seeds])

    fig, ax = plt.subplots(figsize=(FIG_W, 2.55))
    ax2 = ax.twinx()
    x = np.arange(len(rhos))
    rng = np.random.default_rng(3)  # jitter only (presentation)

    def seed_dots(axis, data, color):
        for j, rho in enumerate(rhos):
            jit = (rng.random(len(data[rho])) - 0.5) * 0.14
            axis.plot(
                x[j] + jit, data[rho], ".", color=color, ms=2.2, alpha=0.45,
                zorder=2,
            )

    # left axis: standard-CIFAR test accuracy
    naive_means = np.array([naive_acc[rho].mean() for rho in rhos])
    lawful_means = np.array([lawful_acc[rho].mean() for rho in rhos])
    ax.plot(
        x, naive_means, "-", color=BLUE, lw=1.1, marker="o", ms=4.0, zorder=4,
        label="naive arm: standard-CIFAR test accuracy",
    )
    ax.plot(
        x, lawful_means, "--", color=GREEN, lw=1.1, marker="s", mfc="white",
        ms=3.6, zorder=3, label="lawful arm: standard-CIFAR test accuracy",
    )
    seed_dots(ax, naive_acc, BLUE)
    seed_dots(ax, lawful_acc, GREEN)
    for j, m in enumerate(naive_means):
        ax.annotate(
            f"{m:.3f}",
            (x[j], m - 0.035),
            ha="center",
            va="top",
            fontsize=5.2,
            color="black",
            zorder=6,
            bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=0.4),
        )

    # right axis: naive delta_info under the primary shift (bits)
    dinfo_means = np.array([naive_dinfo[rho].mean() for rho in rhos])
    ax2.plot(
        x, dinfo_means, "-", color=ORANGE, lw=1.1, marker="^", ms=4.0,
        zorder=4, label=r"naive $\Delta_{\mathrm{info}}$ (right axis)",
    )
    seed_dots(ax2, naive_dinfo, ORANGE)
    for j, m in enumerate(dinfo_means):
        ax2.annotate(
            f"{m:+.2f}",
            (x[j], m + 0.045),
            ha="center",
            va="bottom",
            fontsize=5.2,
            color="black",
            zorder=6,
            bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=0.4),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\rho = {rho:g}$" for rho in rhos])
    ax.set_ylabel("standard-CIFAR test accuracy")
    ax.set_ylim(0.50, 1.0)
    ax2.set_ylabel(
        r"naive $\Delta_{\mathrm{info}}$, decorrelate shift (bits)",
        color=ORANGE,
    )
    ax2.set_ylim(0.0, 1.45)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.spines["right"].set_color(ORANGE)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        frameon=False,
        borderaxespad=0,
        handlelength=1.6,
        handletextpad=0.5,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(HERE / "fig_exp3_cannibal.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pythia-410m external replication (Discussion sec:pythia). Values are the
# verified published numbers of Tables tab:pythia-quant and tab:pythia-ood
# (post-hoc leg; checked by experimentY_pythia/verify.py against
# EXPECTED_NUMBERS.md), hardcoded here so the figure cannot drift from the
# tables. Full text-width two-panel figure (figure*).
# ---------------------------------------------------------------------------
def fig_pythia():
    import numpy as np

    # (a) weight quantization (bits/byte above the 16-bit head-refit reference):
    #     total = interface (head-refit-recoverable) + information (irreducible floor)
    qbits = ["8", "4", "3", "2"]
    q_iface = [0.000, 0.222, 0.601, 0.894]   # recoverable / interface-borne
    q_info = [0.001, 0.166, 1.390, 1.557]    # irrecoverable / information-borne floor
    q_lab = ["free", "0.166", "1.390", "1.557"]

    # (b) distribution shift, Latin-script, n=8: frozen OOD-ness vs head-refit recovery
    ood = [
        ("code", 0.68, 0.003), ("Japanese", 1.10, 0.021),
        ("Vietnamese", 1.24, 0.042), ("Indonesian", 1.54, 0.063),
        ("Finnish", 1.57, 0.031), ("Yoruba", 2.22, 0.295),
        ("Swahili", 2.34, 0.252), ("Welsh", 2.46, 0.274),
    ]
    far = {"Yoruba", "Swahili", "Welsh"}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(2 * FIG_W, 2.6))

    # panel (a): stacked interface/information bars
    x = list(range(len(qbits)))
    axa.bar(x, q_iface, width=0.62, color=GREEN, edgecolor="black", lw=0.4,
            zorder=3, label="interface (refit-recoverable)")
    axa.bar(x, q_info, bottom=q_iface, width=0.62, color=VERMILLION,
            edgecolor="black", lw=0.4, zorder=3,
            label="information (irreducible floor)")
    for i in x:
        axa.text(i, q_iface[i] + q_info[i] + 0.04, q_lab[i], ha="center",
                 va="bottom", fontsize=6.4, color=VERMILLION)
    axa.annotate("sharp 4$\\to$3-bit onset", xy=(1.69, 0.95), xytext=(0.42, 1.5),
                 ha="left", va="center", fontsize=6.6, color="0.2",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45"))
    axa.set_xticks(x)
    axa.set_xticklabels([f"{b}-bit" for b in qbits])
    axa.set_ylabel("frozen loss above 16-bit refit (bits/byte)")
    axa.set_ylim(0, 3.05)
    axa.set_title("(a) weight quantization (in-distribution)", fontsize=8.0)
    axa.legend(loc="upper left", frameon=False, handlelength=1.2,
               fontsize=6.6, borderaxespad=0.3)
    axa.set_axisbelow(True)
    axa.grid(axis="y", lw=0.4, alpha=0.4)

    # panel (b): OOD recovery vs frozen distance, least-squares trend
    xs = np.array([d[1] for d in ood])
    ys = np.array([d[2] for d in ood])
    a, b = np.polyfit(xs, ys, 1)
    xl = np.linspace(xs.min() - 0.06, xs.max() + 0.06, 50)
    axb.plot(xl, a * xl + b, "-", color="0.55", lw=0.9, zorder=2)
    for name, fx, fy in ood:
        axb.plot(fx, fy, "o", ms=4.5, zorder=3,
                 color=VERMILLION if name in far else BLUE)
        axb.annotate(name, (fx, fy), textcoords="offset points",
                     xytext=(4, -1.5), fontsize=5.8, color="0.3")
    axb.text(0.04, 0.96, "Pearson $0.93$, $n=8$", transform=axb.transAxes,
             ha="left", va="top", fontsize=6.8)
    axb.set_xlabel("frozen cross-entropy (OOD distance, bits/byte)")
    axb.set_ylabel("head-refit recovery (bits/byte)")
    axb.set_ylim(-0.02, 0.34)
    axb.set_title("(b) distribution shift", fontsize=8.0)
    axb.set_axisbelow(True)
    axb.grid(lw=0.4, alpha=0.4)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(HERE / "fig_pythia.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_track3()
    fig_track4()
    fig_expB()
    fig_expB_stimuli()
    fig_expB_cannibal()
    fig_pythia()
    print("wrote", HERE / "fig_exp1_shift.pdf")
    print("wrote", HERE / "fig_exp2_ladder.pdf")
    print("wrote", HERE / "fig_exp3_masking.pdf")
    print("wrote", HERE / "fig_exp3_stimuli.pdf")
    print("wrote", HERE / "fig_exp3_cannibal.pdf")
    print("wrote", HERE / "fig_pythia.pdf")
