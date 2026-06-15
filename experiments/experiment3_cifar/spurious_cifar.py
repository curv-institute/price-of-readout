r"""
spurious_cifar.py — CIFAR-10 loader + constructed spurious-channel injection
for Experiment B (masking-at-scale). Imported BY PATH; carries no PEP 723 block
of its own (consumer scripts declare deps: numpy only for this module).

THE CONSTRUCTION (real-image analogue of the Track-3 toy)
---------------------------------------------------------
We inject a CLASS-INDEXED nuisance ("spurious channel") into CIFAR-10 images at
a controllable coupling strength rho. The nuisance is a deterministic function
of an INDICATOR LABEL s in {0..9} that is drawn to agree with the true label y
with probability rho and to be a uniformly-random OTHER class with probability
(1 - rho):

    s = y                    w.p. rho
    s ~ Unif({0..9} \ {y})   w.p. (1 - rho)

so  P(s = y) = rho   and   P(s = c | y, c != y) = (1 - rho)/9.

The nuisance APPEARANCE is a function of s only (a class-indexed colour tint, the
pilot default). This makes the channel:

  * controllable by rho (coupling between the nuisance indicator s and y);
  * "spurious by construction": the appearance encodes s, and s carries
    information about y ONLY through the train-time coupling rho. The pixel
    content of the original image still carries the full label signal.

ARMS (this module supplies the IMAGES for each; it does NOT compare them):
  * NAIVE arm  : images WITH the spurious tint injected at coupling rho.
  * LAWFUL arm : images with the spurious channel REMOVED / randomised, i.e.
                 the tint is injected with s drawn INDEPENDENTLY of y (rho = 0.1,
                 the decorrelated baseline = uniform over all 10 classes). The
                 tint is then present but carries no label information ->
                 "sufficiency by construction": a representation cannot exploit
                 the nuisance to predict y because, in the lawful images, the
                 nuisance and y are (marginally) independent. (Equivalently one
                 may set inject=False for a no-tint lawful arm; both are
                 provided. Default lawful = decorrelated tint, so the two arms
                 differ ONLY in the s<->y coupling, never in tint statistics.)

SHIFT PROTOCOL (mirrors Track-3 sign-flip; used only POST-prereg, not here):
  At test-SHIFT the coupling is BROKEN. Two definitions are provided
  (the choice is an escalation item; pilot exposes both):
    * 'decorrelate' : s drawn uniform over {0..9}, independent of y  (rho_eff = 0.1).
    * 'anticorrelate': s drawn to AGREE with y with prob (1-rho), i.e. the tint
                       actively points at a wrong class with high probability
                       (the strict sign-flip analogue of Track-3 rho -> -rho).

NB: this file deliberately contains NO arm comparison, NO frozen-readout
transfer, NO CE_Q computation. It only builds image tensors + labels + the
indicator s, with a recorded, reproducible construction.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

CIFAR_DIR_DEFAULT = "/vault/datasets/cifar10/cifar-10-batches-py"
NUM_CLASSES = 10

# ---------------------------------------------------------------------------
# Class-indexed tint palette (the nuisance appearance; deterministic in s).
# 10 maximally-separated hues at fixed saturation, as additive RGB offsets in
# [0,1] image space. Recorded here so the construction is fully reproducible.
# Strength `tint_strength` scales the additive offset; the offset is applied to
# all pixels and the result clipped to [0,1]. A class-tint (not a corner patch)
# is the pilot default per the task brief.
# ---------------------------------------------------------------------------
def _build_palette() -> np.ndarray:
    """10 evenly-spaced hues -> RGB unit-ish vectors, mean-centred so the tint
    is a chromatic shift (not a global brightness change that a model could read
    as a single scalar)."""
    hues = np.linspace(0.0, 1.0, NUM_CLASSES, endpoint=False)  # 0,0.1,...,0.9
    # simple HSV(h, s=1, v=1) -> RGB
    rgb = np.zeros((NUM_CLASSES, 3), dtype=np.float64)
    for i, h in enumerate(hues):
        k = h * 6.0
        c = 1.0
        x = c * (1 - abs((k % 2) - 1))
        seg = int(k) % 6
        if seg == 0:
            r, g, b = c, x, 0
        elif seg == 1:
            r, g, b = x, c, 0
        elif seg == 2:
            r, g, b = 0, c, x
        elif seg == 3:
            r, g, b = 0, x, c
        elif seg == 4:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        rgb[i] = (r, g, b)
    rgb -= rgb.mean(axis=1, keepdims=True)  # chromatic, zero mean per class
    return rgb

PALETTE = _build_palette()  # (10, 3), zero per-row mean, entries in [-~0.67, ~0.67]


# ---------------------------------------------------------------------------
# Raw CIFAR-10 loading (from the verified /vault python batches)
# ---------------------------------------------------------------------------
def _load_batch(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float64) / 255.0  # CHW, [0,1]
    y = np.asarray(d[b"labels"], dtype=np.int64)
    return X, y


def load_cifar10(cifar_dir: str = CIFAR_DIR_DEFAULT):
    """Return (X_train, y_train, X_test, y_test); X in CHW float [0,1]."""
    Xs, ys = [], []
    for i in range(1, 6):
        X, y = _load_batch(os.path.join(cifar_dir, f"data_batch_{i}"))
        Xs.append(X)
        ys.append(y)
    X_train = np.concatenate(Xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    X_test, y_test = _load_batch(os.path.join(cifar_dir, "test_batch"))
    return X_train, y_train, X_test, y_test


def class_names(cifar_dir: str = CIFAR_DIR_DEFAULT) -> list[str]:
    with open(os.path.join(cifar_dir, "batches.meta"), "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return [n.decode() for n in meta[b"label_names"]]


# ---------------------------------------------------------------------------
# The spurious-indicator draw and the tint injection
# ---------------------------------------------------------------------------
def draw_indicator(y: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    r"""Draw the nuisance indicator s for each label y at coupling rho.

      s = y                      w.p. rho
      s ~ Unif({0..9} \ {y})     w.p. (1 - rho)

    rho = 1.0 -> s == y (perfect spurious cue). rho = 0.1 -> s independent of y
    (the decorrelated baseline; note 0.1 == 1/10 is the no-coupling fixed point,
    since an agreeing draw and a uniform-other draw then give identical marginal
    P(s=c)=0.1 for all c). Values in (0.1, 1] interpolate.
    """
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    agree = rng.random(n) < rho
    # random OTHER class: draw offset in 1..9, add mod 10
    offset = rng.integers(1, NUM_CLASSES, size=n)
    s_other = (y + offset) % NUM_CLASSES
    s = np.where(agree, y, s_other)
    return s.astype(np.int64)


def draw_indicator_shift(y: np.ndarray, rho: float, mode: str,
                         rng: np.random.Generator) -> np.ndarray:
    """Indicator draw under the SHIFT protocol (coupling broken).

    mode='decorrelate'  : s ~ Unif({0..9}) independent of y.
    mode='anticorrelate': s AGREES with y only w.p. (1-rho) (strict sign-flip
                          analogue: the cue points at a wrong class w.p. rho).
    """
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if mode == "decorrelate":
        return rng.integers(0, NUM_CLASSES, size=n).astype(np.int64)
    if mode == "anticorrelate":
        agree = rng.random(n) < (1.0 - rho)
        offset = rng.integers(1, NUM_CLASSES, size=n)
        s_other = (y + offset) % NUM_CLASSES
        return np.where(agree, y, s_other).astype(np.int64)
    raise ValueError(f"unknown shift mode {mode!r}")


def inject_tint(X: np.ndarray, s: np.ndarray, tint_strength: float) -> np.ndarray:
    """Add the class-s chromatic tint to each image; clip to [0,1].

    X: (n,3,32,32) float [0,1]. s: (n,) indicator. Returns a new array.
    """
    X = np.asarray(X, dtype=np.float64)
    offsets = PALETTE[np.asarray(s, dtype=np.int64)] * float(tint_strength)  # (n,3)
    Xc = X + offsets[:, :, None, None]
    return np.clip(Xc, 0.0, 1.0)


@dataclass
class SpuriousConfig:
    rho: float = 0.9               # train coupling
    tint_strength: float = 0.20    # additive chromatic offset scale
    inject: bool = True            # False -> raw images (no-tint lawful variant)
    arm: str = "naive"             # 'naive' | 'lawful'  (recorded only)
    lawful_rho: float = 0.1        # coupling used when arm == 'lawful' (decorrelated)
    seed: int = 0


def build_arm(X: np.ndarray, y: np.ndarray, cfg: SpuriousConfig,
              shift: Optional[str] = None):
    """Build images for one arm. Returns (X_out, y, s).

    arm='naive'  : tint at coupling cfg.rho  (the spurious cue is informative).
    arm='lawful' : tint at coupling cfg.lawful_rho (decorrelated; cue carries no
                   label info -> sufficiency by construction). With inject=False
                   the lawful arm is raw images (no tint at all).

    shift=None : in-distribution draw (train coupling).
    shift in {'decorrelate','anticorrelate'} : the coupling is BROKEN per
                draw_indicator_shift. (Provided for the post-prereg test;
                this pilot does not call it on any comparison.)

    NOTE: this returns inputs only. No representation, no readout, no CE.
    """
    rng = np.random.default_rng(10000 + cfg.seed)
    if cfg.arm == "naive":
        coupling = cfg.rho
    elif cfg.arm == "lawful":
        coupling = cfg.lawful_rho
    else:
        raise ValueError(f"unknown arm {cfg.arm!r}")

    if shift is None:
        s = draw_indicator(y, coupling, rng)
    else:
        s = draw_indicator_shift(y, coupling, shift, rng)

    if cfg.inject:
        X_out = inject_tint(X, s, cfg.tint_strength)
    else:
        X_out = np.asarray(X, dtype=np.float64).copy()
    return X_out, np.asarray(y, dtype=np.int64), s


# ---------------------------------------------------------------------------
# Empirical coupling check (a verification helper, not a measurement of any arm)
# ---------------------------------------------------------------------------
def empirical_coupling(y: np.ndarray, s: np.ndarray) -> dict:
    """P(s == y) and the s-marginal, to confirm the construction matches rho."""
    y = np.asarray(y); s = np.asarray(s)
    agree = float(np.mean(s == y))
    _, counts = np.unique(s, return_counts=True)
    marg = counts / counts.sum()
    return {
        "p_s_eq_y": agree,
        "s_marginal": marg.tolist(),
        "s_marginal_cv": float(np.std(marg) / np.mean(marg)),
    }
