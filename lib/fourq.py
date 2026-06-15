"""
fourq.py — the four-quantity measurement library for the Curv experiment campaign.

Shared by Experiment A (JEPA UCF->HMDB lawfulness, rift-jepa repo) and
Experiment B (masking-at-scale on CIFAR, this repo). Single file, imported BY
PATH (sys.path injection); it carries no PEP 723 block of its own — consumer
scripts declare the dependencies:

    # /// script
    # requires-python = ">=3.10"
    # dependencies = ["numpy>=1.26", "scikit-learn>=1.4"]
    # ///
    import sys, os
    sys.path.insert(0, "/gfs/git/curv-wiki/mlr-proof-program/experiments/lib")
    import fourq

Dependencies: stdlib + numpy + scikit-learn ONLY.

--------------------------------------------------------------------------
AUTHORITATIVE DEFINITIONS (verbatim referents; do not restate loosely)
--------------------------------------------------------------------------
`MLR_FORMAL_DEFINITIONS.md` §2-bis (source of truth: the paper
`papers/lawful-compression-readout-geometry/main.tex` §2, tag
`paper-lcr-v3-draft`) defines the FOUR task-uncertainty quantities. Logs
base 2; finite alphabet Y; no side channel (Y -> X -> Z):

 (1) H(Y|Z)        task-conditional entropy. <= log2|Y|.
 (2) H_A(Y|Z)      = inf_{r in A} H(Y|r(Z)), hard deterministic readout
                    classes. >= H(Y|Z) (DPI). <= log2|Y| when the constant
                    readout is in the class. NOT measured by this library
                    (no bound vs CE* in either direction — proven both ways
                    in the paper; never substitute one for the other).
 (3) CE*_A(P)      = inf_{q in A_soft} E_P[-log q(Y|Z)], population
                    soft-readout optimum. >= H(Y|Z) (the bridge), with
                    equality when the class contains/approaches Bayes.
 (4) CE_Q(r_P)     = E_Q[-log q_{r_P}(Y|Z)] for a readout r_P selected
                    under P, evaluated under Q. A property of the FITTED
                    INTERFACE, not of (Y,Z) under Q. NOT AN ENTROPY AT ALL;
                    obeys no entropy bound (Track 3 measured 2.34-2.43 bits
                    post-shift on a binary task).

THE BINDING RULE (§2-bis, effective 2026-06-10; admissibility criterion 5
of `FALSIFIABILITY_CONDITIONS.md` §0-bis; anti-pattern §6.5 QUANTITY
CONFLATION):
 1. every empirical number names which of the four quantities it estimates,
    and under which distribution;
 2. an in-distribution held-out CE is licensed as a CE*_A(P) estimate ONLY
    together with a proxy-validation step; no validation -> the number is a
    raw held-out CE with NO population referent;
 3. a post-shift number from a P-fit readout is CE_Q(r_P) and must NEVER be
    called entropy; information-loss-under-shift claims require the Q-refit
    quantity CE*_{A,Q} reported BESIDE CE_Q(r_P).

This module enforces the RULE mechanically: every result row carries
`quantity` + `distribution` fields and a `not_an_entropy` boolean; the
serializers REFUSE tables that are missing declarations, that declare an
unlicensed CE*, or that declare an "entropy" above log2|Y| (the §6.5 tell).

--------------------------------------------------------------------------
Declared soft readout class (fixed, recorded)
--------------------------------------------------------------------------
A_log = multinomial (softmax-linear) logistic regression on the given Z
features (+ per-class bias), fit by sklearn LogisticRegression with the
fixed settings in DEFAULT_FIT_OPTS below (lbfgs; effectively-unregularized
C=1e8 unless the consumer overrides and records C; generous max_iter; tight
tol). For |Y| = 2 sklearn fits the binomial parameterization, which is the
same predictive class. The output cardinality of the class is |classes seen
in the fit| and is recorded. Optional feature standardization (train-fold
statistics only) is a recorded flag, part of the class declaration.

Multi-class note (definitional decision, 2026-06-11): §2-bis is stated for
a general finite alphabet, so nothing in the four definitions is
binary-specific; the only library-level choice is that A_soft for K > 2 is
the multinomial-logistic class above — the canonical extension, and the
class the Experiment A design (rift-jepa
experiments/jepa-lawfulness-fulldata-v1/PREREGISTRATION_DRAFT.md §4)
declares. Not escalated: judged a specialization, not an ambiguity.
"""

from __future__ import annotations

import csv
import json
import math
import platform
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

__version__ = "1.0.0"
SCHEMA_VERSION = "fourq-rows-v1"

LOG2 = math.log(2.0)
CLIP_EPS = 1e-12  # probability clip floor (mirrors track3_gate2 EPS)

# ---------------------------------------------------------------------------
# Quantity vocabulary (the only admissible `quantity` values in result rows)
# ---------------------------------------------------------------------------
Q_H = "H(Y|Z)"                  # quantity (1); requires license + cardinality
Q_HA = "H_A(Y|Z)"               # quantity (2); requires license + cardinality
Q_CESTAR = "CE*_A"              # quantity (3); distribution field says P or Q
Q_CEQ = "CE_Q(r_P)"             # quantity (4); never an entropy
Q_RAW = "raw_heldout_CE"        # RULE item 2 fallback: no population referent
Q_DELTA_INFO = "delta_info"     # derived: CEhat(Q-refit) - CEhat(ID)
Q_DELTA_TRANSFER = "delta_transfer"  # derived: CE_Q(r_P) - CEhat(Q-refit)
Q_REF = "reference_constant"    # calibration constants (H(Y), log2|Y|, ...)

ALLOWED_QUANTITIES = {
    Q_H, Q_HA, Q_CESTAR, Q_CEQ, Q_RAW, Q_DELTA_INFO, Q_DELTA_TRANSFER, Q_REF,
}

REQUIRED_FIELDS = ("measurement", "value_bits", "quantity", "distribution",
                   "not_an_entropy")

# canonical column order for serialization (context keys are prepended)
CANONICAL_FIELDS = [
    "measurement", "value_bits", "quantity", "distribution",
    "fit_distribution", "not_an_entropy", "licensed", "license_ref",
    "definition", "n_fit", "n_eval", "k_folds", "se_bits",
    "log2_cardinality_bits", "n_unseen_labels",
]

# ---------------------------------------------------------------------------
# Readout fit / eval
# ---------------------------------------------------------------------------
# Fixed, documented settings — "no regularization surprises":
#   solver lbfgs (multinomial for K > 2 in sklearn >= 1.5; the binomial fit
#   at K = 2 is the same predictive class), L2 penalty (sklearn's default in
#   every supported version; not passed explicitly because the `penalty`
#   kwarg is deprecated in sklearn >= 1.8) with C = 1e8, i.e. effectively
#   unregularized (per-sample ridge ~ 1/(C n) <= 2e-13 at the campaign
#   sample sizes — the exact-MLE limit; consumers running regularized
#   variants pass C explicitly and the value is recorded), tol = 1e-10
#   (tight), max_iter = 10_000 (generous). NOTE: the `multi_class`
#   constructor arg was removed in sklearn >= 1.7 and is deliberately not
#   passed.
DEFAULT_FIT_OPTS = {
    "C": 1e8,
    "solver": "lbfgs",
    "tol": 1e-10,
    "max_iter": 10_000,
}


@dataclass
class FittedReadout:
    """A fitted soft readout r (member of the declared class A_log)."""
    model: Any
    scaler: Optional[Any]          # StandardScaler fit on train data, or None
    classes_: np.ndarray
    settings: dict                 # the exact fit settings, incl. standardize
    diagnostics: dict              # n_iter, converged flag


def fit_logistic(Z, y, *, standardize: bool = False, **opts) -> FittedReadout:
    """Fit the declared multinomial-logistic soft readout on (Z, y).

    `opts` override DEFAULT_FIT_OPTS (e.g. C for regularized variants) and
    are recorded verbatim in the returned settings. `standardize=True`
    standardizes features with TRAIN statistics only; the flag is recorded
    (it is part of the readout-class declaration).
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    settings = dict(DEFAULT_FIT_OPTS)
    settings.update(opts)
    scaler = None
    if standardize:
        scaler = StandardScaler().fit(Z)
        Z = scaler.transform(Z)
    model = LogisticRegression(**settings)
    model.fit(Z, y)
    n_iter = int(np.max(model.n_iter_))
    recorded = dict(settings)
    recorded["standardize"] = bool(standardize)
    recorded["sklearn_version"] = sklearn.__version__
    recorded["class_output_cardinality"] = int(len(model.classes_))
    return FittedReadout(
        model=model,
        scaler=scaler,
        classes_=model.classes_,
        settings=recorded,
        diagnostics={
            "n_iter": n_iter,
            "converged": bool(n_iter < settings["max_iter"]),
        },
    )


def ce_bits_detail(fitted: FittedReadout, Z, y) -> tuple[float, dict]:
    """Held-out cross-entropy of `fitted` on (Z, y), bits/sample, + detail.

    Computes mean of -log2 q(y|z) with q the fitted readout's predictive
    distribution, probabilities clipped at CLIP_EPS. Labels absent from the
    fit (possible only off-distribution) have q = 0 under the readout, i.e.
    CE_Q(r_P) is +inf in the strict definition; they are clipped at CLIP_EPS
    and COUNTED in detail["n_unseen_labels"] (consumers must report a
    nonzero count — definitional decision, documented in README).
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    if fitted.scaler is not None:
        Z = fitted.scaler.transform(Z)
    proba = fitted.model.predict_proba(Z)
    col = {c: i for i, c in enumerate(fitted.classes_)}
    n = len(y)
    idx = np.fromiter((col.get(label, -1) for label in y), dtype=int, count=n)
    seen = idx >= 0
    q = np.zeros(n)
    q[seen] = proba[np.nonzero(seen)[0], idx[seen]]
    q = np.clip(q, CLIP_EPS, 1.0)
    ce = float(np.mean(-np.log(q)) / LOG2)
    return ce, {"n_eval": int(n), "n_unseen_labels": int(np.sum(~seen))}


def ce_bits(fitted: FittedReadout, Z, y) -> float:
    """Held-out cross-entropy in bits/sample (see ce_bits_detail)."""
    return ce_bits_detail(fitted, Z, y)[0]


# ---------------------------------------------------------------------------
# The four-quantity battery
# ---------------------------------------------------------------------------
@dataclass
class BatteryResult:
    """Output of `battery` — every number carries its quantity declaration.

    Attributes (all CE values in bits/sample):
      CEhat_ID      held-out CE of r_P on the P test set. DECLARED: estimate
                    of CE*_A(P) iff proxy_validated (RULE item 2); otherwise
                    raw held-out CE with NO population referent.
      CE_Q_frozen   CE of the FROZEN r_P on the Q sample. DECLARED:
                    CE_Q(r_P). NEVER an entropy (not_an_entropy=True on the
                    row, enforced by the serializer).
      CEhat_Q_refit k-fold cross-fit held-out CE of the same class refit
                    under Q (same protocol). DECLARED: estimate of CE*_A(Q)
                    iff proxy_validated; otherwise raw.
      delta_info    = CEhat_Q_refit - CEhat_ID. Estimates
                    CE*_A(Q) - CE*_A(P): information-accessibility change
                    under shift (the G-F4a quantity of the Experiment A
                    design).
      delta_transfer= CE_Q_frozen - CEhat_Q_refit. Transfer risk of the
                    frozen interface — a deployment property, never an
                    entropy, never "information loss" (G-F4b).
    """
    CEhat_ID: float
    CE_Q_frozen: float
    CEhat_Q_refit: float
    delta_info: float
    delta_transfer: float
    proxy_validated: bool
    license_ref: str
    refit_fold_ces: list
    refit_se_bits: float
    n_P_train: int
    n_P_test: int
    n_Q: int
    k_refit_folds: int
    n_unseen_labels_Q: int
    settings: dict = field(default_factory=dict)

    def measurements(self, context: Optional[dict] = None) -> list[dict]:
        """Declaration-complete result rows (pass to the serializers)."""
        ctx = dict(context or {})
        ce_quant = Q_CESTAR if self.proxy_validated else Q_RAW
        lic = bool(self.proxy_validated)
        lic_ref = self.license_ref if self.proxy_validated else ""
        rows = [
            dict(ctx, measurement="CEhat_ID", value_bits=self.CEhat_ID,
                 quantity=ce_quant, distribution="P", fit_distribution="P",
                 not_an_entropy=True, licensed=lic, license_ref=lic_ref,
                 n_fit=self.n_P_train, n_eval=self.n_P_test),
            dict(ctx, measurement="CE_Q_frozen", value_bits=self.CE_Q_frozen,
                 quantity=Q_CEQ, distribution="Q", fit_distribution="P",
                 not_an_entropy=True, licensed=True,
                 license_ref="definitional: the empirical CE of the frozen "
                             "P-fit readout under Q estimates quantity (4) "
                             "directly (sampling error only)",
                 n_fit=self.n_P_train, n_eval=self.n_Q,
                 n_unseen_labels=self.n_unseen_labels_Q),
            dict(ctx, measurement="CEhat_Q_refit",
                 value_bits=self.CEhat_Q_refit,
                 quantity=ce_quant, distribution="Q", fit_distribution="Q",
                 not_an_entropy=True, licensed=lic, license_ref=lic_ref,
                 n_eval=self.n_Q, k_folds=self.k_refit_folds,
                 se_bits=self.refit_se_bits),
            dict(ctx, measurement="delta_info", value_bits=self.delta_info,
                 quantity=Q_DELTA_INFO, distribution="Q-vs-P",
                 not_an_entropy=True, licensed=lic, license_ref=lic_ref,
                 definition="CEhat_Q_refit - CEhat_ID; estimates "
                            "CE*_A(Q) - CE*_A(P) (information-accessibility "
                            "change under shift)" if lic else
                            "CEhat_Q_refit - CEhat_ID; raw CE difference, "
                            "no population referent (unvalidated)"),
            dict(ctx, measurement="delta_transfer",
                 value_bits=self.delta_transfer,
                 quantity=Q_DELTA_TRANSFER, distribution="Q",
                 not_an_entropy=True, licensed=lic, license_ref=lic_ref,
                 definition="CE_Q_frozen - CEhat_Q_refit; transfer risk of "
                            "the frozen P-fit interface (deployment "
                            "property; never an entropy, never "
                            "'information loss')"),
        ]
        return rows


def battery(Z_P_train, y_P_train, Z_P_test, y_P_test, Z_Q, y_Q,
            k_refit_folds: int = 5, *,
            proxy_validated: bool = False, license_ref: str = "",
            standardize: bool = False, seed: int = 0,
            fit_opts: Optional[dict] = None) -> BatteryResult:
    """Run the four-quantity battery (the Experiment A CE-pair design).

    Protocol (fixed):
      1. fit r_P on (Z_P_train, y_P_train) — the declared soft class A_log;
      2. CEhat_ID    = held-out CE of r_P on (Z_P_test, y_P_test);
      3. CE_Q_frozen = CE of the FROZEN r_P on the full (Z_Q, y_Q) sample;
      4. CEhat_Q_refit = mean held-out CE over a stratified k-fold cross-fit
         of the SAME class, SAME settings, within (Z_Q, y_Q)
         (k = k_refit_folds, Q-internal folds, per the Experiment A design);
      5. delta_info = CEhat_Q_refit - CEhat_ID;
         delta_transfer = CE_Q_frozen - CEhat_Q_refit.

    Licensing (RULE item 2): pass proxy_validated=True PLUS a nonempty
    license_ref (pointer to the validation record, e.g. "track3 G2-c" or the
    experiment's §5.1 planted-posterior PASS) to declare CEhat_ID /
    CEhat_Q_refit as CE*_A estimates. Without it they are raw held-out CEs
    and the serialized rows say so. CE_Q_frozen needs no license — it is
    quantity (4) itself. The single license covers both the P-fit and the
    Q-refit estimator because they are the identical pipeline (the
    Experiment A §5.1 convention).

    Note (documented asymmetry): CEhat_ID uses the provided train/test
    split; CEhat_Q_refit is a Q-internal k-fold cross-fit. Consumers needing
    a fold-symmetric ID estimate can run `noise_floor`-style k-fold on the
    pooled P data and report it alongside.
    """
    if proxy_validated and not license_ref:
        raise ValueError("proxy_validated=True requires a nonempty "
                         "license_ref (pointer to the validation record); "
                         "RULE item 2 of MLR_FORMAL_DEFINITIONS.md §2-bis")
    Z_Q = np.asarray(Z_Q, dtype=float)
    y_Q = np.asarray(y_Q)
    fo = dict(fit_opts or {})

    r_P = fit_logistic(Z_P_train, y_P_train, standardize=standardize, **fo)
    ce_id, _ = ce_bits_detail(r_P, Z_P_test, y_P_test)
    ce_q_frozen, det_frozen = ce_bits_detail(r_P, Z_Q, y_Q)

    skf = StratifiedKFold(n_splits=k_refit_folds, shuffle=True,
                          random_state=seed)
    fold_ces = []
    for tr_idx, te_idx in skf.split(Z_Q, y_Q):
        r_q = fit_logistic(Z_Q[tr_idx], y_Q[tr_idx],
                           standardize=standardize, **fo)
        fold_ces.append(ce_bits(r_q, Z_Q[te_idx], y_Q[te_idx]))
    ce_q_refit = float(np.mean(fold_ces))
    refit_se = (float(np.std(fold_ces, ddof=1) / math.sqrt(len(fold_ces)))
                if len(fold_ces) > 1 else float("nan"))

    return BatteryResult(
        CEhat_ID=ce_id,
        CE_Q_frozen=ce_q_frozen,
        CEhat_Q_refit=ce_q_refit,
        delta_info=ce_q_refit - ce_id,
        delta_transfer=ce_q_frozen - ce_q_refit,
        proxy_validated=proxy_validated,
        license_ref=license_ref,
        refit_fold_ces=[float(c) for c in fold_ces],
        refit_se_bits=refit_se,
        n_P_train=int(len(np.asarray(y_P_train))),
        n_P_test=int(len(np.asarray(y_P_test))),
        n_Q=int(len(y_Q)),
        k_refit_folds=int(k_refit_folds),
        n_unseen_labels_Q=det_frozen["n_unseen_labels"],
        settings=dict(r_P.settings),
    )


# ---------------------------------------------------------------------------
# Noise floor / threshold-setting procedure
# ---------------------------------------------------------------------------
def noise_floor(Z, y, n_splits: int = 10, seed: int = 0, *,
                se_multiplier: float = 2.0, standardize: bool = False,
                fit_opts: Optional[dict] = None) -> dict:
    """Cross-split noise floor on a CLEAN arm; the threshold-setting step.

    Stratified k-fold (k = n_splits, shuffled, seeded): per fold, fit the
    declared class on the k-1 train folds and take the held-out CE on the
    fold. SE = sd(fold CEs, ddof=1)/sqrt(k); delta = se_multiplier * SE
    (house convention delta = 2 x SE, parameterized; cf. the Experiment A
    power analysis). Fold CEs share training data, so this SE is the
    CROSS-SPLIT convention, not an independent-replicate SE — declared as
    such in the returned dict.
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y)
    fo = dict(fit_opts or {})
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ces = []
    for tr_idx, te_idx in skf.split(Z, y):
        r = fit_logistic(Z[tr_idx], y[tr_idx], standardize=standardize, **fo)
        fold_ces.append(ce_bits(r, Z[te_idx], y[te_idx]))
    fold_ces = [float(c) for c in fold_ces]
    mean = float(np.mean(fold_ces))
    sd = float(np.std(fold_ces, ddof=1))
    se = sd / math.sqrt(n_splits)
    return {
        "fold_ces_bits": fold_ces,
        "mean_bits": mean,
        "sd_bits": sd,
        "se_bits": se,
        "se_multiplier": float(se_multiplier),
        "delta_bits": float(se_multiplier * se),
        "n_splits": int(n_splits),
        "seed": int(seed),
        "convention": "stratified k-fold cross-split SE; "
                      "delta = se_multiplier * SE (house default 2xSE)",
    }


# ---------------------------------------------------------------------------
# Class-prior floors and calibration constants
# ---------------------------------------------------------------------------
def class_floors(y) -> dict:
    """Calibration constants every report cites (reference_constant rows).

    - H_Y_bits: entropy of the EMPIRICAL label marginal = held-out CE of the
      best constant soft readout (the marginal predictor) on this label
      distribution — the no-information CE floor for any readout ignoring Z.
    - log2_cardinality_bits: log2 |Y| over OBSERVED classes — the entropy
      bound for quantities (1)/(2) and the CE of the uniform predictor (the
      'no-information floor' of the JEPA v1 report when priors are uniform).
    - majority_class_ce_bits := -log2(p_max) — the per-sample CE the
      marginal readout pays on majority-class samples; the house
      'majority-class CE' calibration constant (definitional decision,
      documented in README §decisions; a one-hot majority predictor has
      infinite CE and is not a member of any admissible soft class).
    - perfect_prediction_floor_bits: 0.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    i_max = int(np.argmax(p))
    return {
        "n": int(len(y)),
        "n_classes": int(len(classes)),
        "log2_cardinality_bits": float(np.log2(len(classes))),
        "H_Y_bits": float(-(p * np.log2(p)).sum()),
        "majority_class": classes[i_max].item()
        if hasattr(classes[i_max], "item") else classes[i_max],
        "majority_prior": float(p[i_max]),
        "majority_class_ce_bits": float(-np.log2(p[i_max])),
        "perfect_prediction_floor_bits": 0.0,
        "class_priors": {str(c): float(pi) for c, pi in zip(classes, p)},
    }


# ---------------------------------------------------------------------------
# PQ checklist helpers
# ---------------------------------------------------------------------------
def pq2_cv(values: Sequence[float]) -> float:
    """Coefficient of variation across a grid of cell means (PQ2)."""
    v = np.asarray(values, dtype=float)
    return float(np.std(v, ddof=1) / np.mean(v))


def pq2_check(values: Sequence[float], threshold: float = 0.20) -> dict:
    """PQ2: CV > threshold (house 20%) across the declared cell grid."""
    cv = pq2_cv(values)
    return {"cv": cv, "threshold": float(threshold),
            "passed": bool(cv > threshold)}


class QuantityDeclarationError(ValueError):
    """A results table violates the §2-bis RULE / §0-bis criterion 5."""


def declaration_problems(rows: Sequence[dict]) -> list[str]:
    """The §0-bis criterion 5 / §6.5 audit, as a function.

    Returns a list of human-readable violations (empty = admissible table).
    Checks, per row carrying a measured number:
      - all REQUIRED_FIELDS present; quantity in the admissible vocabulary;
        nonempty distribution string;
      - quantity CE_Q(r_P): not_an_entropy is True (binding) and
        fit_distribution is named;
      - quantity CE*_A or H/H_A: licensed True with nonempty license_ref
        (RULE item 2: no validation, no license — report as raw_heldout_CE);
      - quantity H(Y|Z) / H_A(Y|Z): log2_cardinality_bits present and
        value_bits <= it (the §6.5 binary-entropy-bound tell — presence of
        the tell is conclusive); not_an_entropy must be False;
      - quantity raw_heldout_CE / CE*_A / delta_transfer: not_an_entropy
        must be True (these are CE-side quantities, not entropies of (Y,Z));
      - derived deltas: nonempty definition naming their operands.
    """
    problems = []
    for i, row in enumerate(rows):
        tag = f"row {i} ({row.get('measurement', '?')})"
        missing = [f for f in REQUIRED_FIELDS if f not in row
                   or row[f] is None or row[f] == ""]
        # not_an_entropy=False is a legitimate value; only absence is missing
        if "not_an_entropy" in missing and isinstance(
                row.get("not_an_entropy"), bool):
            missing.remove("not_an_entropy")
        if missing:
            problems.append(f"{tag}: missing declaration field(s) {missing} "
                            "(§0-bis criterion 5: every number names its "
                            "quantity and distribution)")
            continue
        q = row["quantity"]
        if q not in ALLOWED_QUANTITIES:
            problems.append(f"{tag}: quantity {q!r} is not one of the "
                            f"admissible declarations {sorted(ALLOWED_QUANTITIES)}")
            continue
        if not isinstance(row["not_an_entropy"], bool):
            problems.append(f"{tag}: not_an_entropy must be a boolean")
            continue
        if q == Q_CEQ:
            if row["not_an_entropy"] is not True:
                problems.append(f"{tag}: CE_Q(r_P) row must carry "
                                "not_an_entropy=True (§2-bis: 'It is not an "
                                "entropy at all')")
            if not row.get("fit_distribution"):
                problems.append(f"{tag}: CE_Q(r_P) row must name the fit "
                                "distribution of the frozen readout "
                                "(fit_distribution)")
        if q in (Q_CESTAR, Q_H, Q_HA):
            if row.get("licensed") is not True or not row.get("license_ref"):
                problems.append(f"{tag}: quantity {q} requires licensed=True "
                                "with a nonempty license_ref (RULE item 2: "
                                "no validation, no license — report as "
                                f"{Q_RAW})")
        if q in (Q_H, Q_HA):
            if row["not_an_entropy"] is not False:
                problems.append(f"{tag}: {q} declares an entropy; "
                                "not_an_entropy must be False")
            bound = row.get("log2_cardinality_bits")
            if bound is None or bound == "":
                problems.append(f"{tag}: {q} row must carry "
                                "log2_cardinality_bits (the entropy bound)")
            elif float(row["value_bits"]) > float(bound):
                problems.append(f"{tag}: declared {q} = "
                                f"{float(row['value_bits']):.4f} bits exceeds "
                                f"log2|Y| = {float(bound):.4f} — the §6.5 "
                                "tell; this number is definitionally not an "
                                "entropy and has been conflated")
        if q in (Q_CESTAR, Q_RAW, Q_DELTA_TRANSFER):
            if row["not_an_entropy"] is not True:
                problems.append(f"{tag}: {q} is a CE-side quantity, not an "
                                "entropy of (Y,Z); not_an_entropy must be "
                                "True")
        if q in (Q_DELTA_INFO, Q_DELTA_TRANSFER):
            if not row.get("definition"):
                problems.append(f"{tag}: derived quantity {q} must carry a "
                                "definition naming its operands")
    return problems


def check_quantity_declarations(rows: Sequence[dict]) -> None:
    """Raise QuantityDeclarationError unless `rows` is declaration-complete."""
    problems = declaration_problems(rows)
    if problems:
        raise QuantityDeclarationError(
            "results table is inadmissible under MLR_FORMAL_DEFINITIONS.md "
            "§2-bis RULE / FALSIFIABILITY_CONDITIONS.md §0-bis criterion 5 "
            "(§6.5 QUANTITY CONFLATION audit):\n  - "
            + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# Serialization (house schema; REFUSES undeclared tables)
# ---------------------------------------------------------------------------
def _ordered_fieldnames(rows: Sequence[dict]) -> list[str]:
    context, canonical = [], []
    for row in rows:
        for k in row:
            if k in CANONICAL_FIELDS:
                if k not in canonical:
                    canonical.append(k)
            elif k not in context:
                context.append(k)
    canonical.sort(key=CANONICAL_FIELDS.index)
    return context + canonical


def write_results_csv(rows: Sequence[dict], path: str) -> None:
    """Write declaration-complete rows as CSV (track3_gate2 conventions:
    snake_case columns, *_bits value suffix, one row per measurement, full
    float precision, context keys first). REFUSES undeclared tables."""
    check_quantity_declarations(rows)
    fieldnames = _ordered_fieldnames(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_results_json(rows: Sequence[dict], path: str,
                       meta: Optional[dict] = None) -> None:
    """Write declaration-complete rows + environment metadata as JSON
    (mirrors results_summary.json's `_env` convention). REFUSES undeclared
    tables."""
    check_quantity_declarations(rows)
    payload = {
        "schema": SCHEMA_VERSION,
        "rows": list(rows),
        "_env": {
            "fourq_version": __version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    if meta:
        payload["_meta"] = meta
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
