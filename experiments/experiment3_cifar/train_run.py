# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "scikit-learn>=1.4", "torch", "torchvision"]
#
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# torchvision = { index = "pytorch-cu130" }
# ///
r"""
train_run.py — ONE Experiment-B training run (run-B grid producer).

`expB-masking-real-v1` PREREGISTRATION.md §2/§3/§4/§6. One invocation = one
(rho, seed, arm) cell: builds the cell's image sets via the PINNED
spurious_cifar.py (sha256 ab8f6e61...) using the §2 committed rng streams,
trains the §3 CNN9 recipe end-to-end, and extracts the 256-d penultimate
features Z for the four §4 evaluation sets needed by the once-only analysis:

    Z^A_ID   half-A test (5000) under the cell's ID coupling  -> P-fit pool
    Z^B_ID   half-B test (5000) under the cell's ID coupling  -> P-held-out
    Z^B_dec  half-B (5000) re-tinted under the decorrelate shift -> Q (gated)
    Z^B_anti half-B (5000) re-tinted under the anticorrelate shift -> Q (descr.)

plus the standard (untinted) CIFAR test accuracy as a §3 PIPELINE-SANITY number
(no gate, not an arm comparison), and a run-marker JSON of metadata.

THIS SCRIPT COMPUTES NO ARM COMPARISON, NO CE_Q, NO Δ, NO READOUT FIT. It only
trains one arm and dumps features + labels. All measurement / unblinding is the
separate analysis agent's once-only job (analyze_masking.py).

§2 committed rng streams (the module uses default_rng(10000 + cfg.seed)):
    train-set construction   : cfg.seed = σ
    ID test-set indicator    : cfg.seed = 1000 + σ
    decorrelate shift set     : cfg.seed = 2000 + σ
    anticorrelate shift set  : cfg.seed = 3000 + σ
The half-A / half-B split is the §4 committed
train_test_split(stratify=y, test_size=0.5, random_state=0) on the 10000 test
images, applied ONCE to the index set and reused for every tint draw, so the
P-fit pool and the eval pool are a fixed image partition.

cuDNN fix (BINDING on the GB10 nodes): self-re-exec with system cuDNN prepended
to LD_LIBRARY_PATH before importing torch.
"""
import os, sys

# --- cuDNN LD_LIBRARY_PATH self-re-exec shim (GB10 fix; BINDING) ---
_SYS_CUDNN = "/usr/lib/aarch64-linux-gnu"
if _SYS_CUDNN not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = _SYS_CUDNN + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse, json, time, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spurious_cifar as sc

HERE = os.path.dirname(os.path.abspath(__file__))

# §2 pin: the construction module sha256 (verified at startup; abort on mismatch)
SPURIOUS_SHA256 = "ab8f6e61d54be4ea13b23aee326aabccb2f272c4bfffa4431ace45b70c9050ba"

# CIFAR per-channel normalisation (§3; applied AFTER tint injection)
MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)

RHO_GRID = (0.7, 0.9, 0.95, 0.99)
SEEDS = tuple(range(10))


class CNN9(nn.Module):
    """§3 architecture, verbatim from the pilot smoke (param count 1,739,210).
    Penultimate Z = the 256-d post-GAP vector."""
    def __init__(self, n_classes=10, feat_dim=256):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            blk(3, 64), blk(64, 64), nn.MaxPool2d(2),          # 32->16
            blk(64, 128), blk(128, 128), nn.MaxPool2d(2),       # 16->8
            blk(128, 256), blk(256, 256), nn.MaxPool2d(2),      # 8->4
            blk(256, feat_dim),                                 # 4x4 feat_dim
        )
        self.feat_dim = feat_dim
        self.classifier = nn.Linear(feat_dim, n_classes)

    def penultimate(self, x):
        h = self.features(x)
        return F.adaptive_avg_pool2d(h, 1).flatten(1)  # (B, feat_dim) = Z

    def forward(self, x):
        return self.classifier(self.penultimate(x))


def normalise(X):
    return ((X - MEAN) / STD).astype(np.float32)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# --- §2 image construction via the pinned module, the committed rng streams ----
def _tinted_set(X, y, coupling, stream_seed, strength, shift=None):
    """Build a tinted image set on the given images at the given coupling, using
    cfg.seed = stream_seed (the module keys default_rng(10000 + cfg.seed)).
    shift=None -> draw_indicator (ID); else draw_indicator_shift(coupling, shift).
    Returns (X_tinted_normalised, y, s)."""
    cfg = sc.SpuriousConfig(rho=coupling, lawful_rho=coupling, tint_strength=strength,
                            arm="naive", seed=stream_seed)
    # arm='naive' with rho=coupling makes build_arm use `coupling` directly for
    # both the ID and shift draws (we pass the resolved coupling explicitly so the
    # train/ID/shift streams all key off cfg.seed exactly as §2 specifies).
    X_out, y_out, s = sc.build_arm(X, y, cfg, shift=shift)
    return normalise(X_out), y_out, s


def iterate(X, y, bs, shuffle, device, augment=False):
    n = len(X)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        idx = order[i:i + bs]
        xb = X[idx].to(device, non_blocking=True)
        yb = y[idx].to(device, non_blocking=True)
        if augment:  # §3 random crop (reflect pad 4) + h-flip, batch-level
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, dims=[3])
            pad = F.pad(xb, (4, 4, 4, 4), mode="reflect")
            ox, oy = torch.randint(0, 9, (2,))
            xb = pad[:, :, oy:oy + 32, ox:ox + 32]
        yield xb, yb


@torch.no_grad()
def evaluate(model, X, y, bs, device):
    model.eval()
    correct = 0
    for xb, yb in iterate(X, y, bs, False, device):
        correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(X)


@torch.no_grad()
def extract_features(model, Xnp, bs, device):
    """Extract penultimate Z for a normalised numpy image array (no labels needed)."""
    model.eval()
    X = torch.from_numpy(Xnp)
    feats = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device, non_blocking=True)
        feats.append(model.penultimate(xb).cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["naive", "lawful"])
    ap.add_argument("--rho", required=True, type=float)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--tint", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--outdir", default="/vault/datasets/features/expB")
    args = ap.parse_args()

    assert args.rho in RHO_GRID, f"rho {args.rho} not in committed grid {RHO_GRID}"
    assert args.seed in SEEDS, f"seed {args.seed} not in committed seed list {SEEDS}"
    assert abs(args.tint - 0.20) < 1e-12, "tint strength is pinned at 0.20 (§2 ruling Q1)"

    sigma = args.seed
    # The lawful arm's coupling is rho-independent (rho_eff = 1/K = 0.1); we tag
    # the cell by the nominal rho so the lawful replicate spread per §2 is
    # recoverable, but its constructed images are identical across rho at fixed σ.
    id_coupling = args.rho if args.arm == "naive" else 0.1

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.arm}_rho{args.rho}_seed{sigma}"
    feat_path = os.path.join(args.outdir, f"features_{tag}.npz")
    marker_path = os.path.join(args.outdir, f"run_{tag}.json")

    # --- §2 pin check ---------------------------------------------------------
    got = sha256_file(os.path.join(HERE, "spurious_cifar.py"))
    if got != SPURIOUS_SHA256:
        print(f"FATAL: spurious_cifar.py sha256 {got} != pinned {SPURIOUS_SHA256}")
        sys.exit(3)

    torch.manual_seed(sigma)
    np.random.seed(sigma)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{tag}] device={device} cudnn={torch.backends.cudnn.version()} "
          f"torch={torch.__version__} sklearn-split=random_state0")

    # --- load CIFAR; fixed §4 half-A / half-B index partition -----------------
    Xtr, ytr, Xte, yte = sc.load_cifar10()
    idx = np.arange(len(yte))
    idxA, idxB = train_test_split(idx, stratify=yte, test_size=0.5, random_state=0)
    XteA, yteA = Xte[idxA], yte[idxA]
    XteB, yteB = Xte[idxB], yte[idxB]
    print(f"[{tag}] train {Xtr.shape} | test split A={len(idxA)} B={len(idxB)} "
          f"id_coupling={id_coupling}")

    # --- §2 train set: cfg.seed = σ ------------------------------------------
    Xtr_n, ytr_a, s_tr = _tinted_set(Xtr, ytr, id_coupling, sigma, args.tint, shift=None)
    emp_tr = sc.empirical_coupling(ytr_a, s_tr)
    Xtr_t = torch.from_numpy(Xtr_n)
    ytr_t = torch.from_numpy(ytr_a)
    Xte_std_t = torch.from_numpy(normalise(Xte))  # standard untinted, sanity acc
    yte_t = torch.from_numpy(yte)

    # --- train CNN9 (§3) ------------------------------------------------------
    model = CNN9().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=(len(Xtr_t) + args.bs - 1) // args.bs)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, seen = 0.0, 0
        for xb, yb in iterate(Xtr_t, ytr_t, args.bs, True, device, augment=True):
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                print(f"[{tag}] epoch {ep}: NON-FINITE LOSS — abort"); sys.exit(2)
            loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(yb); seen += len(yb)
        if ep % 5 == 0 or ep == args.epochs - 1:
            acc = evaluate(model, Xte_std_t, yte_t, 512, device)
            print(f"[{tag}]   epoch {ep:2d}: train_loss={tot/seen:.4f} "
                  f"std_test_acc={acc:.4f}", flush=True)
    train_secs = time.time() - t0
    std_test_acc = evaluate(model, Xte_std_t, yte_t, 512, device)

    # --- §4 feature extraction for the four evaluation sets -------------------
    # Z^A_ID : half-A under the cell's ID coupling (cfg.seed = σ ... but the
    #          module re-draws on the SAME images; we build the A/B sets from the
    #          SAME ID draw used for train? No — §4 splits the 10000 TEST images;
    #          the train set is the 50000-image train split. The ID test draw uses
    #          cfg.seed = 1000 + σ on the 10000 test images, then the A/B index
    #          partition selects the fit pool vs held-out pool. Same draw -> same
    #          s for an image regardless of which half it lands in.
    tf = time.time()
    Xte_id_n, yte_id, s_id = _tinted_set(Xte, yte, id_coupling, 1000 + sigma,
                                         args.tint, shift=None)
    Xte_dec_n, _, s_dec = _tinted_set(Xte, yte, id_coupling, 2000 + sigma,
                                      args.tint, shift="decorrelate")
    Xte_anti_n, _, s_anti = _tinted_set(Xte, yte, id_coupling, 3000 + sigma,
                                        args.tint, shift="anticorrelate")

    ZA_ID = extract_features(model, Xte_id_n[idxA], 512, device)   # P-fit pool
    ZB_ID = extract_features(model, Xte_id_n[idxB], 512, device)   # P-held-out
    ZB_dec = extract_features(model, Xte_dec_n[idxB], 512, device)  # Q (gated)
    ZB_anti = extract_features(model, Xte_anti_n[idxB], 512, device)  # Q (descr.)
    extract_secs = time.time() - tf

    nonfinite = {
        "ZA_ID": int(np.sum(~np.isfinite(ZA_ID))),
        "ZB_ID": int(np.sum(~np.isfinite(ZB_ID))),
        "ZB_dec": int(np.sum(~np.isfinite(ZB_dec))),
        "ZB_anti": int(np.sum(~np.isfinite(ZB_anti))),
    }

    np.savez_compressed(
        feat_path,
        ZA_ID=ZA_ID, yA=yteA,
        ZB_ID=ZB_ID, yB=yteB,
        ZB_dec=ZB_dec, ZB_anti=ZB_anti,
        # the s indicators per set (B-half) — diagnostics for the analysis agent
        sA_id=s_id[idxA], sB_id=s_id[idxB],
        sB_dec=s_dec[idxB], sB_anti=s_anti[idxB],
        idxA=idxA, idxB=idxB,
    )

    marker = {
        "experiment": "expB-masking-real-v1",
        "cell": {"arm": args.arm, "rho": args.rho, "seed": sigma},
        "id_coupling": id_coupling, "tint_strength": args.tint,
        "rng_streams": {"train": sigma, "id_test": 1000 + sigma,
                        "decorrelate": 2000 + sigma, "anticorrelate": 3000 + sigma},
        "split": {"random_state": 0, "stratified": True, "test_size": 0.5,
                  "nA": int(len(idxA)), "nB": int(len(idxB))},
        "model": "CNN9", "n_params": int(n_params), "feat_dim": 256,
        "epochs": args.epochs, "bs": args.bs, "lr": args.lr,
        "optimizer": "SGD nesterov momentum0.9 wd5e-4 OneCycleLR max_lr0.1",
        "std_test_acc": float(std_test_acc),
        "empirical_coupling_train": emp_tr,
        "Z_shapes": {"ZA_ID": list(ZA_ID.shape), "ZB_ID": list(ZB_ID.shape),
                     "ZB_dec": list(ZB_dec.shape), "ZB_anti": list(ZB_anti.shape)},
        "nonfinite_counts": nonfinite,
        "train_secs": float(train_secs), "extract_secs": float(extract_secs),
        "spurious_sha256": got,
        "device": device, "cudnn": torch.backends.cudnn.version(),
        "torch": torch.__version__,
        "feature_file": os.path.basename(feat_path),
        "completed": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(marker_path, "w") as f:
        json.dump(marker, f, indent=2)
    print(f"[{tag}] DONE std_test_acc={std_test_acc:.4f} train={train_secs:.1f}s "
          f"extract={extract_secs:.1f}s nonfinite={nonfinite} -> {feat_path}",
          flush=True)


if __name__ == "__main__":
    main()
