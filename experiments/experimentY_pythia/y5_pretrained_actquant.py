# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "transformers>=4.40", "safetensors", "huggingface_hub"]
#
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# ///
"""Y5: activation-quantization PoR decomposition on Pythia-410m (real-model confirm of X2).

Per-tensor dynamic symmetric fake-quant of the INPUT ACTIVATIONS to every transformer-block
nn.Linear (weights stay bf16), via forward pre-hooks. frozen-head vs head-refit on ID. Tests
whether the activation-quant 'mixed/interface-borne' finding (X2) holds on a real pretrained
model. Reuses y1_pretrained_por harness."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import y1_pretrained_por as Y1  # noqa: E402

_A = {"v": 16}


def aq(x):
    b = _A["v"]
    if b >= 16:
        return x
    qmax = 2 ** (b - 1) - 1
    amax = x.detach().abs().amax().clamp(min=1e-8)
    s = amax / qmax
    return torch.round(x / s).clamp(-qmax, qmax) * s


def hook(mod, inp):
    return (aq(inp[0]),) + inp[1:]


def install(model):
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.shape[0] != model.config.vocab_size \
           and ("layers." in name or "layer." in name or "h." in name):
            m.register_forward_pre_hook(hook); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--abits", type=int, required=True)
    ap.add_argument("--idfile", default=None,
                    help="override the ID (WikiText-103 test) corpus path (else Y1.TEXT['id'], i.e. $DATA_ROOT)")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--eval-bytes", type=int, default=2_000_000)
    ap.add_argument("--refit-bytes", type=int, default=8_000_000)
    ap.add_argument("--refit-tok", type=int, default=400_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    idpath = args.idfile or Y1.TEXT["id"]
    ids, bl = Y1.load_tokens(tok, idpath, args.eval_bytes, args.ctx)
    tr, _ = Y1.load_tokens(tok, idpath, args.refit_bytes, args.ctx)
    import time; t0 = time.time()
    _A["v"] = args.abits
    m = Y1.fresh(args.model, dev); install(m)
    frozen = Y1.eval_bpb(m, ids, bl, args.ctx, dev)
    m2 = Y1.fresh(args.model, dev); install(m2)
    Y1.head_refit(m2, tr, args.ctx, dev, args.refit_tok)
    refit = Y1.eval_bpb(m2, ids, bl, args.ctx, dev)
    res = {"abits": args.abits, "frozen_id": frozen, "refit_id": refit, "wall_s": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(res, indent=2))
    print("Y5_OK " + json.dumps(res))


if __name__ == "__main__":
    main()
