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
"""Y1: scale-confirm the PoR damage taxonomy on a REAL pretrained LM (Pythia-410m).

Two PoR decompositions on an off-the-shelf model, both with head-only refit (freeze
body, train lm_head + final LayerNorm) — the interface-isolating refit used in expD/F:

  mode=quant  : fake-quant the transformer-block weights at b bits (per-output-channel
                symmetric), frozen-head vs head-refit CE on ID. Tests whether the
                information-borne sub-4-bit frontier holds on a real pretrained model.
  mode=shift  : frozen LM on a domain shift (code / Japanese) vs head-refit-on-shift.
                Tests whether transfer collapse is interface-borne (refit recovers) at
                real-model scale.

ID = WikiText-103 test; shifts = expC code (q1) / Japanese (q2) raw text, tokenized
with the model's own tokenizer. CE reported in bits/byte (exact: sum token NLL / raw
bytes). §2-bis: frozen = CE(r_P^quant/shift); head-refit = readout-budget optimum.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT = {"id": "/vault/datasets/text/wikitext103_raw/test.txt",
        "q1": "/vault/datasets/text/expc_q1_code_python/data.txt",
        "q2": "/vault/datasets/text/expc_q2_ja_wikipedia/data.txt",
        "sw": "/vault/datasets/text/ood_sw_wikipedia/data.txt",      # Swahili — OOD
        "te": "/vault/datasets/text/ood_te_wikipedia/data.txt"}      # Telugu — very OOD


def per_out_quant_(W, bits):
    if bits >= 16:
        return
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    W.copy_(torch.round(W / s).clamp(-qmax, qmax) * s)


def quant_body_(model, bits):
    n = 0
    for name, p in model.named_parameters():
        # transformer block linear weights: 2-D, in a layer block, not embeddings/head/ln
        if p.dim() == 2 and ("layers." in name or "layer." in name or "h." in name) \
           and ("embed" not in name) and ("ln" not in name) and (p.shape[0] != model.config.vocab_size):
            per_out_quant_(p.data, bits); n += 1
    return n


def load_tokens(tok, path, max_bytes, ctx, offset=0):
    raw = Path(path).read_bytes()[offset:offset + max_bytes]
    text = raw.decode("utf-8", errors="ignore")
    ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    ids = np.array(ids, dtype=np.int64)
    # byte length per token for bits/byte accounting
    blens = np.array([len(tok.decode([int(i)]).encode("utf-8")) for i in np.unique(ids)])
    bytelen = np.zeros(int(ids.max()) + 1, dtype=np.int64)
    for i in np.unique(ids):
        bytelen[i] = len(tok.decode([int(i)]).encode("utf-8"))
    return ids, bytelen


@torch.no_grad()
def eval_bpb(model, ids, bytelen, ctx, dev, max_windows=400):
    model.eval()
    starts = list(range(0, len(ids) - (ctx + 1), ctx))[:max_windows]
    tot_bits, tot_bytes = 0.0, 0
    for b0 in range(0, len(starts), 8):
        ss = starts[b0:b0 + 8]
        if not ss:
            break
        batch = np.stack([ids[s:s + ctx + 1] for s in ss])
        x = torch.from_numpy(batch[:, :-1]).to(dev); y = torch.from_numpy(batch[:, 1:]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).logits
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                             reduction="sum")
        tot_bits += float(ce.item()) / math.log(2)
        for s in ss:
            tot_bytes += int(bytelen[ids[s + 1:s + 1 + ctx]].sum())
    return tot_bits / tot_bytes


def head_refit(model, ids, ctx, dev, budget_tok, lr=1e-4, seed=0):
    head_names = ("embed_out", "lm_head", "final_layer_norm", "ln_f")
    for n, p in model.named_parameters():
        p.requires_grad_(any(h in n for h in head_names))
    ps = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr)
    rng = np.random.default_rng(seed); ms = len(ids) - (ctx + 1); seen = 0
    model.train()
    while seen < budget_tok:
        st = rng.integers(0, ms, 8); batch = np.stack([ids[s:s + ctx + 1] for s in st])
        x = torch.from_numpy(batch[:, :-1]).to(dev); y = torch.from_numpy(batch[:, 1:]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        seen += 8 * ctx


def fresh(model_name, dev):
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--mode", required=True, choices=["quant", "shift"])
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--split", default="id", choices=["id", "q1", "q2", "sw", "te"])
    ap.add_argument("--textfile", default=None, help="shift mode: override corpus path for this split")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--eval-bytes", type=int, default=2_000_000)
    ap.add_argument("--refit-bytes", type=int, default=8_000_000)
    ap.add_argument("--refit-tok", type=int, default=400_000)
    ap.add_argument("--refit-offset", type=int, default=0,
                    help="byte offset into the refit corpus (for an eval-disjoint refit)")
    ap.add_argument("--refit-textfile", default=None,
                    help="refit corpus path override (e.g. wikitext train, disjoint from the test-split eval)")
    ap.add_argument("--refit-seed", type=int, default=0,
                    help="RNG seed for refit-window sampling (default 0 reproduces the main run)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()

    if args.mode == "quant":
        refit_path = args.refit_textfile or TEXT["id"]
        ids, bl = load_tokens(tok, TEXT["id"], args.eval_bytes, args.ctx)
        tr, _ = load_tokens(tok, refit_path, args.refit_bytes, args.ctx, args.refit_offset)
        m = fresh(args.model, dev); quant_body_(m, args.bits)
        frozen = eval_bpb(m, ids, bl, args.ctx, dev)
        m2 = fresh(args.model, dev); quant_body_(m2, args.bits)
        head_refit(m2, tr, args.ctx, dev, args.refit_tok, seed=args.refit_seed)
        refit = eval_bpb(m2, ids, bl, args.ctx, dev)
        res = {"mode": "quant", "bits": args.bits, "frozen_id": frozen, "refit_id": refit}
    else:
        path = args.textfile or TEXT[args.split]
        refit_path = args.refit_textfile or path
        ids, bl = load_tokens(tok, path, args.eval_bytes, args.ctx)
        tr, _ = load_tokens(tok, refit_path, args.refit_bytes, args.ctx, args.refit_offset)
        m = fresh(args.model, dev)
        frozen = eval_bpb(m, ids, bl, args.ctx, dev)
        m2 = fresh(args.model, dev)
        head_refit(m2, tr, args.ctx, dev, args.refit_tok, seed=args.refit_seed)
        refit = eval_bpb(m2, ids, bl, args.ctx, dev)
        res = {"mode": "shift", "split": args.split, "frozen": frozen, "refit": refit}
    res["refit_source"] = {"textfile": refit_path, "offset": args.refit_offset,
                           "bytes": args.refit_bytes, "seed": args.refit_seed}
    res["wall_s"] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(res, indent=2))
    print("Y1_OK " + json.dumps(res))


if __name__ == "__main__":
    main()
