# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.23", "pyarrow>=15"]
# ///
"""Fetch WikiText-103-raw-v1 (Salesforce/wikitext) and flatten each split to a
single raw UTF-8 text file under $DATA_ROOT/wikitext103_raw/ (default
/vault/datasets/text). Records sha256 + byte counts in MANIFEST.json. Idempotent:
skips existing outputs.

Output root: --out-root PATH > $DATA_ROOT env var > /vault/datasets/text default.
Point the runner ($DATA_ROOT) at the same root afterwards (see README.md)."""
import argparse, hashlib, json, os, sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

_ap = argparse.ArgumentParser(description="Fetch WikiText-103-raw-v1 splits.")
_ap.add_argument("--out-root", default=None,
                 help="output root (default: $DATA_ROOT env var, else /vault/datasets/text)")
_args = _ap.parse_args()
ROOT = Path(_args.out_root or os.environ.get("DATA_ROOT", "/vault/datasets/text"))
OUT = ROOT / "wikitext103_raw"
OUT.mkdir(parents=True, exist_ok=True)
print(f"output root: {ROOT}", flush=True)
manifest_path = OUT / "MANIFEST.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

local = snapshot_download(
    repo_id="Salesforce/wikitext",
    repo_type="dataset",
    allow_patterns=["wikitext-103-raw-v1/*"],
)
src = Path(local) / "wikitext-103-raw-v1"
print(f"snapshot at {src}", flush=True)

for split in ["train", "validation", "test"]:
    out_file = OUT / f"{split}.txt"
    if out_file.exists() and split in manifest:
        print(f"{split}: exists, skipping", flush=True)
        continue
    parts = sorted(src.glob(f"{split}-*.parquet"))
    assert parts, f"no parquet for {split}"
    h = hashlib.sha256()
    n_bytes = 0
    with open(out_file, "wb") as f:
        for p in parts:
            tbl = pq.read_table(p, columns=["text"])
            for chunk in tbl.column("text").to_pylist():
                b = chunk.encode("utf-8")
                f.write(b)
                h.update(b)
                n_bytes += len(b)
    manifest[split] = {
        "file": str(out_file),
        "sha256": h.hexdigest(),
        "bytes": n_bytes,
        "source": "hf:Salesforce/wikitext wikitext-103-raw-v1 " + ",".join(p.name for p in parts),
    }
    print(f"{split}: {n_bytes} bytes sha256={h.hexdigest()[:16]}...", flush=True)

manifest_path.write_text(json.dumps(manifest, indent=2))
print("WIKITEXT_FETCH_OK", flush=True)
