# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.23", "pyarrow>=15"]
# ///
"""More Latin-script languages spanning OOD-distance from the Pile, for the Y3b recovery-vs-OOD
curve. Latin script => bits/byte comparable to English (no multi-byte artifact)."""
import hashlib, json
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

CAP = 16 * 1024 * 1024
OUT = Path("/vault/datasets/text")
# (code, name) — varying Pile representation, all Latin script
LANGS = [("id", "indonesian"), ("vi", "vietnamese"), ("fi", "finnish"),
         ("cy", "welsh"), ("yo", "yoruba")]

for code, nm in LANGS:
    d = OUT / f"ood_{code}_wikipedia"; d.mkdir(parents=True, exist_ok=True)
    if (d / "data.txt").exists():
        print(f"{nm}: exists"); continue
    files = sorted(f for f in list_repo_files("wikimedia/wikipedia", repo_type="dataset")
                   if f"20231101.{code}/" in f and f.endswith(".parquet"))
    if not files:
        print(f"{nm} ({code}): NO FILES"); continue
    h = hashlib.sha256(); n = 0
    with open(d / "data.txt", "wb") as f:
        for fn in files:
            if n >= CAP: break
            loc = hf_hub_download("wikimedia/wikipedia", fn, repo_type="dataset")
            for ch in pq.read_table(loc, columns=["text"]).column("text").to_pylist():
                b = (ch + "\n").encode("utf-8"); f.write(b); h.update(b); n += len(b)
                if n >= CAP: break
    (d / "MANIFEST.json").write_text(json.dumps({"sha256": h.hexdigest(), "bytes": n, "lang": code}, indent=2))
    print(f"{nm} ({code}): {n} bytes sha256={h.hexdigest()[:12]}", flush=True)
print("OOD_LATIN_OK")
