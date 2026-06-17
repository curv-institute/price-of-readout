# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.23", "pyarrow>=15"]
# ///
"""Fetch corpora spanning OOD-distance from the Pile (for Y3 shift scale-confirm on Pythia).
Low-resource languages clearly under-represented in the Pile + a structured domain."""
import hashlib, json
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

CAP = 16 * 1024 * 1024
OUT = Path("/vault/datasets/text")

def flatten(repo, pattern, column, name, note):
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if pattern in f and f.endswith(".parquet"))
    assert files, f"none match {pattern} in {repo}"
    h = hashlib.sha256(); n = 0; used = []
    with open(d / "data.txt", "wb") as f:
        for fn in files:
            if n >= CAP: break
            loc = hf_hub_download(repo, fn, repo_type="dataset"); used.append(fn)
            for chunk in pq.read_table(loc, columns=[column]).column(column).to_pylist():
                b = (chunk + "\n").encode("utf-8"); f.write(b); h.update(b); n += len(b)
                if n >= CAP: break
    (d / "MANIFEST.json").write_text(json.dumps({"sha256": h.hexdigest(), "bytes": n,
        "source": f"hf:{repo} {note}", "files": used}, indent=2))
    print(f"{name}: {n} bytes sha256={h.hexdigest()[:16]}...", flush=True)

# Swahili wikipedia — low-resource, clearly OOD for Pile (English-dominant)
flatten("wikimedia/wikipedia", "20231101.sw/", "text", "ood_sw_wikipedia", "swahili wiki")
# Telugu wikipedia — Dravidian script, very OOD
flatten("wikimedia/wikipedia", "20231101.te/", "text", "ood_te_wikipedia", "telugu wiki")
print("OOD_FETCH_OK")
