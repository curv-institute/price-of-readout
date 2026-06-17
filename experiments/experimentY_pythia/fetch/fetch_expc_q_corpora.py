# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.23", "pyarrow>=15"]
# ///
"""Fetch expC shift corpora, sha256-pinned, to /vault/datasets/text/.

Q1 (structural shift):  source code — codeparrot/codeparrot-clean-valid, Python,
    files concatenated in dataset order until the byte cap. (The flatten() call and
    the committed MANIFEST/sha256 are authoritative: codeparrot-clean-valid.)
Q2 (byte-distribution shift): non-Latin text — wikimedia/wikipedia 20231101.ja,
    articles concatenated in dataset order until the byte cap.

Acquisition only: nothing here reads or evaluates these against any model or
tokenizer (expC clean-arm wall). Caps keep eval+refit budgets covered (~64MB).
"""
import hashlib, json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

CAP = 64 * 1024 * 1024
OUT = Path("/vault/datasets/text")


def iter_records(local: str, column: str):
    if local.endswith(".json.gz"):
        import gzip
        import json as _json
        with gzip.open(local, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield _json.loads(line)[column]
        return
    if local.endswith(".parquet"):
        tbl = pq.read_table(local, columns=[column])
        yield from tbl.column(column).to_pylist()
    else:  # .json — array or JSON-lines
        import json as _json
        with open(local, "r", encoding="utf-8") as fh:
            first = fh.read(1)
            fh.seek(0)
            if first == "[":
                for rec in _json.load(fh):
                    yield rec[column]
            else:
                for line in fh:
                    if line.strip():
                        yield _json.loads(line)[column]


def flatten(repo: str, pattern: str, column: str, out_dir: Path, source_note: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                   if pattern in f and (f.endswith(".parquet") or f.endswith(".json") or f.endswith(".json.gz")))
    assert files, f"no data files matching {pattern} in {repo}"
    h = hashlib.sha256()
    n = 0
    used = []
    out_file = out_dir / "data.txt"
    with open(out_file, "wb") as f:
        for fname in files:
            if n >= CAP:
                break
            local = hf_hub_download(repo, fname, repo_type="dataset")
            used.append(fname)
            for chunk in iter_records(local, column):
                b = (chunk + "\n").encode("utf-8")
                f.write(b)
                h.update(b)
                n += len(b)
                if n >= CAP:
                    break
    manifest = {
        "file": str(out_file), "sha256": h.hexdigest(), "bytes": n,
        "source": f"hf:{repo} {source_note}", "parquet_files": used,
        "cap_bytes": CAP, "order": "dataset order, truncated at cap",
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"{out_dir.name}: {n} bytes sha256={h.hexdigest()[:16]}... ({len(used)} parquet files)", flush=True)


flatten("codeparrot/codeparrot-clean-valid", "file-", "content",
        OUT / "expc_q1_code_python", "codeparrot-clean-valid python content")
flatten("wikimedia/wikipedia", "20231101.ja/", "text",
        OUT / "expc_q2_ja_wikipedia", "wikipedia 20231101.ja text")
print("Q_FETCH_OK")
