"""Resumable, hash-verified downloads without importing PyTorch.

Stores only inference files in ignored models/. SHA-256 hashes are checked against
the pinned Hub commit's LFS metadata. No training files or executable model code.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from questionforge.learning.pipeline import PLANNER  # noqa: E402
from questionforge.registry import GENERATORS, VERIFIER  # noqa: E402
from questionforge.verification import MULTILINGUAL_QA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEEP = {"config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "added_tokens.json", "spiece.model", "sentencepiece.bpe.model",
        "vocab.txt", "vocab.json", "merges.txt", "README.md"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(job):
    spec, entry, directory = job
    name = entry["rfilename"]
    target = directory / name
    expected = entry.get("lfs", {}).get("sha256")
    if target.exists() and (not expected or sha256(target) == expected):
        return
    partial = target.with_name(target.name + ".partial")
    url = f"https://huggingface.co/{spec.model_id}/resolve/{spec.revision}/{name}?download=true"
    for attempt in range(1000):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            expected_size = entry.get("size", 0)
            end = min(offset + 8 * 1024 * 1024 - 1, expected_size - 1) if expected_size else ""
            headers = {"Range": f"bytes={offset}-{end}"} if expected_size else {}
            request = urllib.request.Request(url + f"&offset={offset}", headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                append = response.status == 206 and offset > 0
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise IOError("Server returned a different range than requested")
                with partial.open("ab" if append else "wb") as stream:
                    last_report = stream.tell()
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        if stream.tell() - last_report >= 50 * 1024 * 1024:
                            print(f"Progress {spec.model_id}/{name}: {stream.tell() / 1e6:.0f} MB", flush=True)
                            last_report = stream.tell()
            if partial.stat().st_size < entry.get("size", 0):
                if partial.stat().st_size <= offset:
                    raise IOError(f"Empty response at offset {offset}")
                print(f"Partial {spec.model_id}/{name}: {partial.stat().st_size / 1e6:.0f}/{expected_size / 1e6:.0f} MB", flush=True)
                continue
            if expected and sha256(partial) != expected:
                partial.unlink()
                raise ValueError(f"SHA256 mismatch for {name}")
            partial.replace(target)
            print(f"Downloaded {spec.model_id}/{name} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
            return
        except Exception as exc:
            print(f"Retry {attempt + 1}: {spec.model_id}/{name}: {exc}", flush=True)
            if attempt == 999:
                raise
            time.sleep(min(2 ** attempt, 30))


def main():
    jobs = []
    specs = list(GENERATORS.values()) + [VERIFIER]
    requested = sys.argv[1:]
    if requested:
        choices = {**GENERATORS, "verifier": VERIFIER, "multilingual": MULTILINGUAL_QA, "planner": PLANNER}
        unknown = set(requested) - choices.keys()
        if unknown:
            raise ValueError(f"Unknown model selections: {sorted(unknown)}; choose {sorted(choices)}")
        specs = [choices[x] for x in requested]
    for spec in specs:
        url = f"https://huggingface.co/api/models/{spec.model_id}/revision/{spec.revision}?blobs=true"
        directory = ROOT / "models" / spec.model_id.replace("/", "--")
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / "hub_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if metadata.get("sha") != spec.revision:
            with urllib.request.urlopen(url, timeout=120) as response:
                metadata = json.load(response)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        names = {entry["rfilename"] for entry in metadata["siblings"]}
        weight = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
        files = [entry for entry in metadata["siblings"] if entry["rfilename"] in KEEP | {weight}]
        print(f"Preparing {spec.model_id}: {sum(e.get('size', 0) for e in files) / 1e6:.0f} MB", flush=True)
        jobs.extend((spec, entry, directory) for entry in files)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch_file, jobs))
    print("All requested checkpoints are ready.", flush=True)


if __name__ == "__main__":
    main()
