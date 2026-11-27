"""Resumable pinned, shard-aware download for the optional stronger learning model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from questionforge.learning.quality import QUALITY_MODEL
from scripts.download_models import sha256

SMALL_FILES = {"config.json", "generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "model.safetensors.index.json", "README.md", "LICENSE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = QUALITY_MODEL
    directory = root / "models" / model.model_id.replace("/", "--")
    directory.mkdir(parents=True, exist_ok=True)
    metadata_file = directory / "hub_metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    else:
        url = f"https://huggingface.co/api/models/{model.model_id}/revision/{model.revision}?blobs=true"
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    assert metadata["sha"] == model.revision
    prefix = f"https://huggingface.co/{model.model_id}/resolve/{model.revision}/"
    entries = [e for e in metadata["siblings"] if e["rfilename"] in SMALL_FILES]
    for entry in entries:
        target = directory / entry["rfilename"]
        if not target.exists():
            with urllib.request.urlopen(prefix + entry["rfilename"] + "?download=true", timeout=120) as response:
                target.write_bytes(response.read())
        content = target.read_bytes()
        if entry.get("lfs"):
            assert hashlib.sha256(content).hexdigest() == entry["lfs"]["sha256"], entry["rfilename"]
        else:
            actual_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            assert actual_blob == entry["blobId"], entry["rfilename"]
    shards = [e for e in metadata["siblings"] if e["rfilename"].endswith(".safetensors")]
    block = 2 * 1024 * 1024
    jobs, directories, finished = [], {}, {}
    total = sum(e["size"] for e in shards)
    for entry in shards:
        name = entry["rfilename"]
        target = directory / name
        if target.exists() and sha256(target) == entry["lfs"]["sha256"]:
            finished[name] = entry["lfs"]["sha256"]
            continue
        chunks = root / ".cache" / "quality-model-blocks" / name
        chunks.mkdir(parents=True, exist_ok=True)
        directories[name] = chunks
        for index in range((entry["size"] + block - 1) // block):
            jobs.append((entry, index))

    def fetch(job):
        entry, index = job
        name = entry["rfilename"]
        start, end = index * block, min(entry["size"], (index + 1) * block) - 1
        chunk = directories[name] / f"{index:05d}.bin"
        if chunk.exists() and chunk.stat().st_size == end - start + 1:
            return end - start + 1
        for attempt in range(12):
            try:
                url = prefix + name + f"?download=true&block=2m&offset={start}"
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                        raise ValueError("Unexpected weight range")
                    content = response.read()
                if len(content) != end - start + 1:
                    raise ValueError("Incomplete weight range")
                chunk.write_bytes(content)
                return len(content)
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(attempt + 1, 4))

    print(f"Downloading {model.model_id}: {total / 1e9:.2f} GB; {len(jobs)} resumable blocks", flush=True)
    downloaded = sum(e["size"] for e in shards if e["rfilename"] in finished)
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            downloaded += future.result()
            if time.monotonic() - last_report >= 40:
                print(f"Quality model: {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
                last_report = time.monotonic()
    for entry in shards:
        name = entry["rfilename"]
        if name in finished:
            continue
        target = directory / name
        pending = target.with_suffix(".safetensors.partial")
        digest = hashlib.sha256()
        with pending.open("wb") as stream:
            for index in range((entry["size"] + block - 1) // block):
                content = (directories[name] / f"{index:05d}.bin").read_bytes()
                stream.write(content)
                digest.update(content)
        assert pending.stat().st_size == entry["size"] and digest.hexdigest() == entry["lfs"]["sha256"], name
        pending.replace(target)
        finished[name] = digest.hexdigest()
        print("SHA256 verified:", name, flush=True)
    receipt = {"model": model.model_id, "revision": model.revision, "weight_sha256": finished,
               "all_weights_verified": True}
    (directory / "download_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("Stronger learning model ready; every shard verified.", flush=True)


if __name__ == "__main__":
    main()
