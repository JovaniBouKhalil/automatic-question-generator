"""Create the requested ZIP while excluding environments, source archives and large weights."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    required = ["README.md", "presentation/QuestionForge.pptx", "results/benchmark_test/summary.json",
                "results/novel_test/summary.json", "results/lora_test/summary.json", "HANDOFF.md"]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError("Finish required artifacts before packaging: " + ", ".join(missing))
    folders = ["questionforge", "scripts", "tests", "docs", "notebooks", "results", "presentation",
               "artifacts/english_lora", "artifacts/seed_study", "artifacts/article_isolated_lora",
               "artifacts/expanded",
               "artifacts/science_lora",
               "data/novel", "data/authoring", ".streamlit", ".github"]
    files = [ROOT / name for name in ["README.md", "HANDOFF.md", "pyproject.toml",
                                     "requirements-lock.txt", "app.py", "run_app.bat", ".gitignore", ".gitattributes"] if (ROOT / name).is_file()]
    for folder in folders:
        files.extend(path for path in (ROOT / folder).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and not path.name.startswith("~$")
                     and not path.name.endswith((".pyc", ".log")))
    files = sorted(set(files))
    manifest = []
    for path in files:
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError(f"Unexpected large submission file: {path.relative_to(ROOT)}")
        manifest.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    directory = ROOT / "submission"
    directory.mkdir(exist_ok=True)
    target = directory / "QuestionForge-submission.zip"
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "QuestionForge/" + path.relative_to(ROOT).as_posix())
        archive.writestr("QuestionForge/SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2))
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise IOError("ZIP integrity check failed")
    print(f"Packaged {len(files)} files: {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
