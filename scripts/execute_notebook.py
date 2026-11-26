"""Execute the demonstration notebook using this environment without global kernel registration."""

import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "notebooks/QuestionForge.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    # Use this interpreter directly; no user-level Jupyter configuration is modified.
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, timeout=1200, km=manager,
                            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.write(notebook, path)
    print(f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells without errors: {path}")


if __name__ == "__main__":
    main()
