
"""
Run the 3 pipeline notebooks sequentially to create the raw files needed by datasets.SP100Stocks.

Usage (from repo root):
    python build_pipeline.py

This is equivalent to:
    jupyter nbconvert --to notebook --execute <nb> --output <nb>.executed.ipynb
"""
import sys
from pathlib import Path
import subprocess

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "datasets").exists() and (p / "notebooks").exists():
            return p
    return start

REPO_ROOT = find_repo_root(Path.cwd())
NB_DIR = REPO_ROOT / "notebooks"

def find_notebook(patterns):
    cands = list(NB_DIR.glob("*.ipynb"))
    for pref in patterns:
        for p in cands:
            if p.name.startswith(pref):
                return p
    for kw in patterns:
        kw = kw.lower()
        for p in cands:
            if kw in p.name.lower():
                return p
    return None

def run_nb(nb_path: Path) -> int:
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=0",
        str(nb_path),
        "--output", nb_path.name.replace(".ipynb", ".executed.ipynb"),
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode

def main():
    nb1 = find_notebook(["1-", "data_collection", "preprocessing"])
    nb2 = find_notebook(["2-", "graph_creation", "graph"])
    nb3 = find_notebook(["3-", "torch_geometric_dataset", "dataset"])

    if nb1 is None or nb2 is None or nb3 is None:
        print("ERROR: missing notebooks under notebooks/. Expected 1-,2-,3-*.ipynb.")
        sys.exit(2)

    for nb in [nb1, nb2, nb3]:
        rc = run_nb(nb)
        if rc != 0:
            print(f"ERROR: notebook failed: {nb.name} (returncode={rc})")
            sys.exit(rc)

    print("DONE. Raw files should exist under data/SP100/raw/.")
    sys.exit(0)

if __name__ == "__main__":
    main()
