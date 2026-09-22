#!/usr/bin/env python3
"""Run the registered trees cells once, sequentially in the foreground."""
import json
from pathlib import Path
import subprocess

ROOT = Path("/workspace/cells")
OUTPUT = ROOT / "dsFlowerClient/inst/extdata/campaign/trees"
TOOL = ROOT / "dsFlowerClient/tools/campaign/trees/run_cell.R"


def run(dataset, epsilon):
    subprocess.run([
        "Rscript", str(TOOL), "--contract", "random_forest", "--dataset", dataset,
        "--epsilon", str(epsilon), "--replicates", "3", "--rounds", "1",
        "--sites", "3", "--root", str(ROOT), "--out", str(OUTPUT)],
        check=True, timeout=1800)


for dataset in ("breast", "cdc9k"):
    for epsilon in (1, 8, 4):
        run(dataset, epsilon)

endpoint = [json.loads((OUTPUT / f"pilot_{dataset}_random_forest_eps8.json").read_text())
            for dataset in ("breast", "cdc9k")]
if any(not cell["diagnostic"]["pass"] for cell in endpoint):
    print("Primary epsilon-8 diagnostic failed; executing the single registered cdc45k alternative.", flush=True)
    run("cdc45k", 8)
subprocess.run(["python3", str(TOOL.with_name("summarize.py")), str(OUTPUT)], check=True)
