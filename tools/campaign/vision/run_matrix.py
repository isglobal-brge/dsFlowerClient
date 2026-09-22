#!/usr/bin/env python3
"""Foreground, fail-fast training grid. This driver never opens the test split."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main(root, epsilons):
    tools = Path(__file__).resolve().parent
    protocol = json.loads((tools / "protocol.json").read_text())
    for epsilon in epsilons:
        assert epsilon in protocol["epsilon_order"]
        for seed in protocol["seeds"]:
            started = time.monotonic()
            run = root / "runs" / f"pytorch_resnet18-eps{epsilon}-seed{seed}"
            status = run / "federation-status.json"
            print(datetime.now(timezone.utc).isoformat(), "TRAIN", epsilon, seed, flush=True)
            if status.exists():
                assert json.loads(status.read_text())["status"] == "trained_unscored", "Retain and diagnose failed attempt"
            else:
                subprocess.run(["Rscript", str(tools / "run_federated.R"), str(root), str(epsilon), str(seed)],
                               check=True, timeout=1800)
            if not (run / "twins-status.json").exists():
                subprocess.run([sys.executable, str(tools / "central_twins.py"),
                                "--root", str(root), "--run", str(run)], check=True, timeout=1800)
            timing = run / "execution-timing.json"
            if not timing.exists():
                timing.write_text(json.dumps(dict(elapsed_s=time.monotonic()-started,
                    scope="This foreground driver invocation: federation if needed, feature verification, and twins if needed; central reused across epsilon for each seed."), indent=2) + "\n")
            print(datetime.now(timezone.utc).isoformat(), "TRAINED_UNSCORED", epsilon, seed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epsilons", type=int, nargs="+", default=[1, 8, 4])
    args = parser.parse_args()
    main(args.root, args.epsilons)
