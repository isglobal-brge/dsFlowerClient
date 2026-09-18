#!/usr/bin/env python3
"""Execute the frozen public matrix, retaining failed cells and their logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from assemble_evidence import load_replicate, planned_cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    tools = Path(__file__).resolve().parent
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    logs = Path("/workspace/logs")
    gates = json.loads(Path(os.environ["F_SEG_GATES_JSON"]).read_text())
    if any(gates.get(f"segmentation_6_1_{i}") is not True for i in range(1, 8)):
        raise ValueError("all seven mechanism gates must pass before matrix execution")
    pending, results = [], []
    for cell in planned_cells():
        dataset, variant, epsilon, seed = cell
        work = runs / f"{dataset}-{variant}-eps{epsilon}-seed{seed}"
        if work.exists():
            # Resume completed cells only after checking their actual artifacts.
            load_replicate(work, *cell)
            results.append({"dataset": dataset, "variant": variant, "epsilon": epsilon,
                            "seed": seed, "status": "executed", "previously_completed": True})
        else:
            pending.append(cell)

    def execute(cell):
        dataset, variant, epsilon, seed = cell
        name = f"{dataset}-{variant}-eps{epsilon}-seed{seed}"
        work = runs / name
        if work.exists():
            raise ValueError("Refusing to overwrite an existing cell: " + name)
        work.mkdir()
        env = dict(os.environ, F_SEG_VARIANT=variant)
        env.pop("F_SEG_SYNTHETIC", None)
        prepared = root / "prepared" / dataset
        split = work / "effective-split.json"
        commands = [
            [str(tools / "run_federated.sh"), str(prepared),
             str(prepared / f"split-{seed}.json"), str(epsilon), str(seed), str(work)],
            [sys.executable, str(tools / "score_public.py"), "--features", str(root / "features" / dataset),
             "--split", str(split), "--probabilities", str(work / "public-probabilities.csv"),
             "--artifact", str(work / "artifact" / "model.pt"), "--out", str(work / "channel-b.json")],
            [sys.executable, str(tools / "central_twins.py"), "--features", str(root / "features" / dataset),
             "--split", str(split), "--capture", str(work / "public-capture"),
             "--gates", env["F_SEG_GATES_JSON"], "--epsilon", str(epsilon), "--out", str(work / "twins")],
        ]
        started = time.monotonic()
        result = {"dataset": dataset, "variant": variant, "epsilon": epsilon, "seed": seed,
                  "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
        print(json.dumps(result), flush=True)
        with (logs / ("segmentation-" + name + ".log")).open("w") as log:
            for phase, command in zip(("federation", "channel_b", "twins"), commands):
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    result.update(status="failed", phase=phase, exit_code=process.returncode)
                    break
            else:
                result["status"] = "executed"
        result["elapsed_s"] = time.monotonic() - started
        (work / "execution-status.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, cell) for cell in pending]
        for future in as_completed(futures):
            results.append(future.result())
            (root / "matrix-status.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(r["status"] != "executed" for r in results):
        raise SystemExit("Matrix includes failures; retain all evidence and inspect logs.")


if __name__ == "__main__":
    main()
