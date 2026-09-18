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

from assemble_evidence import load_replicate, planned_cells, released_artifact


def write_status(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--batch-size", type=int, choices=(16, 64), default=16)
    parser.add_argument("--cell", help="Execute one exact planned cell name; existing attempts still refuse overwrite")
    args = parser.parse_args()
    root = args.root.resolve()
    tools = Path(__file__).resolve().parent
    runs = root / ("runs-batch%d" % args.batch_size)
    runs.mkdir(exist_ok=True)
    logs = Path("/workspace/logs")
    gates = json.loads(Path(os.environ["F_SEG_GATES_JSON"]).read_text())
    if any(gates.get(f"segmentation_6_1_{i}") is not True for i in range(1, 8)):
        raise ValueError("all seven mechanism gates must pass before matrix execution")
    pending, results = [], []
    planned = planned_cells()
    if args.cell is not None:
        planned = [c for c in planned if "%s-%s-eps%d-seed%d" % c == args.cell]
        if not planned:
            parser.error("--cell must name a preregistered cell")
    for cell in planned:
        dataset, variant, epsilon, seed = cell
        work = runs / f"{dataset}-{variant}-eps{epsilon}-seed{seed}"
        if work.exists():
            # Resume completed cells only after checking their actual artifacts.
            load_replicate(work, *cell, batch_size=args.batch_size)
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
        env = dict(os.environ, F_SEG_VARIANT=variant, F_SEG_BATCH_SIZE=str(args.batch_size),
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
        env.pop("F_SEG_SYNTHETIC", None)
        prepared = root / "prepared" / dataset
        split = work / "effective-split.json"
        started = time.monotonic()
        result = {"dataset": dataset, "variant": variant, "epsilon": epsilon, "seed": seed,
                  "started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "phase": "federation"}
        status_path = work / "execution-status.json"
        write_status(status_path, result)
        print(json.dumps(result), flush=True)
        with (logs / ("segmentation-batch%d-" % args.batch_size + name + ".log")).open("w") as log:
            for phase in ("federation", "channel_b", "twins"):
                result["phase"] = phase
                write_status(status_path, result)
                try:
                    if phase == "federation":
                        command = [str(tools / "run_federated.sh"), str(prepared),
                            str(prepared / f"split-{seed}.json"), str(epsilon), str(seed), str(work)]
                    elif phase == "channel_b":
                        command = [sys.executable, str(tools / "score_public.py"),
                            "--features", str(root / "features" / dataset), "--split", str(split),
                            "--probabilities", str(work / "public-probabilities.csv"),
                            "--artifact", str(released_artifact(work)), "--out", str(work / "channel-b.json")]
                    else:
                        command = [sys.executable, str(tools / "central_twins.py"),
                            "--features", str(root / "features" / dataset), "--split", str(split),
                            "--capture", str(work / "public-capture"), "--gates", env["F_SEG_GATES_JSON"],
                            "--epsilon", str(epsilon), "--batch-size", str(args.batch_size), "--out", str(work / "twins")]
                    process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                except (OSError, ValueError, KeyError) as error:
                    result.update(status="failed", phase=phase, error=str(error))
                    print(str(error), file=log, flush=True)
                    break
                if process.returncode:
                    result.update(status="failed", phase=phase, exit_code=process.returncode)
                    break
            else:
                result["status"] = "executed"
        result["elapsed_s"] = time.monotonic() - started
        write_status(status_path, result)
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, cell) for cell in pending]
        for future in as_completed(futures):
            results.append(future.result())
            (root / ("matrix-status-batch%d%s.json" % (args.batch_size, "-" + args.cell if args.cell else ""))).write_text(json.dumps(results, indent=2) + "\n")
    if any(r["status"] != "executed" for r in results):
        raise SystemExit("Matrix includes failures; retain all evidence and inspect logs.")


if __name__ == "__main__":
    main()
