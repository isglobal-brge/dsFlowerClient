#!/usr/bin/env python3
"""Resume scoring only after verified completed public federation; never refit it."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from assemble_evidence import (read_json, released_artifact, sha256,
    validate_captures, validate_split_provenance, load_replicate)
from run_matrix import write_status


def verify_completed_federation(work, batch):
    previous = read_json(work / "execution-status.json")
    status = read_json(work / "federation-status.json")
    if previous.get("status") != "failed" or previous.get("phase") != "federation":
        raise ValueError("only explicit failed federation exits may resume here")
    if status.get("status") != "predicted_pending_public_metric_summary" or status.get("cleanup_ok") is not True:
        raise ValueError("completed prediction and verified cleanup required")
    if any((work / name).exists() for name in ("channel-b.json", "twins", "execution-status-before-postprocessing.json")):
        raise ValueError("postprocessing already attempted; preserve its outcome")
    key = tuple(status[name] for name in ("dataset", "variant", "epsilon", "seed"))
    split = validate_split_provenance(work, key[0], key[1], key[3], status)
    artifact = released_artifact(work, status)
    if sha256(artifact) != status["model_sha256"] or not (work / "public-probabilities.csv").is_file():
        raise ValueError("completed artifact or probabilities missing/mismatched")
    metadata = read_json(artifact.parent / "metadata.json")
    if metadata.get("status") != "success" or metadata.get("n_clients") != 3:
        raise ValueError("released model metadata is not successful three-node federation")
    capture = work / "public-capture"
    initial = read_json(capture / "public-initial.json")
    if initial["seed"] != key[3] or initial["config"]["batch-size"] != batch:
        raise ValueError("captured initialization differs from requested arm")
    validate_captures([read_json(p) for p in capture.glob("accountant-*.json")],
                      list(map(len, split["sites"])), key[2], batch_size=batch)
    return key, previous, artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(16,64), required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    work = args.work.resolve()
    key, previous, artifact = verify_completed_federation(work, args.batch_size)
    (work / "execution-status-before-postprocessing.json").write_text(json.dumps(previous, indent=2)+'\n')
    result = dict(previous, status="running", phase="channel_b", previous_execution=previous,
                  recovery="Existing successful federation artifact/predictions; no federation retraining")
    result.pop("exit_code", None)
    commands = [("channel_b", [sys.executable, str(tools / "score_public.py"),
        "--features", str(args.root / "features" / key[0]), "--split", str(work / "effective-split.json"),
        "--probabilities", str(work / "public-probabilities.csv"), "--artifact", str(artifact),
        "--out", str(work / "channel-b.json")]),
        ("twins", [sys.executable, str(tools / "central_twins.py"),
        "--features", str(args.root / "features" / key[0]), "--split", str(work / "effective-split.json"),
        "--capture", str(work / "public-capture"), "--gates", os.environ["F_SEG_GATES_JSON"],
        "--epsilon", str(key[2]), "--batch-size", str(args.batch_size), "--out", str(work / "twins")])]
    for phase, command in commands:
        result["phase"] = phase
        write_status(work / "execution-status.json", result)
        process = subprocess.run(command)
        if process.returncode:
            result.update(status="failed", exit_code=process.returncode)
            write_status(work / "execution-status.json", result)
            raise SystemExit(process.returncode)
    result["status"] = "executed"
    write_status(work / "execution-status.json", result)
    try:
        load_replicate(work, *key, batch_size=args.batch_size)
    except Exception as error:
        result.update(status="failed", phase="evidence_validation", error=str(error))
        write_status(work / "execution-status.json", result)
        raise
    print(json.dumps(result))


if __name__ == "__main__":
    main()
