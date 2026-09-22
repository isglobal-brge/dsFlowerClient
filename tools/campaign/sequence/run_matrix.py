#!/usr/bin/env python3
"""Foreground execution; stop at the first failure and preserve its directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(root, resume_unscored=False):
    if (root / "test-scoring-started.json").exists():
        raise RuntimeError("Test scoring has started; refuse any further training")
    tools = Path(__file__).resolve().parent
    protocol = json.loads((tools / "protocol.json").read_text())
    preflight = json.loads((root / "preflight.json").read_text())
    assert preflight["attempts"][0]["contract"] == "pytorch_lstm"
    assert preflight["attempts"][0]["status"] == "passed"
    env = dict(os.environ, R_LIBS_USER=str((root / "Rlib").resolve()),
               DSFLOWER_VENV_ROOT=str((root / "venvs").resolve()), DSFLOWER_CLIENT_VENV_ROOT=str((root / "client").resolve()),
               PYTHONPATH=str((root / "Rlib/dsFlowerClient/flower_app").resolve()),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    parent = Path("/tmp/dsflower-sequence-parent")
    parent.mkdir(mode=0o700, exist_ok=True)
    env["DSFLOWER_NODE_SECRET_FILE"] = str(parent / "secret")
    env["DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET"] = "1"
    python = str(Path(env["DSFLOWER_VENV_ROOT"]) / "pytorch-gpu/bin/python")
    for epsilon in protocol["epsilon_order"]:
        for seed in protocol["seeds"]:
            name = f"pytorch_lstm-eps{epsilon}-seed{seed}"
            run = root / "runs" / name
            started = time.monotonic()
            retained_elapsed = 0.0
            for phase, command in (
                ("federation", ["Rscript", str(tools / "run_federated.R"), str(root), str(epsilon), str(seed), "pytorch_lstm"]),
                ("central", [python, str(tools / "central_twins.py"), "--root", str(root), "--run", str(run)]),
            ):
                status_path = run / ("federation-status.json" if phase == "federation" else "central/status.json")
                if resume_unscored and status_path.exists():
                    status = json.loads(status_path.read_text())
                    assert status["status"] == "trained_unscored"
                    if phase == "federation":
                        assert status["cleanup_ok"]
                    model = (Path(status["output_dir"]) if phase == "federation" else run / "central") / "model.pt"
                    assert hashlib.sha256(model.read_bytes()).hexdigest() == status["model_sha256"]
                    retained_elapsed += status["elapsed_s"]
                    print("RETAIN completed unscored", name, phase, flush=True)
                    continue
                print(datetime.now(timezone.utc).isoformat(), name, phase, flush=True)
                log = root / "logs" / f"{name}-{phase}.log"
                with log.open("x") as stream:
                    result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1500)
                if result.returncode:
                    print(f"FAILED {name} {phase}; see {log}", flush=True)
                    return result.returncode
            (run / "execution-status.json").write_text(json.dumps({"status": "trained_unscored",
                "seed": seed, "epsilon": epsilon, "elapsed_s": time.monotonic() - started + retained_elapsed,
                "retained_training_elapsed_s": retained_elapsed,
                "completed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume-unscored", action="store_true",
                        help="retain hash-verified completed phases after a tooling interruption")
    args = parser.parse_args()
    sys.exit(main(args.root, args.resume_unscored))
