#!/usr/bin/env python3
"""Foreground execution; stop at the first failure and preserve its directory."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(root):
    tools = Path(__file__).resolve().parent
    protocol = json.loads((tools / "protocol.json").read_text())
    env = dict(os.environ, R_LIBS_USER=str(root / "Rlib"),
               DSFLOWER_VENV_ROOT=str(root / "venvs"), DSFLOWER_CLIENT_VENV_ROOT=str(root / "client"),
               PYTHONPATH=str(root / "src/dsFlowerClient/inst/flower_app"),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    parent = Path("/tmp/dsflower-sequence-parent")
    parent.mkdir(mode=0o700, exist_ok=True)
    env["DSFLOWER_NODE_SECRET_FILE"] = str(parent / "secret")
    env["DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET"] = "1"
    python = str(root / "venvs/pytorch-gpu/bin/python")
    for epsilon in protocol["epsilon_order"]:
        for seed in protocol["seeds"]:
            name = f"pytorch_lstm-eps{epsilon}-seed{seed}"
            run = root / "runs" / name
            started = time.monotonic()
            for phase, command in (
                ("federation", ["Rscript", str(tools / "run_federated.R"), str(root), str(epsilon), str(seed), "pytorch_lstm"]),
                ("central", [python, str(tools / "central_twins.py"), "--root", str(root), "--run", str(run)]),
            ):
                print(datetime.now(timezone.utc).isoformat(), name, phase, flush=True)
                log = root / "logs" / f"{name}-{phase}.log"
                with log.open("x") as stream:
                    result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1500)
                if result.returncode:
                    print(f"FAILED {name} {phase}; see {log}", flush=True)
                    return result.returncode
            (run / "execution-status.json").write_text(json.dumps({"status": "trained_unscored",
                "seed": seed, "epsilon": epsilon, "elapsed_s": time.monotonic() - started,
                "completed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    sys.exit(main(parser.parse_args().root))
