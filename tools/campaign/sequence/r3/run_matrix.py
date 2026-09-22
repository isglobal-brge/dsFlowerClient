#!/usr/bin/env python3
"""Run the frozen R3 federations in the foreground; retain every attempt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(root):
    tools = Path(__file__).resolve().parent
    work = root / "r3"
    marker = work / "test-scoring-started.json"
    if marker.exists():
        raise RuntimeError("R3 test scoring has started; refuse further training")
    protocol_path = tools / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert protocol["units"] == ["subject", "window"]
    assert protocol["rounds"] == 5 and len(protocol["seeds"]) == 3
    assert set(protocol["epsilon_order"]) == {1, 4, 8}
    env = dict(os.environ, R_LIBS_USER=str((root / "Rlib").resolve()),
               DSFLOWER_VENV_ROOT=str((root / "venvs").resolve()),
               DSFLOWER_CLIENT_VENV_ROOT=str((root / "client").resolve()),
               PYTHONPATH=str((root / "Rlib/dsFlowerClient/flower_app").resolve()),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    parent = Path("/tmp/dsflower-sequence-r3-parent")
    parent.mkdir(mode=0o700, exist_ok=True)
    env["DSFLOWER_NODE_SECRET_FILE"] = str(parent / "secret")
    env["DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET"] = "1"
    (work / "logs").mkdir(parents=True, exist_ok=True)
    for unit in protocol["units"]:
        for epsilon in protocol["epsilon_order"]:
            for seed in protocol["seeds"]:
                if marker.exists():
                    raise RuntimeError("R3 test scoring has started; refuse further training")
                assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == protocol_sha
                name = f"{unit}-eps{epsilon}-seed{seed}"
                run = work / "runs" / name
                if run.exists():
                    raise RuntimeError(f"Existing run directory; refuse overwrite: {run}")
                print(datetime.now(timezone.utc).isoformat(), name, "federation", flush=True)
                started = time.monotonic()
                log = work / "logs" / f"{name}-federation.log"
                with log.open("x") as stream:
                    result = subprocess.run(
                        ["Rscript", str(tools / "run_federated.R"), str(root),
                         str(epsilon), str(seed), unit], env=env,
                        stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
                if result.returncode:
                    print(f"FAILED {name}; see {log}", flush=True)
                    return result.returncode
                status = json.loads((run / "federation-status.json").read_text())
                assert status["status"] == "trained_unscored" and status["cleanup_ok"]
                assert (status["unit"], status["epsilon"], status["seed"]) == (unit, epsilon, seed)
                model = Path(status["output_dir"]) / "model.pt"
                assert hashlib.sha256(model.read_bytes()).hexdigest() == status["model_sha256"]
                assert len(list((run / "public-capture").glob("accountant-*.json"))) == 15
                (run / "execution-status.json").write_text(json.dumps({
                    "status": "trained_unscored", "unit": unit, "seed": seed,
                    "epsilon": epsilon, "elapsed_s": time.monotonic() - started,
                    "protocol_sha256": protocol_sha,
                    "completed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
                print("COMPLETED", name, f"{status['elapsed_s']:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    sys.exit(main(parser.parse_args().root))
