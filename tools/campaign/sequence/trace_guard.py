#!/usr/bin/env python3
"""Synthetic guarded recurrent import probe; never reads the dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CODE = r'''
import base64, importlib.metadata, json, sys, traceback
import sitecustomize as guard
original = guard._abort
def traced_abort(message):
    traceback.print_stack(file=sys.stderr)
    original(message)
guard._abort = traced_abort
print(json.dumps({"python": sys.version, "versions": {p: importlib.metadata.version(p)
    for p in ("torch", "opacus", "flwr")}, "kind": sys.argv[1]}), flush=True)
print("PHASE client_app import", flush=True)
from dsflower_runner import client_app, params
print("PHASE recurrent model construction", flush=True)
spec = {"kind": "graph", "output": "out", "nodes": [
    {"name": "x", "op": "reshape", "in": ["@in"], "shape": [128, 9]},
    {"name": "h", "op": sys.argv[1], "in": ["x"], "hidden": 32},
    {"name": "out", "op": "linear", "in": ["h"], "out": "@out"}]}
cfg = {"model-spec-b64": base64.b64encode(json.dumps(spec).encode()).decode(),
       "num-features": 1152, "num-classes": 6, "loss-name": "cross_entropy"}
model = params.load_user_model(cfg, 1152, "cross_entropy")
print("PHASE optimizer construction", flush=True)
import torch
optimizer = torch.optim.SGD(model.parameters(), lr=.001)
x = torch.zeros(7, 1152)
y = torch.arange(7) % 6
loss = torch.nn.functional.cross_entropy(model(x), y)
loss.backward()
optimizer.step()
print("PASS guarded recurrent import and ordinary optimizer step", flush=True)
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root, out = args.root, args.out
    out.mkdir(parents=True, exist_ok=False)
    runner = (root / "Rlib/dsFlower/flower_app/dsflower_runner").resolve()
    digest = hashlib.sha256()
    for path in sorted(runner.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            digest.update(str(path.relative_to(runner)).encode() + b"\n" + path.read_bytes() + b"\0")
    results = {}
    with tempfile.TemporaryDirectory() as manifest:
        Path(manifest, "pinned_packages.json").write_text(json.dumps({"dsflower_runner": digest.hexdigest()}))
        Path(manifest, "manifest.json").write_text(json.dumps({"dp-track": "neural"}))
        env = dict(os.environ, DSFLOWER_MANIFEST_DIR=manifest, OMP_NUM_THREADS="2",
                   PYTHONPATH=os.pathsep.join([str((root / "Rlib/dsFlower/python").resolve()), str(runner.parent)]))
        for kind in ("lstm", "gru"):
            result = subprocess.run([sys.executable, "-c", CODE, kind], env=env,
                                    capture_output=True, text=True, timeout=120)
            (out / (kind + ".log")).write_text(result.stdout + result.stderr)
            results[kind] = {"exit_code": result.returncode, "stdout": result.stdout,
                             "stderr": result.stderr}
            print(kind, result.returncode, result.stdout, flush=True)
    (out / "trace.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
