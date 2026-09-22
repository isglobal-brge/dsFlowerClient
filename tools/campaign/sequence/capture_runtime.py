#!/usr/bin/env python3
"""Verify release identity and record the installed benchmark environment."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

root = Path(sys.argv[1]).resolve()
tools = Path(__file__).resolve().parent
release = json.loads((tools / "release-source.json").read_text())
env = dict(os.environ, R_LIBS_USER=str(root / "Rlib"))
code = '''library(dsFlower); library(dsFlowerClient); cat(jsonlite::toJSON(list(
  R = R.version.string, DSI = as.character(packageVersion("DSI")),
  DSLite = as.character(packageVersion("DSLite")), dsFlower = as.character(packageVersion("dsFlower")),
  dsFlowerClient = as.character(packageVersion("dsFlowerClient")),
  server_runner_sha256 = dsFlower:::.compute_harness_hash(),
  client_runner_sha256 = dsFlowerClient:::.compute_local_runner_hash()), auto_unbox=TRUE))'''
r = json.loads(subprocess.check_output(["Rscript", "-e", code], env=env, text=True))
assert r["server_runner_sha256"] == r["client_runner_sha256"] == release["dsFlower"]["runner_sha256"]
assert r["dsFlower"] == r["dsFlowerClient"] == "0.5.0"
assert torch.cuda.is_available()
files = {}
for path in sorted(tools.rglob("*")):
    if path.is_file() and "__pycache__" not in path.parts:
        files[str(path.relative_to(tools))] = hashlib.sha256(path.read_bytes()).hexdigest()
result = {"recorded_at": datetime.now(timezone.utc).isoformat(),
    "host": {"pod_id": "5abpdvx54g6uz2", "pod_name": "pod-flower-sequence",
             "platform": platform.platform(), "gpu": torch.cuda.get_device_name(),
             "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
             "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()},
    "packages": {**r, "python": sys.version, "python_packages": {
        name: importlib.metadata.version(name) for name in ("torch", "opacus", "flwr", "numpy", "pandas", "pyarrow", "cryptography")}},
    "release_sources": release, "tooling_sha256": files,
    "tooling_commit": os.environ.get("SEQUENCE_TOOLING_COMMIT"),
    "privacy_randomness": "Release-owned cryptographic secrets; no deterministic-noise replacement or secret publication.",
    "initialization_seeds": [20260922, 20260923, 20260924]}
(root / "runtime.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["packages"], indent=2))
