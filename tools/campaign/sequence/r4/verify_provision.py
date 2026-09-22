#!/usr/bin/env python3
"""Verify frozen provisioning and record installed source and runtime identity."""
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

root = Path("/workspace/cells-sequence")
for name in "flwr numpy pandas pyarrow cryptography torch opacus torchvision PIL nibabel pydicom nrrd SimpleITK monai".split():
    importlib.import_module(name)
assert torch.cuda.is_available()
env = dict(os.environ, R_LIBS_USER="/opt/cells-sequence/Rlib",
           DSFLOWER_VENV_ROOT="/opt/cells-sequence/venvs",
           DSFLOWER_CLIENT_VENV_ROOT="/opt/cells-sequence/client")
r_code = '''library(dsFlower); library(dsFlowerClient)
writeLines(dsFlower:::.python_env_spec_hash("pytorch-gpu"), "/opt/cells-sequence/venvs/pytorch-gpu/.dsflower_ready")
writeLines(dsFlowerClient:::.client_venv_marker(), "/opt/cells-sequence/client/venv/.dsflower_client_ready")
stopifnot(dsFlower:::.venv_is_healthy("/opt/cells-sequence/venvs/pytorch-gpu", "pytorch-gpu"),
          dsFlowerClient:::.client_venv_is_healthy())
cat(jsonlite::toJSON(list(R=R.version.string,DSI=as.character(packageVersion("DSI")),
DSLite=as.character(packageVersion("DSLite")),dsFlower=as.character(packageVersion("dsFlower")),
dsFlowerClient=as.character(packageVersion("dsFlowerClient")),
server_runner_sha256=dsFlower:::.compute_harness_hash(),
client_runner_sha256=dsFlowerClient:::.compute_local_runner_hash()),auto_unbox=TRUE))'''
packages = json.loads(subprocess.check_output(["Rscript", "-e", r_code], env=env, text=True))
assert packages["dsFlower"] == packages["dsFlowerClient"] == "0.5.1"
assert packages["server_runner_sha256"] == packages["client_runner_sha256"] == "2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724"
guard = root / "Rlib/dsFlower/python/sitecustomize.py"
packages["server_guard_sha256"] = hashlib.sha256(guard.read_bytes()).hexdigest()
assert packages["server_guard_sha256"] == "3ae7c9ce6750c81c00e8c70e618d0bc52978486d7736587f39c716ff401fe98d"
packages["python"] = sys.version
packages["python_packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
pins = Path(__file__).with_name("requirements-r3.txt").read_text().splitlines()
for pin in pins:
    if not pin or pin.startswith("#"):
        continue
    name, version = pin.split("==")
    assert importlib.metadata.version(name) == version, pin
source = {}
for name in ("dsFlower", "dsFlowerClient"):
    path = root / "src" / name
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    source[name] = {"commit": git("rev-parse", "HEAD"), "origin": git("remote", "get-url", "origin"),
                    "package_source_unmodified": not bool(git("status", "--porcelain", "--", "R", "inst/flower_app", "inst/python", "DESCRIPTION", "configure"))}
    assert source[name]["package_source_unmodified"]
audit = json.loads((root / "prepared/audit.json").read_text())
assert audit["test_accessed"] is False
assert audit["archive_sha256"] == "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
result = {"recorded_at": datetime.now(timezone.utc).isoformat(),
          "pod_name": "pod-flower-sequence-2", "platform": platform.platform(),
          "gpu": torch.cuda.get_device_name(), "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
          "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
          "packages": packages, "sources": source,
          "storage": {name: str((root / name).resolve()) for name in ("Rlib", "venvs", "client")},
          "prepared": {k: audit[k] for k in ("archive_sha256", "train_npz_sha256", "n_train_windows", "n_train_subjects", "site_subjects", "n_per_site", "test_accessed")},
          "all_r3_python_versions_match": True, "pod_left_running": True}
(root / "r4/provisioning-runtime.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
