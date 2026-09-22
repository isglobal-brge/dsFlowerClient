#!/usr/bin/env python3
"""Read-only, bounded preflight for the public vision campaign; scores no data."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import socket
import subprocess
import time


def probe(argv, timeout):
    started = time.monotonic()
    try:
        run = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        result = dict(returncode=run.returncode, stdout=run.stdout, stderr=run.stderr,
                      timed_out=False)
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        result = dict(returncode=None, stdout=decode(error.stdout),
                      stderr=decode(error.stderr), timed_out=True)
    result.update(argv=argv, timeout_s=timeout,
                  elapsed_s=round(time.monotonic() - started, 3))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/workspace/cells-vision")
    parser.add_argument("--library", default="/workspace/cells-vision/Rlib")
    parser.add_argument("--runner-sha256", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    result = dict(schema="dsflower-vision-runtime-preflight-v1",
                  generated_at=datetime.now(timezone.utc).isoformat(),
                  hostname=socket.gethostname(),
                  expected_versions={"dsFlower": "0.5.1", "dsFlowerClient": "0.5.0"}, expected_runner_sha256=args.runner_sha256,
                  installed_versions=None, installed_runner_sha256=None,
                  training_started=False, scoring_started=False)
    result["workspace_mount"] = [line for line in Path("/proc/self/mountinfo").read_text().splitlines()
                                 if " /workspace " in line]
    result["gpu_probe"] = probe(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used",
                                  "--format=csv"], 10)
    result["gpus"] = list(csv.DictReader(io.StringIO(result["gpu_probe"]["stdout"]),
                                        skipinitialspace=True))
    paths = ["/workspace", str(Path(args.library) / "dsFlower/DESCRIPTION"),
             str(Path(args.root) / "prepared/busbra/audit.json")]
    result["filesystem_probes"] = [probe(["stat", "--", path], 10) for path in paths]
    result["release_probe"] = None
    if any(item["returncode"] != 0 for item in result["filesystem_probes"]):
        result.update(status="blocked", blocker_kind="workspace_unavailable",
                      error="Cannot access /workspace, the installed release, or prepared BUS-BRA; runtime verification and training cannot start.")
    else:
        r_code = '''args <- commandArgs(TRUE)
.libPaths(c(args[[1]], .libPaths()))
library(dsFlower)
library(dsFlowerClient)
cat(jsonlite::toJSON(list(
  r_version = R.version.string,
  versions = list(dsFlower = as.character(packageVersion("dsFlower")),
                  dsFlowerClient = as.character(packageVersion("dsFlowerClient"))),
  runner_sha256 = list(dsFlower = dsFlower:::.compute_harness_hash(),
                      dsFlowerClient = dsFlowerClient:::.compute_local_runner_hash())),
  auto_unbox = TRUE))
'''
        result["release_probe"] = probe(["Rscript", "-e", r_code, args.library], 45)
        check = result["release_probe"]
        if check["returncode"] == 0:
            release = json.loads(check["stdout"])
            result["installed_versions"] = release["versions"]
            result["r_version"] = release["r_version"]
            result["installed_runner_sha256"] = release["runner_sha256"]
            matched = (release["versions"] == result["expected_versions"]
                       and all(v == args.runner_sha256 for v in release["runner_sha256"].values()))
            result.update(status="verified" if matched else "blocked",
                          blocker_kind=None if matched else "release_mismatch",
                          error=None if matched else "Expected dsFlower 0.5.1, dsFlowerClient 0.5.0, and unchanged canonical runners.")
        else:
            result.update(status="blocked", blocker_kind="release_verification_failed",
                          error="Installed R packages could not be verified; see release_probe.")
    if result["status"] == "verified":
        result["torch_probe"] = probe([str(Path(args.root) / "venvs/pytorch-gpu/bin/python"),
            "-c", "import json, torch, torchvision; assert torch.cuda.is_available(); "
            "print(json.dumps(dict(torch=torch.__version__, torchvision=torchvision.__version__, "
            "cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())))"], 45)
        if result["torch_probe"]["returncode"] != 0:
            result.update(status="blocked", blocker_kind="cuda_unavailable", error="CUDA runtime verification failed.")
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
