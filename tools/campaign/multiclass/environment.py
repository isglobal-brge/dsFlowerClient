#!/usr/bin/env python3
"""Record provisioning and dependency versions without reading node secrets."""
import argparse
import datetime
import json
import platform
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, default=Path("/workspace/cells"))
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
root = args.root
start = (root / "provision-start.txt").read_text().strip()
end = (root / "provision-end.txt").read_text().strip()
dt = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
packages = {}
for name, python in {"node": root / "venvs/pytorch/bin/python",
                     "client": root / "client/venv/bin/python"}.items():
    packages[name] = json.loads(subprocess.check_output(
        ["uv", "pip", "list", "--python", str(python), "--format", "json"], text=True))
r_code = 'x <- installed.packages(); cat(jsonlite::toJSON(as.data.frame(x[,c("Package","Version")]), dataframe="rows"))'
r_packages = json.loads(subprocess.check_output(["Rscript", "-e", r_code], text=True))
def optional_text(path):
    path = Path(path)
    return path.read_text().strip() if path.exists() else None
record = dict(schema="dsflower-campaign-environment-v1", pod_id="6aq9cxaigfwlby",
              pod_name="pod-flower-multiclass", platform=platform.platform(),
              provisioning_started_at=start, provisioning_finished_at=end,
              provisioning_elapsed_s=(dt(end)-dt(start)).total_seconds(),
              memory_cgroup_bytes=optional_text("/sys/fs/cgroup/memory.max"),
              cpu_cgroup=optional_text("/sys/fs/cgroup/cpu.max"),
              os_release=optional_text("/etc/os-release"),
              uv_version=subprocess.check_output(["uv", "--version"], text=True).strip(),
              python_packages=packages, r_packages=r_packages,
              provisioning_notes="CRAN jammy apt R 4.6.1; R CMD INSTALL of unchanged v0.5.0 sources; configure-created external uv environments; CPU torch preinstalled for prediction, remaining prediction dependencies installed by the package helper.")
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(record, indent=2) + "\n")
print(f"Provisioning: {record['provisioning_elapsed_s']:.0f}s")
