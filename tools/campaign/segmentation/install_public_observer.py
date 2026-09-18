#!/usr/bin/env python3
"""Install public instrumentation only into the dedicated pod campaign venv."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import sysconfig


def install():
    prefix = Path(sys.prefix)
    if prefix != Path("/workspace/segmentation/venv") or prefix == Path(sys.base_prefix):
        raise RuntimeError("run with /workspace/segmentation/venv/bin/python only")
    destination = Path(sysconfig.get_path("purelib"))
    if prefix.resolve() not in destination.resolve().parents:
        raise RuntimeError("campaign site-packages is outside the isolated environment")
    source = Path(__file__).parent / "benchmark_hooks/segmentation_public_observer.py"
    target = destination / source.name
    shutil.copyfile(source, target)
    target.chmod(0o644)
    pth = destination / "segmentation_public_observer.pth"
    pth.write_text("import segmentation_public_observer; segmentation_public_observer.install()\n")
    pth.chmod(0o644)
    assert target.read_bytes() == source.read_bytes()
    print(json.dumps({"observer_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "observer": str(target), "pth": str(pth),
                      "runner_or_integrity_hook_modified": False}, sort_keys=True))


if __name__ == "__main__":
    install()
