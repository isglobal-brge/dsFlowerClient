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
    if prefix.resolve() != Path("/workspace/cells-sequence/venvs/pytorch-gpu").resolve() or prefix == Path(sys.base_prefix):
        raise RuntimeError("run with /workspace/cells-sequence/venvs/pytorch-gpu/bin/python only")
    destination = Path(sysconfig.get_path("purelib"))
    if prefix.resolve() not in destination.resolve().parents:
        raise RuntimeError("campaign site-packages is outside the isolated environment")
    source = Path(__file__).parent / "benchmark_hooks/sequence_public_observer.py"
    target = destination / source.name
    temporary = target.with_suffix(".py.tmp")
    shutil.copyfile(source, temporary)
    temporary.chmod(0o644)
    temporary.replace(target)
    pth = destination / "sequence_public_observer.pth"
    temporary = pth.with_suffix(".pth.tmp")
    temporary.write_text("import sequence_public_observer; sequence_public_observer.install()\n")
    temporary.chmod(0o644)
    temporary.replace(pth)
    assert target.read_bytes() == source.read_bytes()
    print(json.dumps({"observer_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "observer": str(target), "pth": str(pth),
                      "runner_or_integrity_hook_modified": False}, sort_keys=True))


if __name__ == "__main__":
    install()
