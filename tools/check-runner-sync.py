#!/usr/bin/env python3
"""Fail CI when the client-bundled runner differs from dsFlower's canonical copy."""

import argparse
import hashlib
from pathlib import Path
import sys


def tree(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        rel = path.relative_to(root).as_posix()
        result[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        default=str(Path(__file__).resolve().parents[2] / "dsFlower"),
        help="path to the dsFlower repository",
    )
    args = parser.parse_args()
    client = Path(__file__).resolve().parents[1] / "inst/flower_app/dsflower_runner"
    server = Path(args.server).resolve() / "inst/flower_app/dsflower_runner"
    if not server.is_dir():
        parser.error("canonical server runner not found: %s" % server)
    left, right = tree(server), tree(client)
    if left == right:
        print("dsflower_runner trees are byte-identical")
        return 0
    for name in sorted(set(left) | set(right)):
        if left.get(name) != right.get(name):
            print("runner mismatch: %s" % name, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
