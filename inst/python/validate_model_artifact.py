"""Preflight one public validation artifact without contacting a data node."""

import json
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 2:
        raise ValueError("expected one bounded JSON validation contract")
    contract_path = Path(sys.argv[1])
    if not contract_path.is_file() or contract_path.stat().st_size > 256 * 1024:
        raise ValueError("validation preflight contract is missing or too large")
    with contract_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("validation preflight contract must be an object")

    flower_app = Path(__file__).resolve().parents[1] / "flower_app"
    sys.path.insert(0, str(flower_app))
    from dsflower_runner import validation

    arrays = validation.public_model_arrays(config)
    if not isinstance(arrays, list) or not arrays:
        raise ValueError("validation artifact produced no public model arrays")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("invalid validation artifact: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
