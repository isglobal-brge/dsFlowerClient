"""Validate one public dsFlower declarative model without touching node data."""

import importlib.util
import json
from pathlib import Path
import sys


def _load_builder():
    path = (Path(__file__).resolve().parents[1] / "flower_app" /
            "dsflower_runner" / "model_spec.py")
    spec = importlib.util.spec_from_file_location(
        "_dsflower_client_model_spec", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) != 2:
        raise ValueError("expected one bounded JSON contract path")
    path = Path(sys.argv[1])
    if not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ValueError("model preflight contract is missing or too large")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    builder = _load_builder()
    loss = str(payload["loss_name"])
    cfg = {
        "num-classes": int(payload["num_classes"]),
        "num-labels": int(payload["num_labels"]),
    }
    out_dim = builder.output_width(loss, cfg)
    builder.build_from_spec(
        payload["spec"], int(payload["input_dim"]), int(out_dim),
        num_labels=int(payload["num_labels"]),
        output_limit=builder.output_limit_for_loss(loss))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("invalid declarative model: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
