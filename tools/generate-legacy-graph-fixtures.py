"""Regenerate saved graph fixtures with the actual dsFlower 0.7.0 builders.

Run with the PyTorch interpreter and --main-server / --main-client pointing to
unmodified 0.7.0 checkouts. Tests consume the resulting fixtures, never this
historical source at runtime.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-server", required=True, type=Path)
    parser.add_argument("--main-client", required=True, type=Path)
    args = parser.parse_args()
    builder_path = args.main_server / "inst/flower_app/dsflower_runner/model_spec.py"
    predictor_path = args.main_client / "inst/python/predict_helper.py"
    builder = load("main_model_spec", builder_path)
    predictor = load("main_predict_helper", predictor_path)
    predictor._load_model_spec_module = lambda: builder
    output = Path(__file__).resolve().parents[1] / "inst/python/tests/fixtures/legacy_graph"
    output.mkdir(parents=True, exist_ok=True)
    cases = {
        "named": {"kind": "graph", "nodes": [
            {"name": "hidden", "op": "linear", "out": 3, "in": ["@in"]},
            {"name": "head", "op": "linear", "in": ["hidden"]}], "output": "head"},
        "colliding": {"kind": "graph", "nodes": [
            {"name": "n0", "op": "linear", "out": 2, "in": ["@in"]},
            {"name": "n1", "op": "linear", "out": 2, "in": ["@in"]},
            {"name": "n2", "op": "concat", "in": ["n1", "n0"]},
            {"name": "n3", "op": "linear", "in": ["n2"]}], "output": "n3"},
    }
    for name, spec in cases.items():
        torch.manual_seed(5)
        model = builder.build_from_spec(spec, 2, 1)
        artifact = output / (name + ".pt")
        torch.save(model.state_dict(), artifact)
        inputs = np.asarray([[1., 2.], [3., 4.]], dtype=np.float32)
        encoded = base64.b64encode(json.dumps(spec).encode()).decode()
        metadata = {
            "producer": "dsFlower/dsFlowerClient 0.7.0 (main)",
            "builder_sha256": hashlib.sha256(builder_path.read_bytes()).hexdigest(),
            "predictor_sha256": hashlib.sha256(predictor_path.read_bytes()).hexdigest(),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "model_spec": spec, "loss_name": "bce_logits", "inputs": inputs.tolist(),
            "predictions": predictor.predict_pytorch_spec(
                artifact, inputs, "prob", encoded, "bce_logits"),
            "state_keys": list(model.state_dict()),
        }
        (output / (name + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
