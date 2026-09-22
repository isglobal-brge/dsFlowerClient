"""Regenerate the small real training release over fixed public synthetic data.

Run with NumPy installed from the package root. The fixed secret is solely for
this public test fixture and never changes the production randomness source.
"""
import base64
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "inst" / "flower_app"))
from dsflower_runner import native_tree_engine, native_tree_request, seeding
from dsflower_runner.native_tree_runtime_probe import _request


def training_release():
    request = _request("random_forest")
    # Match the R request ABI's field order before training and binding it.
    request = {key: request[key] for key in (
        "contract", "engine", "mode", "task", "public_schema",
        "parameters", "resources")}
    raw = json.dumps(request, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    b64 = base64.b64encode(raw).decode("ascii")
    sha = hashlib.sha256(raw).hexdigest()
    parsed = native_tree_request.parse_request_wire(b64, sha)
    manifest = native_tree_request.public_backend_manifest(parsed)
    features = np.tile([[-0.75], [-0.25], [0.25], [0.75]], (128, 1))
    target = np.tile([0.0, 0.0, 1.0, 1.0], 128)
    original = seeding._node_secret
    seeding._node_secret = lambda: b"\x00" * 32
    try:
        member = native_tree_engine.train_model(manifest, features, target)
    finally:
        seeding._node_secret = original
    artifact, digest = native_tree_engine.build_ensemble(manifest, [member])
    profile = native_tree_engine.build_prediction_profile(
        parsed, b64, sha, artifact, digest)
    native_tree_engine.parse_ensemble(manifest, artifact)
    return artifact, profile


if __name__ == "__main__":
    artifact, profile = training_release()
    directory = Path(__file__).parent
    (directory / "native-tree-training.json").write_bytes(artifact)
    (directory / "native-tree-training.profile.json").write_bytes(profile)
