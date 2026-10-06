"""Regenerate the small real training release over fixed public synthetic data.

Regenerate with NumPy 2.4.6. The fixed secret and public execution profile are
solely for this canonical-container fixture. The actual runner content hash
remains bound, so source changes require deliberate regeneration. Production
releases always use the actual runtime; no platform portability is inferred
for native arithmetic that has not been tested.
"""
import base64
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "inst" / "flower_app"))
from dsflower_runner import native_tree_engine, native_tree_request, seeding, tree_release
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
    original_runtime = seeding._runtime_fingerprint
    original_numeric = tree_release.numeric_execution_profile
    runner_hash = original_runtime(False)["runner_sha256"]
    fixture_runtime = {"python": "3.11.16", "implementation": "CPython",
        "system": "linux", "machine": "x86_64", "backend": {"kind": "cpu"},
        "packages": {"cryptography": "46.0.7", "numpy": "2.4.6",
                     "opacus": "absent", "torch": "absent", "torchvision": "absent"},
        "runner_sha256": runner_hash}
    seeding._node_secret = lambda: b"\x00" * 32
    seeding._runtime_fingerprint = lambda *args: fixture_runtime
    tree_release.numeric_execution_profile = lambda: {
        "byteorder": "little", "contract": "dsflower-tree-gaussian-numeric-v1",
        "machine": "x86_64", "numpy": "2.4.6",
        "rng": "chacha20-box-muller-four-sample/v2", "system": "linux"}
    try:
        member = native_tree_engine.train_model(manifest, features, target)
    finally:
        seeding._node_secret = original
        seeding._runtime_fingerprint = original_runtime
        tree_release.numeric_execution_profile = original_numeric
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
