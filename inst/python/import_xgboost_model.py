"""Build a portable dsFlower bundle from one safe external XGBoost JSON.

This helper is an isolated, data-only importer.  It never loads an upstream
model runtime and it never attributes differential-privacy provenance to the
external model.  The caller supplies the already canonical public native-tree
request which binds feature and target semantics.
"""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile


MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MODEL_FILE = "model.xgboost-ensemble.json"
PROFILE_FILE = "model.xgboost-ensemble.profile.json"
IMPORT_CONTRACT = "dsflower-external-xgboost-import-v1"
EXTERNAL_ENSEMBLE_CONTRACT = "dsflower-external-xgboost-ensemble-v1"
ERROR_MESSAGE = "dsFlower external XGBoost import rejected\n"
_FORBIDDEN_MODULE_ROOTS = frozenset((
    "catboost", "cloudpickle", "dill", "flwr", "joblib", "lightgbm",
    "numpy", "pandas", "pickle", "sklearn", "torch", "xgboost",
))


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError("invalid importer arguments")


def _canonical_json(value):
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _runner_modules():
    runner_root = Path(__file__).resolve().parents[1] / "flower_app"
    if not runner_root.is_dir():
        raise RuntimeError("bundled runner is unavailable")
    sys.path.insert(0, str(runner_root))
    from dsflower_runner import native_tree_engine
    from dsflower_runner import native_tree_request
    from dsflower_runner import xgboost_predictor
    from dsflower_runner.xgboost_sanitizer import sanitize_xgboost_json
    return (
        native_tree_engine, native_tree_request, xgboost_predictor,
        sanitize_xgboost_json,
    )


def _read_regular_file(path, byte_cap):
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("artifact path is invalid")
    if os.path.islink(path):
        raise ValueError("artifact must not be a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= byte_cap:
            raise ValueError("artifact is not a bounded regular file")
        chunks = []
        remaining = int(info.st_size)
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("artifact changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("artifact changed while being read")
        payload = b"".join(chunks)
        if len(payload) != info.st_size:
            raise ValueError("artifact changed while being read")
        return payload
    finally:
        os.close(descriptor)


def _empty_output_directory(path):
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("output directory is invalid")
    if os.path.islink(path) or not os.path.isdir(path):
        raise ValueError("output directory must be an existing real directory")
    with os.scandir(path) as entries:
        if next(entries, None) is not None:
            raise ValueError("output directory must be empty")
    return os.path.abspath(path)


def _require_exact_entries(directory, paths):
    expected = {os.path.basename(path) for path in paths}
    with os.scandir(directory) as entries:
        actual = {entry.name for entry in entries}
    if actual != expected:
        raise ValueError("output staging directory changed during import")


def _write_temporary(directory, payload):
    descriptor, path = tempfile.mkstemp(
        prefix=".dsflower-xgboost-import-", dir=directory)
    try:
        os.chmod(path, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    os.close(descriptor)
    return path


def _install_exclusive(temporary, destination):
    """Install without ever replacing a pre-existing destination."""
    # Windows rename is atomic and refuses an existing destination.  POSIX
    # rename replaces, so use same-directory hard-link creation there.
    if os.name == "nt":
        os.rename(temporary, destination)
        return
    linked = False
    try:
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        os.unlink(temporary)
    except Exception:
        if linked:
            _remove(destination)
        raise


def _remove(path):
    if path:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _sync_directory(path):
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _build_ensemble(raw_model, manifest, predictor_module, sanitizer):
    # This dependency-free profile function is also what the pure predictor
    # uses when it re-sanitizes every ensemble member before private I/O.
    canonical, arguments, task, _base_score, _bounds = \
        predictor_module._prediction_profile(manifest)
    safe_model, safe_digest = sanitizer(raw_model, **arguments)
    model_value = json.loads(safe_model.decode("ascii"))
    ensemble = _canonical_json({
        "aggregation": "mean_prediction",
        "contract": EXTERNAL_ENSEMBLE_CONTRACT,
        "engine": "xgboost",
        "models": [model_value],
        "public_schema_sha256": canonical["public_schema"]["sha256"],
        "task": task,
        "version": 1,
    })
    if len(ensemble) > canonical["resources"]["max_artifact_bytes"]:
        raise ValueError("ensemble exceeds its public byte ceiling")
    # Re-open the final bytes through the exact runtime parser.  Besides
    # checking the canonical container, this repeats member sanitization.
    parsed = predictor_module.parse_xgboost_ensemble(ensemble, manifest)
    schema = canonical["public_schema"]
    row = [
        float(lower) + (float(upper) - float(lower)) / 2.0
        for lower, upper in zip(schema["lower"], schema["upper"])
    ]
    prediction = parsed.predict([row])
    if len(prediction) != 1 or not math.isfinite(float(prediction[0])) or \
            (task == "binary" and not 0.0 <= float(prediction[0]) <= 1.0):
        raise ValueError("imported predictor probe failed")
    if hashlib.sha256(safe_model).hexdigest() != safe_digest:
        raise RuntimeError("sanitizer digest mismatch")
    return ensemble


def _public_manifest(request, request_b64, request_sha256, spec,
                     ensemble, ensemble_sha256, profile):
    schema = request["public_schema"]
    target = schema["target"]
    if request["task"] == "binary":
        target_levels = [item["value"] for item in target["levels"]]
        target_bounds = None
    else:
        target_levels = None
        target_bounds = {
            "lower": target["lower"], "upper": target["upper"]}
    profile_sha256 = hashlib.sha256(profile).hexdigest()
    sanitization = {
        "profile": EXTERNAL_ENSEMBLE_CONTRACT,
        "privacy_basis": "external-unverified",
        "contains_raw_records": False,
        # The format has no auxiliary statistic channel, but external leaf
        # values were not produced by dsFlower's DP training mechanism.
        "contains_unnoised_statistics": True,
        "contains_feature_names": False,
        "contains_target_name": False,
        "contains_training_history": False,
        "contains_backend_logs": False,
        "contains_paths": False,
        "contains_executable_payload": False,
    }
    return {
        "artifact": {
            "file": spec["model_file"],
            "format": spec["ensemble_format"],
            "sha256": ensemble_sha256,
            "size_bytes": len(ensemble),
        },
        "contract": IMPORT_CONTRACT,
        "data_kind": "tabular",
        "engine": "xgboost",
        "feature_lower": schema["lower"],
        "feature_upper": schema["upper"],
        "features": schema["features"],
        "native_tree_request_b64": request_b64,
        "native_tree_request_sha256": request_sha256,
        "prediction_profile": {
            "file": spec["profile_file"],
            "sha256": profile_sha256,
            "size_bytes": len(profile),
        },
        "public_schema_sha256": schema["sha256"],
        "sanitization": sanitization,
        "target_bounds": target_bounds,
        "target_levels": target_levels,
        "task": request["task"],
        "track": "native_tree",
        "version": 1,
    }


def _reject_forbidden_imports():
    loaded = {name.partition(".")[0] for name in sys.modules}
    if loaded.intersection(_FORBIDDEN_MODULE_ROOTS):
        raise RuntimeError("forbidden model runtime was loaded")


def import_model(artifact_path, request_path, request_sha256, output_dir):
    output_dir = _empty_output_directory(output_dir)
    raw_model = _read_regular_file(artifact_path, MAX_ARTIFACT_BYTES)
    request_bytes = _read_regular_file(request_path, MAX_MANIFEST_BYTES)
    request_b64 = base64.b64encode(request_bytes).decode("ascii")
    (native_tree_engine, native_tree_request, xgboost_predictor,
     sanitizer) = _runner_modules()
    request = native_tree_request.parse_request_wire(
        request_b64, request_sha256)
    if request.get("engine") != "xgboost" or request.get("task") not in (
            "binary", "regression"):
        raise ValueError("request is not an XGBoost import profile")
    manifest = native_tree_request.public_backend_manifest(request)
    ensemble = _build_ensemble(
        raw_model, manifest, xgboost_predictor, sanitizer)
    ensemble_sha256 = hashlib.sha256(ensemble).hexdigest()
    profile = native_tree_engine.build_prediction_profile(
        request, request_b64, request_sha256, ensemble, ensemble_sha256)
    spec = native_tree_engine.validate_prediction_profile(
        profile, request, request_b64, request_sha256, ensemble)
    if spec["engine"] != "xgboost" or spec["model_file"] != MODEL_FILE or \
            spec["profile_file"] != PROFILE_FILE:
        raise RuntimeError("prediction profile engine mismatch")
    manifest_value = _public_manifest(
        request, request_b64, request_sha256, spec, ensemble,
        ensemble_sha256, profile)
    manifest_bytes = _canonical_json(manifest_value)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("public import manifest exceeds its byte ceiling")
    _reject_forbidden_imports()

    temporary_model = None
    temporary_profile = None
    installed = []
    try:
        temporary_model = _write_temporary(output_dir, ensemble)
        temporary_profile = _write_temporary(output_dir, profile)
        _require_exact_entries(
            output_dir, (temporary_model, temporary_profile))
        model_path = os.path.join(output_dir, spec["model_file"])
        profile_path = os.path.join(output_dir, spec["profile_file"])
        _install_exclusive(temporary_model, model_path)
        temporary_model = None
        installed.append(model_path)
        _install_exclusive(temporary_profile, profile_path)
        temporary_profile = None
        installed.append(profile_path)
        _require_exact_entries(output_dir, installed)
        _sync_directory(output_dir)
    except Exception:
        _remove(temporary_model)
        _remove(temporary_profile)
        for path in reversed(installed):
            _remove(path)
        raise
    return manifest_bytes


def _arguments(argv):
    parser = _ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    try:
        if not sys.flags.isolated or not sys.flags.no_site:
            raise RuntimeError("isolated Python without site is required")
        args = _arguments(sys.argv[1:] if argv is None else argv)
        result = import_model(
            args.artifact, args.request, args.request_sha256,
            args.output_dir)
        sys.stdout.write(result.decode("ascii"))
        sys.stdout.flush()
        return 0
    except Exception:
        try:
            sys.stderr.write(ERROR_MESSAGE)
            sys.stderr.flush()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
