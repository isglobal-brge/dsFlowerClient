"""Dependency-free local prediction for a sanitized dsFlower XGBoost bundle.

The helper accepts only data files and the public prediction sidecar.  It never
loads XGBoost, pickle, a native library, or executable model content.
"""

import argparse
import base64
import csv
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import sys


PROFILE_CONTRACT = "dsflower-xgboost-prediction-profile-v1"
OUTPUT_CONTRACT = "dsflower-xgboost-local-prediction-v1"
ENSEMBLE_FORMAT = "dsflower-xgboost-ensemble-json-v1"
MAX_PROFILE_BYTES = 128 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_ROWS = 5_000_000
_HEX_64_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_FIELDS = frozenset((
    "artifact", "contract", "native_tree_request_b64",
    "native_tree_request_sha256", "public_schema_sha256", "task", "version",
))
_ARTIFACT_FIELDS = frozenset(("format", "sha256", "size_bytes"))


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _canonical_json(value):
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _read_bounded(path, byte_cap, where):
    with open(path, "rb") as handle:
        payload = handle.read(byte_cap + 1)
        if len(payload) > byte_cap or handle.read(1):
            raise ValueError("%s exceeds its byte bound" % where)
    if not payload:
        raise ValueError("%s is empty" % where)
    return payload


def _runner_modules():
    runner_root = Path(__file__).resolve().parents[1] / "flower_app"
    if not runner_root.is_dir():
        raise RuntimeError("bundled predictor is unavailable")
    sys.path.insert(0, str(runner_root))
    from dsflower_runner import native_tree_request, xgboost_predictor
    return native_tree_request, xgboost_predictor


def _load_predictor(model_path, profile_path):
    profile_bytes = _read_bounded(
        profile_path, MAX_PROFILE_BYTES, "prediction profile")
    if any(byte > 0x7F for byte in profile_bytes):
        raise ValueError("prediction profile is not ASCII")
    try:
        profile = json.loads(
            profile_bytes.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-finite JSON number")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError,
            RecursionError) as exc:
        raise ValueError("prediction profile is invalid") from exc
    if _canonical_json(profile) != profile_bytes or not isinstance(
            profile, dict) or frozenset(profile) != _PROFILE_FIELDS:
        raise ValueError("prediction profile is not canonical")

    binding = profile["artifact"]
    version = profile["version"]
    if not isinstance(binding, dict) or frozenset(binding) != _ARTIFACT_FIELDS or \
            profile["contract"] != PROFILE_CONTRACT or \
            type(version) is not int or version != 1 or \
            profile["task"] not in ("binary", "regression") or \
            binding["format"] != ENSEMBLE_FORMAT or \
            type(binding["size_bytes"]) is not int or not (
                1 <= binding["size_bytes"] <= MAX_ARTIFACT_BYTES) or \
            not isinstance(binding["sha256"], str) or not _HEX_64_RE.fullmatch(
                binding["sha256"]) or \
            not isinstance(profile["public_schema_sha256"], str) or \
            not _HEX_64_RE.fullmatch(profile["public_schema_sha256"]):
        raise ValueError("prediction profile contract is invalid")

    native_tree_request, xgboost_predictor = _runner_modules()
    request = native_tree_request.parse_request_wire(
        profile["native_tree_request_b64"],
        profile["native_tree_request_sha256"],
    )
    if request["engine"] != "xgboost" or request["task"] != profile["task"] or \
            request["public_schema"]["sha256"] != \
            profile["public_schema_sha256"]:
        raise ValueError("prediction profile differs from its request")

    artifact = _read_bounded(model_path, MAX_ARTIFACT_BYTES, "ensemble")
    if len(artifact) != binding["size_bytes"] or not hmac.compare_digest(
            hashlib.sha256(artifact).hexdigest(), binding["sha256"]):
        raise ValueError("ensemble differs from its prediction profile")

    # This dedicated constructor derives only the public prediction view.  Do
    # not fabricate or accept node privacy/data-scope state in this helper.
    manifest = native_tree_request.public_backend_manifest(request)
    predictor = xgboost_predictor.parse_xgboost_ensemble(artifact, manifest)
    features = request["public_schema"]["features"]
    if not isinstance(features, list) or not features or any(
            not isinstance(name, str) or not name for name in features) or \
            len(set(features)) != len(features):
        raise ValueError("prediction feature contract is invalid")
    return predictor, tuple(features), profile["task"]


def _parse_cell(value):
    if value in ("NA", "NaN"):
        return float("nan")
    if value in ("Inf", "+Inf"):
        return float("inf")
    if value == "-Inf":
        return float("-inf")
    if not value or len(value.encode("utf-8")) > 64:
        raise ValueError("prediction feature is invalid")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("prediction feature is invalid") from exc
    if not math.isfinite(number):
        raise ValueError("prediction feature is invalid")
    return number


def _predict_csv(predictor, features, task, pred_type, data_path, output_path,
                 expected_rows):
    if pred_type not in ("response", "prob") or \
            task == "regression" and pred_type == "prob":
        raise ValueError("prediction type is invalid for this task")
    if type(expected_rows) is not int or not 0 <= expected_rows <= MAX_ROWS:
        raise ValueError("prediction row count is invalid")
    if os.path.getsize(data_path) > MAX_INPUT_BYTES:
        raise ValueError("prediction input exceeds its byte bound")
    if os.path.exists(output_path):
        raise ValueError("prediction output already exists")

    temporary = output_path + ".tmp"
    if os.path.exists(temporary):
        raise ValueError("prediction output temporary already exists")
    rows = 0
    try:
        with open(data_path, "r", encoding="utf-8", newline="") as source, \
                open(temporary, "x", encoding="ascii", newline="") as output:
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            reader = csv.reader(source, strict=True)
            header = next(reader, None)
            if header != list(features):
                raise ValueError("prediction columns differ from the model")
            output.write('{"contract":"%s","predictions":[' % OUTPUT_CONTRACT)
            for record in reader:
                if len(record) != len(features) or rows >= expected_rows:
                    raise ValueError("prediction row shape is invalid")
                prediction = float(predictor.predict_one(
                    [_parse_cell(value) for value in record]))
                if not math.isfinite(prediction):
                    raise ValueError("prediction is non-finite")
                if rows:
                    output.write(",")
                output.write(json.dumps(prediction, allow_nan=False))
                rows += 1
                if rows % 4096 == 0 and output.tell() > MAX_OUTPUT_BYTES:
                    raise ValueError("prediction output exceeds its byte bound")
            if rows != expected_rows:
                raise ValueError("prediction row count differs from its pin")
            output.write(
                '],"task":%s,"type":%s,"version":1}' % (
                    json.dumps(task), json.dumps(pred_type)))
            if output.tell() > MAX_OUTPUT_BYTES:
                raise ValueError("prediction output exceeds its byte bound")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--type", choices=("response", "prob"), required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _arguments(argv)
    try:
        predictor, features, task = _load_predictor(
            args.model, args.profile)
        _predict_csv(
            predictor, features, task, args.type, args.data, args.output,
            args.expected_rows)
    except Exception:
        # The caller receives one fixed diagnostic, never file contents, paths,
        # model fragments, or parser/native exception details.
        sys.stderr.write('{"error":"local XGBoost prediction rejected"}')
        return 2
    sys.stdout.write("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
