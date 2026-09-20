#!/usr/bin/env python3
"""Read-only audit of executed public cohort releases; no training or DP calls.

Write a new report under WORKSPACE/runtime with --out. Partial coverage is
explicit; zero selected cells exits 2, any failed audit exits 1. This supports
human envelope investigations and never edits evidence or marks flags reviewed.
"""
import argparse
import base64
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
import torch

from central_and_score import build, client_app, sha, survival
from metrics import score
from validate_completion import CELLS, require, validate_cell

TOLERANCES = {"c_index_abs": 1e-12, "heldout_nll_abs": 1e-8, "heldout_nll_rel": 1e-7}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def vector(value):
    return value if isinstance(value, list) else [value]


def split_metadata(value):
    # R's historical synthetic evidence unboxed its single feature/bound.
    value = json.loads(canonical(value))
    value["features"] = vector(value["features"])
    for key in ("lower", "upper"):
        value["feature_bounds"][key] = vector(value["feature_bounds"][key])
    return value


def runner_hash():
    """Same path/newline/content/NUL hash as R .compute_local_runner_hash()."""
    root = Path(__file__).resolve().parents[3] / "inst/flower_app/dsflower_runner"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            digest.update(path.relative_to(root).as_posix().encode() + b"\n" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def audit_cell(record, run_dir, split_dir, current_runner_hash):
    require(record["dataset"]["public_fixture"] is True, "only explicitly public fixtures may be read")
    validate_cell(record)
    require(record["status"] == "executed", "only executed releases may be audited")
    require(record["runner_sha256"] == current_runner_hash, "auditor runner differs from executed runner")
    cfg = record["public_config"]
    require(canonical(json.loads((run_dir / "config.json").read_text())) == canonical(cfg), "run configuration drift")
    meta = split_metadata(record["dataset"])
    frozen = split_metadata(json.loads((split_dir / "split.json").read_text()))
    require(canonical(frozen) == canonical(meta), "frozen split manifest differs from archived metadata")
    split_hashes = {"train.csv": meta["train_sha256"], "test.csv": meta["test_sha256"]}
    split_hashes.update({f"site{s['site']}.csv": s["split_sha256"] for s in meta["sites"]})
    for name, expected in split_hashes.items():
        require(sha(split_dir / name) == expected, f"frozen public split hash differs: {name}")

    models = list((run_dir / "federation/artifact").glob("*/model.pt"))
    require(len(models) == 1, "expected exactly one saved release model.pt")
    model_path = models[0]
    checksum = sha(model_path)
    require(checksum == record["artifact_checksum"] == record["results"]["model_sha256"] ==
            record["federation"]["model_sha256"], "released artifact checksum differs")
    artifact = json.loads(model_path.with_name("metadata.json").read_text())
    conf = survival.config_from_run(cfg, cfg["loss-name"])
    spec = json.loads(base64.b64decode(cfg["model-spec-b64"], validate=True))
    artifact_conf = survival.validate_survival_config(artifact["survival_config"], cfg["loss-name"])
    require(canonical(artifact_conf) == canonical(conf), "artifact survival configuration drift")
    require(canonical(artifact["model_spec"]) == canonical(spec), "artifact architecture drift")
    require(artifact["loss_name"] == cfg["loss-name"] and artifact["model"] == record["contract"], "artifact contract drift")
    require(vector(artifact["features"]) == meta["features"] and artifact["target"] == ["time", "event"], "artifact role/order drift")
    bounds = {"lower": vector(artifact["feature_lower"]), "upper": vector(artifact["feature_upper"])}
    require(canonical(bounds) == canonical(meta["feature_bounds"]), "artifact feature-bound drift")
    for wire, name in (("learning-rate", "learning_rate"), ("batch-size", "batch_size"),
                       ("local-epochs", "local_epochs"), ("optimizer-name", "optimizer"),
                       ("weight-decay", "weight_decay"), ("l1-penalty", "l1_penalty"),
                       ("optimizer-momentum", "momentum"), ("optimizer-nesterov", "nesterov"),
                       ("scheduler-name", "scheduler")):
        require(canonical(artifact["model_params"][name]) == canonical(cfg[wire]), "artifact training configuration drift: " + wire)
    require(artifact["num_rounds"] == artifact["requested_num_rounds"] == cfg["num-server-rounds"] and
            canonical(artifact["available_rounds"]) == canonical(list(range(1, cfg["num-server-rounds"] + 1))) and
            artifact["n_clients"] == 3 and artifact["strategy"] == "FedAvg" and
            artifact["framework"] == "pytorch" and artifact["track"] == "neural" and
            artifact["available"] is True and artifact["status"] == "success" and
            artifact["privacy"] == "server-enforced-dp", "artifact federation metadata differs")

    def ids(name, expected):
        values = pd.read_csv(split_dir / name, usecols=["subject_id"], dtype={"subject_id": str})["subject_id"]
        require(len(values) == expected and values.notna().all() and values.is_unique, "public subject census differs: " + name)
        return set(values)
    train_ids = ids("train.csv", meta["n_train"])
    test_ids = ids("test.csv", meta["n_test"])
    require(train_ids.isdisjoint(test_ids), "training and held-out subjects overlap")
    site_ids = [ids(f"site{s['site']}.csv", s["n_subjects"]) for s in meta["sites"]]
    require(set.union(*site_ids) == train_ids and sum(map(len, site_ids)) == len(train_ids), "site partition differs from training subjects")
    frame = pd.read_csv(split_dir / "test.csv", float_precision="round_trip")
    effective = dict(cfg, **{"feature-bounds": meta["feature_bounds"]})
    x = client_app._apply_feature_bounds(frame[meta["features"]].to_numpy(dtype=np.float32), effective)
    require(np.isfinite(x).all(), "nonfinite encoded public held-out features")
    model = build(effective).cpu()
    initial = {name: value.detach().double().clone() for name, value in model.named_parameters()}
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=True)
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "nonfinite released parameters")
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(x)).cpu().numpy()
    recomputed = score(output, frame, conf)
    recorded = record["results"]["federated_dp"]
    require(math.isclose(recomputed["c_index"], recorded["c_index"], rel_tol=0, abs_tol=TOLERANCES["c_index_abs"]),
            "held-out concordance differs from archived score")
    require(math.isclose(recomputed["heldout_nll"], recorded["heldout_nll"],
                         rel_tol=TOLERANCES["heldout_nll_rel"], abs_tol=TOLERANCES["heldout_nll_abs"]),
            "held-out NLL differs from archived score")
    for key in ("n_evaluated_public_subjects", "n_invalid_public_subjects"):
        require(recomputed[key] == recorded[key], "held-out score census differs")
    distance = math.sqrt(sum((p.detach().double() - initial[name]).square().sum().item()
                             for name, p in model.named_parameters()))
    initial_norm = math.sqrt(sum(p.square().sum().item() for p in initial.values()))
    return {"status": "passed", "artifact_sha256": checksum, "split_sha256": split_hashes,
            "checks": {name: True for name in ("public_fixture", "evidence_schema", "runner_identity", "configuration",
                        "artifact_identity", "frozen_splits", "subject_partition", "heldout_scores")},
            "recomputed_federated_dp": recomputed,
            "distance_from_public_initializer": {"seed": 0, "l2": distance, "initializer_l2": initial_norm,
                "relative_l2": distance / initial_norm if initial_norm else None,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "interpretation": "Diagnostic only; neither a privacy proof nor a utility-floor test."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, destination = args.workspace.resolve(), args.out.resolve()
    require(destination.is_relative_to(root / "runtime"), "audit report must be separate under workspace/runtime")
    require(not destination.exists(), "audit reports are immutable; choose a new output filename")
    archive = root / "dsFlowerClient/inst/extdata/campaign/survival"
    require(archive.is_dir(), "survival evidence archive is missing")
    torch.set_num_threads(1)
    started = datetime.datetime.now(datetime.timezone.utc)
    current_hash = runner_hash()
    counts = collections.Counter(dict.fromkeys(("archive_records", "summaries_skipped", "failed_attempts_skipped",
                                               "synthetic_cells_skipped", "executed_cohort_cells_selected"), 0))
    audited, identities = [], set()
    for path in sorted(archive.glob("*.json")):
        counts["archive_records"] += 1
        try:
            record = json.loads(path.read_text())
            if record.get("record_type") == "summary":
                counts["summaries_skipped"] += 1
                continue
            if record["status"] == "failed":
                counts["failed_attempts_skipped"] += 1
                continue
            require(record["status"] == "executed", "archive record is neither executed nor failed")
            if record["dataset"]["dataset"] == "synthetic-public":
                counts["synthetic_cells_skipped"] += 1
                continue
            counts["executed_cohort_cells_selected"] += 1
            meta = record["dataset"]
            key = (meta["dataset"], meta["subset"], record["variant"], record["epsilon"], meta["seed"])
            require(key in CELLS and key not in identities, "unexpected or duplicate cohort identity")
            identity = f"{key[0]}-{key[1]}-{key[2]}-eps{key[3]}-seed{key[4]}"
            require(path.name == f"cell-{identity}.json", "cell filename differs from archived identity")
            identities.add(key)
            result = audit_cell(record, root / "runtime/runs" / identity,
                                root / "data/survival/splits" / f"{key[0]}-{key[1]}-{key[4]}", current_hash)
            result.update(cell=path.name, evidence_sha256=sha(path), identity=identity)
            audited.append(result)
        except (KeyError, ValueError, TypeError, OSError, RuntimeError) as error:
            audited.append({"cell": path.name, "status": "failed", "error": str(error)})
    failed = sum(r["status"] == "failed" for r in audited)
    passed = len(audited) - failed
    report = {"schema_version": 1, "record_type": "public_artifact_audit", "task": "survival",
              "status": "failed" if failed else "passed" if passed else "empty",
              "started_utc": started.isoformat(), "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": dict(counts, audited=len(audited), passed=passed, failed=failed, expected_cohort_cells=len(CELLS),
                            all_cohort_artifacts_passed=identities == CELLS and failed == 0),
              "runner_sha256": current_hash,
              "tool_sha256": {name: sha(Path(__file__).with_name(name)) for name in
                              ("audit_artifacts.py", "central_and_score.py", "metrics.py", "validate_completion.py")},
              "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                           "pandas": pd.__version__, "scipy": scipy.__version__,
                           "platform": platform.platform(), "evaluation_device": "cpu"},
              "score_tolerances": TOLERANCES, "cells": audited,
              "interpretation": "Read-only public artifact audit. Partial coverage is not campaign completion; no privacy proof or automatic envelope review."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("SURVIVAL_ARTIFACT_AUDIT", report["status"], json.dumps(report["scope"], sort_keys=True))
    return 1 if failed else 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
