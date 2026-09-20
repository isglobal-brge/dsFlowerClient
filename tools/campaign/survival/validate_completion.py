#!/usr/bin/env python3
"""Fail closed unless the preregistered survival campaign evidence is complete.

This checks archived records and recomputes summaries; it is not reviewer promotion
or proof that a reported artifact was executed. Failed attempts remain in the archive
but never substitute for an executed cell. Failed utility floors are valid findings.
"""
import argparse
import base64
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import re

SEEDS = (1101, 1102, 1103)
VARIANTS = ("weibull", "lognormal", "hazard")
GROUPS = {(dataset, subset, variant)
          for dataset, subsets in (("support2", ("full", "small600", "heterogeneous")),
                                   ("lung1", ("full",)))
          for subset in subsets for variant in VARIANTS}
CELLS = {group + (epsilon, seed) for group in GROUPS
         for epsilon in ((8,) if group[1] == "heterogeneous" else (1, 4, 8))
         for seed in SEEDS}
EDGES = [0, 7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 540, 730, 1095, 1460, 1825]
COHORTS = {
    "support2": (9105, "9da794bbd5c3a6a816e677cc17535e58c122d9ef4cbefd404489330a9f9cd2de", "CC BY 4.0",
                 ["age", "male", "comorbidities", "diabetes", "dementia", "cancer_0", "cancer_1"] +
                 [f"disease_{i}" for i in range(8)], [100, 1, 10] + [1] * 12),
    "lung1": (422, "132f72b58b9660bf5e6b24b9817b335f1896360bc253e1f2034a3ffee593e6fd", "CC BY-NC 3.0",
              ["age", "male", "t_stage", "n_stage", "m_stage"] + [f"histology_{i}" for i in range(4)],
              [100, 1, 4, 3, 1, 1, 1, 1, 1]),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{%d}" % length, value) is not None


def timestamp(value):
    require(nonempty(value) and value.endswith(("Z", "+00:00")), "timestamp must be UTC")
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def same(actual, expected, path="summary"):
    """Compare nested recomputed envelopes, retaining strict bool/number distinction."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{path}: fields differ")
        for key, value in expected.items():
            same(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{path}: length differs")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{path}[{index}]")
    elif finite(expected):
        require(finite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                f"{path}: differs from executed replicates")
    else:
        require(type(actual) is type(expected) and actual == expected, f"{path}: value differs")


def mechanism(value, n, cfg, epsilon, delta):
    steps = math.ceil(n / cfg["batch-size"])
    pins = dict(adjacency="replace_one", clipping_norm=1, accounting_population=n,
                steps_per_epoch=steps, sample_rate=1 / steps,
                expected_batch_size=max(1, n // steps),
                total_epochs=cfg["num-server-rounds"] * cfg["local-epochs"],
                total_steps=steps * cfg["num-server-rounds"] * cfg["local-epochs"])
    for key, expected in pins.items():
        same(value[key], expected, "mechanism." + key)
    require(finite(value["noise_multiplier"]) and value["noise_multiplier"] > 0, "noise must be positive")
    pins["noise_multiplier"] = float(value["noise_multiplier"])
    # The runtime records clipping_norm as a float in its canonical policy.
    pins["clipping_norm"] = 1.0
    payload = json.dumps(pins, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    require(value["policy_hash"] == hashlib.sha256(b"dsflower/effective-dpsgd-policy/v1\x00" + payload).hexdigest(),
            "effective mechanism policy hash differs")
    for name, limit in (("epsilon", epsilon + 1e-8), ("delta", delta + 1e-12)):
        v = value["independently_recomputed_replace_one_" + name]
        require(finite(v) and 0 < v <= limit, "independently recomputed privacy bound failed")
    require(value["verification_accountant"] == "PRV", "independent accountant missing")
    require(nonempty(value["calibration"]), "calibration provenance missing")


def validate_cell(record):
    require(record["schema_version"] == 1 and record["task"] == "survival", "unknown evidence schema")
    require(record["status"] in ("failed", "executed"), "unexecuted attempt in archive")
    meta = record["dataset"]
    require(meta["public_fixture"] is True, "campaign inputs must be public")
    require(record["delta"] == 1e-5 and record["clip"] == 1 and record["epsilon"] in (1, 4, 8), "privacy pins differ")
    require(record["privacy_unit"] == "patient" and record["adjacency"] == "replace_one" and record["site_count"] == 3,
            "privacy unit, adjacency or site count differs")
    for value in (meta["source"]["sha256"], meta["protocol_sha256"]):
        require(digest(value), "release or protocol hash missing")
    if record["status"] == "failed":
        require(nonempty(record["error"]) and record.get("results") is None, "failure must have an error and no scores")
        require(isinstance(record["cleanup_ok"], bool), "failure cleanup status missing")
        return
    require(record["record_type"] == "cell" and record["cleanup_ok"] is True, "executed cell or cleanup missing")
    start, end = timestamp(record["started_utc"]), timestamp(record["finished_utc"])
    require(end >= start and finite(record["elapsed_s"]) and record["elapsed_s"] > 0, "execution timing invalid")
    require(digest(record["campaign_tools_commit"], 40), "campaign tools revision missing")
    for repo in ("dsFlower", "dsFlowerClient"):
        require(digest(record["package_commits"][repo], 40), "package revision missing")
    require(digest(record["runner_sha256"]), "runner hash missing")
    build = record["installed_build"]
    require(build["commits"] == record["package_commits"] and build["runner_sha256"] == record["runner_sha256"],
            "installed build differs from reported revisions")
    require(timestamp(build["built_utc"]) <= start, "installed build postdates execution")
    for name in ("dsFlower", "dsFlowerClient", "R", "DSI", "DSLite", "dsBase", "resourcer"):
        require(nonempty(record["package_versions"][name]), "R package version missing")
    require(build["package_versions"] == [record["package_versions"][name] for name in ("dsFlower", "dsFlowerClient")],
            "installed package versions differ")
    synthetic = meta["dataset"] == "synthetic-public"
    for field in ("release", "licence"):
        require(nonempty(meta["source"][field]), "dataset release/licence missing")
    if not synthetic:
        require(meta["dataset"] in COHORTS, "unexpected cohort")
        n, release, licence, features, upper = COHORTS[meta["dataset"]]
        require(meta["n_original_rows"] == n and meta["source"]["sha256"] == release and
                meta["source"]["licence"] == licence, "preregistered cohort release differs")
        require(meta["n_test"] == n - math.floor(.8 * n) and
                meta["n_train"] == (600 if meta["subset"] == "small600" else math.floor(.8 * n)), "80/20 subject split differs")
        same(meta["features"], features, "baseline features")
        same(meta["feature_bounds"], {"lower": [0] * len(features), "upper": upper}, "baseline bounds")
        for field in ("url", "licence_url", "attribution"):
            require(nonempty(meta["source"][field]), "dataset source provenance missing")
        require(nonempty(meta["subject_provenance"]) and nonempty(meta["split_rule"]), "subject/split provenance missing")
    for key in ("train_sha256", "test_sha256"):
        require(digest(meta[key]), "split hash missing")
    require(meta["n_train"] > 0 and meta["n_test"] > 0, "public split census missing")
    require(len(meta["sites"]) == 3 and sorted(s["site"] for s in meta["sites"]) == [1, 2, 3], "site census differs")
    require(sum(s["n_subjects"] for s in meta["sites"]) == meta["n_train"], "site subject census differs from train N")
    for site in meta["sites"]:
        require(site["n_subjects"] > 0 and site["source_rows"] == site["n_subjects"] and digest(site["split_sha256"]),
                "subject-level site split invalid")
    cfg = record["public_config"]
    features = meta["features"] if isinstance(meta["features"], list) else [meta["features"]]
    require(cfg["num-features"] == len(features), "model input width differs from fixed feature roles")
    for key, value in {"learning-rate": .05, "optimizer-name": "sgd", "optimizer-momentum": 0,
                       "weight-decay": 0, "l1-penalty": 0, "scheduler-name": "none", "batch-size": 128}.items():
        same(cfg[key], value, "public_config." + key)
    require(cfg["num-server-rounds"] >= 2 and cfg["local-epochs"] >= 1, "synthetic integration horizon too short")
    if not synthetic:
        require(cfg["num-server-rounds"] == 10 and cfg["local-epochs"] == 2, "preregistered cohort horizon differs")
    survival = json.loads(base64.b64decode(cfg["survival-config-b64"], validate=True))
    expected = dict(schema_version=1, time_unit="days", time_origin="baseline", t_min=1, horizon=1825)
    variant = record["variant"]
    require(variant in VARIANTS, "unexpected survival variant")
    if variant == "hazard":
        expected["edges"] = EDGES
        require(record["contract"] == "pytorch_discrete_hazard" and cfg["loss-name"] == "discrete_hazard_nll", "hazard contract differs")
    else:
        expected.update(time_scale=365, distribution=variant, dispersion=1)
        require(record["contract"] == "pytorch_aft" and cfg["loss-name"] == "aft_" + variant + "_nll", "AFT contract differs")
    same(survival, expected, "survival_config")
    spec = json.loads(base64.b64decode(cfg["model-spec-b64"], validate=True))
    same(spec, {"kind": "sequential", "layers": [{"op": "linear", "out": "@out"}]}, "architecture")
    conventions = record["score_conventions"]
    require(conventions["time_ties"] == "excluded" and conventions["risk_ties"] == .5 and "reversed" in conventions["gap"],
            "concordance or gap convention missing")
    require(conventions["hazard_risk"] == "negative left-endpoint restricted mean", "hazard risk convention differs")
    semantics = record["outcome_semantics"]
    require(semantics["target_order"] == ["time", "event"] and semantics["event"] == 1 and semantics["censored"] == 0,
            "outcome roles missing")
    for key in ("time_unit", "baseline", "administrative_censor", "invalid", "preprocessing", "interval_convention"):
        require(nonempty(semantics[key]), "outcome semantics missing")
    require(nonempty(record["mechanism_provenance"]) and record["evaluation"] == "channel B, public held-out subjects only",
            "evaluation/mechanism provenance missing")
    require(nonempty(record["topology"]), "federation topology missing")
    result, federation = record["results"], record["federation"]
    require(result["seed"] == meta["seed"] and result["n_train"] == meta["n_train"] and
            result["minimum_site_n"] == min(s["n_subjects"] for s in meta["sites"]), "result census/seed differs")
    require(federation["n_clients"] == 3 and federation["n_failures"] == 0 and
            federation["n_rounds_run"] == cfg["num-server-rounds"] and federation["cleanup_ok"] is True, "federation incomplete")
    require(digest(record["artifact_checksum"]) and record["artifact_checksum"] == result["model_sha256"] == federation["model_sha256"],
            "released model checksums differ")
    for branch in ("central", "central_dp", "null", "federated_dp"):
        score = result[branch]
        require(finite(score["c_index"]) and 0 <= score["c_index"] <= 1 and finite(score["heldout_nll"]), "finite executed score missing")
        require(score["n_evaluated_public_subjects"] > 0 and score["n_invalid_public_subjects"] >= 0 and
                score["n_evaluated_public_subjects"] + score["n_invalid_public_subjects"] == meta["n_test"], "public score census differs")
    sites = result["site_mechanisms"]
    require(len(sites) == 3 and sorted(s["site"] for s in sites) == [1, 2, 3], "site mechanisms missing")
    for site in sites:
        n = next(s["n_subjects"] for s in meta["sites"] if s["site"] == site["site"])
        mechanism(site, n, cfg, record["epsilon"], record["delta"])
    mechanism(result["pooled_mechanism"], meta["n_train"], cfg, record["epsilon"], record["delta"])
    for name in ("python", "torch", "opacus", "flwr", "numpy", "scipy", "pandas", "platform",
                 "nonprivate_twin_device", "dp_twin_device", "federation_device_rule"):
        require(nonempty(result["versions"][name]), "runtime version/device missing")
    require(result["versions"]["deterministic_algorithms"] is True and isinstance(result["versions"]["cuda_available"], bool),
            "execution backend settings missing")
    twin = result["twin_matching"]
    require(twin["architecture_loss_preprocessing_initialization_optimizer_schedule"] == "exact" and
            twin["initialization_seed"] == 0 and twin["pooled_epochs"] == cfg["num-server-rounds"] * cfg["local-epochs"] and
            nonempty(twin["differences"]), "central twin matching missing")
    require(finite(result["elapsed_s"]) and result["elapsed_s"] > 0, "twin timing missing")
    require(finite(result["max_rss_native_units"]) and result["max_rss_native_units"] > 0, "memory measurement missing")
    memory = result["memory_measurement"]
    require(nonempty(memory["scope"]) and memory["native_unit"] in ("bytes", "KiB") and
            isinstance(memory["federation_peak_memory_measured"], bool), "memory scope, units or limitations missing")


def flagged_keys(env):
    keys = {f"epsilon_envelope_{x['from']}_{x['to']}" for x in env["epsilon_envelope"] if x["flag"]}
    if env["small_n_trend"] and env["small_n_trend"]["flag"]:
        keys.add("small_n_trend")
    keys.update(f"near_central_{x['epsilon']}_{x['seed']}" for x in env["near_central"]
                if x["historic_flag"] or x["minimum_site_companion_flag"])
    return keys


def validate_completion(records):
    from metrics import envelopes
    cells, synthetic, summaries, failures = {}, {}, [], []
    for name, record in records:
        if record.get("record_type") == "summary":
            summaries.append(record)
            continue
        try:
            validate_cell(record)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name}: {error}") from error
        if record["status"] == "failed":
            failures.append({"file": name, "reason": record["error"]})
            continue
        meta = record["dataset"]
        if meta["dataset"] == "synthetic-public":
            key = record["variant"]
            require(meta["seed"] == 1101 and meta["subset"] == "synthetic" and record["epsilon"] == 8, "synthetic pins differ")
            require(key not in synthetic, "duplicate executed synthetic variant")
            synthetic[key] = record
        else:
            key = (meta["dataset"], meta["subset"], record["variant"], record["epsilon"], meta["seed"])
            require(key in CELLS and key not in cells, "unexpected or duplicate executed cohort cell")
            cells[key] = record
    require(set(cells) == CELLS, f"expected exactly 90 cohort cells; found {len(cells)}, missing {len(CELLS - set(cells))}")
    require(set(synthetic) == set(VARIANTS), "expected exactly three executed synthetic variants")
    require(max(timestamp(r["finished_utc"]) for r in synthetic.values()) <=
            min(timestamp(r["started_utc"]) for r in cells.values()), "synthetic gates must precede cohort scoring")
    require(len(summaries) == 1, "expected one executed summary")
    summary = summaries[0]
    require(summary["schema_version"] == 1 and summary["task"] == "survival" and summary["status"] == "executed" and
            summary["expected_matrix_cells"] == 90, "summary is incomplete")
    require("revers" in summary["envelope_interpretation"], "summary omits reversed-sign note")
    timestamp(summary["date_utc"])
    same(sorted(summary["failed_attempts"], key=lambda x: x["file"]), sorted(failures, key=lambda x: x["file"]), "failed_attempts")
    groups = summary["groups"]
    require(len(groups) == 12 and {(g["dataset"], g["subset"], g["variant"]) for g in groups} == GROUPS,
            "summary must contain exactly the 12 preregistered groups")
    split_identity = {}
    for key, record in cells.items():
        split = (key[0], key[1], key[4])
        meta = record["dataset"]
        require(split_identity.setdefault(split, meta) == meta, "matched cells use different public splits or preprocessing")
    require(len({r["dataset"]["protocol_sha256"] for r in list(cells.values()) + list(synthetic.values())}) == 1,
            "executed cells cite different protocols")
    for group in groups:
        key = (group["dataset"], group["subset"], group["variant"])
        require(group["status"] == "executed", "summary group is incomplete")
        by_epsilon = collections.defaultdict(list)
        for cell_key in sorted(cells):
            if cell_key[:3] == key:
                by_epsilon[cell_key[3]].append(cells[cell_key]["results"])
        expected = envelopes(by_epsilon, primary=key[:2] == ("support2", "full"), small_n=key[1] == "small600")
        same(group["envelopes"], expected, "/".join(key))
        for flag in flagged_keys(expected):
            investigation = group["investigations"][flag]
            require(investigation["status"] == "reviewed" and nonempty(investigation["explanation"]),
                    f"{'/'.join(key)}: {flag} requires a reviewed explanation")
    return {"cohort_cells": len(cells), "synthetic_cells": len(synthetic), "groups": len(groups),
            "failed_utility_floors": sum(g["envelopes"]["utility_floor"] is not None and
                                        not g["envelopes"]["utility_floor"]["pass"] for g in groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    try:
        records = [(p.name, json.loads(p.read_text())) for p in sorted(args.evidence.glob("*.json"))]
        result = validate_completion(records)
    except (KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"SURVIVAL_COMPLETION_FAILED: {error}\n")
    print("SURVIVAL_COMPLETION_PASSED", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
