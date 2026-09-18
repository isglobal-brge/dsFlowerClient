#!/usr/bin/env python3
"""Assemble only observed public campaign evidence; incomplete cells stay visible."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from segmentation_metrics import envelopes, mean_interval, near_central_flags

SEEDS = (20260919, 20260920, 20260921)
EPSILONS = (1, 4, 8)
DEFAULT_PROVENANCE = Path(__file__).resolve().parents[3] / "inst/extdata/campaign/segmentation/provenance"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_source_split(path, dataset, seed, provenance=None):
    provenance = DEFAULT_PROVENANCE if provenance is None else provenance
    audit = read_json(provenance / (dataset + "-audit.json"))
    if sha256(path) != audit["split_hashes"][str(seed)]:
        raise ValueError("source split differs from archived preregistered split hash")
    split = read_json(path)
    if split.get("seed") != seed:
        raise ValueError("source split seed differs from requested cell")
    return split


def validate_split_provenance(directory, dataset, variant, seed, status, provenance=None):
    source_path = directory / "source-split.json"
    expected = pinned_source_split(source_path, dataset, seed, provenance)
    if status.get("split_sha256") != sha256(source_path):
        raise ValueError("federation source split differs from archived preregistration")
    if variant == "small192":
        expected["train"] = expected["small_train"]
        expected["sites"] = expected["small_sites"]
    elif variant == "heterogeneous":
        expected["sites"] = expected["heterogeneous_sites"]
    expected["variant"] = variant
    split = read_json(directory / "effective-split.json")
    if split != expected:
        raise ValueError("effective split differs from preregistered variant content")
    return split


def released_artifact(directory, status=None):
    """Resolve the actual fit output recorded by federation, including its run child."""
    status = read_json(directory / "federation-status.json") if status is None else status
    output = status.get("output_dir")
    if not isinstance(output, str) or not output:
        raise ValueError("federation did not record its released artifact directory")
    output = Path(output)
    if not output.is_absolute():
        output = directory / output
    output = output.resolve()
    if not output.is_relative_to((directory / "artifact").resolve()):
        raise ValueError("released artifact directory is outside this campaign cell")
    artifact = output / "model.pt"
    if not artifact.is_file():
        raise FileNotFoundError("recorded federation model.pt is missing")
    return artifact


@lru_cache(maxsize=None)
def independent_accounting(sigma, q, steps, epsilon):
    from opacus.accountants import PRVAccountant
    accountant = PRVAccountant()
    accountant.history = [(sigma, q, steps)]
    delta = 1e-5 / (1 + math.exp(epsilon / 2))
    epsilon_add_remove = accountant.get_epsilon(delta=delta)
    value = 2 * epsilon_add_remove
    delta_replace_one = delta * (1 + math.exp(epsilon_add_remove))
    if not math.isfinite(value) or value > epsilon or delta_replace_one > 1e-5:
        raise ValueError("independent full-horizon accountant exceeds budget")
    return {"accountant": "PRVAccountant", "delta_add_remove": delta,
            "epsilon_replace_one": value, "delta_replace_one": delta_replace_one}


def validate_captures(captures, populations, epsilon, expected_hashes=None,
                      expected_source_rows=None, batch_size=16):
    """Require one observation for every site/round, not merely 30 sidecars.

    expected_hashes optionally maps (feature SHA256, target SHA256) to site N.
    expected_source_rows optionally maps the same hash pairs to source row M.
    No source data or secret is returned.
    """
    groups = defaultdict(list)
    for record in captures:
        if record.get("public_fixture_only") is not True:
            raise ValueError("accountant captures must identify public fixtures")
        groups[(record["features_sha256"], record["targets_sha256"])].append(record)
    if len(groups) != 3 or len(captures) != 30:
        raise ValueError("three distinct sites and thirty node-round captures required")
    if expected_hashes is not None and set(groups) != set(expected_hashes):
        raise ValueError("captured effective tensors differ from exact twins")
    mechanisms = []
    for hashes, rows in sorted(groups.items()):
        if sorted(row["round"] for row in rows) != list(range(1, 11)):
            raise ValueError("each site must have exactly one capture for rounds 1 through 10")
        mechanism = rows[0]["mechanism"]
        n = mechanism["accounting_population"]
        source_rows = rows[0].get("source_rows")
        if type(source_rows) is not int or source_rows < n:
            raise ValueError("public source row census must be an integer at least accounting N")
        if expected_source_rows is not None and expected_source_rows.get(hashes) != source_rows:
            raise ValueError("captured source row census differs from cached public source rows")
        steps = math.ceil(n / batch_size)
        expected = {"adjacency": "replace_one", "clipping_norm": 1.,
                    "steps_per_epoch": steps, "sample_rate": 1 / steps,
                    "expected_batch_size": max(1, n // steps),
                    "total_epochs": 30, "total_steps": 30 * steps}
        if n < 1 or any(mechanism.get(key) != value for key, value in expected.items()):
            raise ValueError("captured mechanism differs from preregistered subject schedule")
        if expected_hashes is not None and expected_hashes[hashes] != n:
            raise ValueError("effective tensor population differs from captured accounting N")
        sigma = mechanism["noise_multiplier"]
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive finite noise is mandatory")
        for row in rows:
            history = row["accountant_history"]
            if type(row.get("source_rows")) is not int or row["source_rows"] != source_rows:
                raise ValueError("public source row census changed between rounds")
            if row["mechanism"] != mechanism or row["accountant_type"] != "PRVAccountant":
                raise ValueError("site accountant or mechanism changed between rounds")
            if row["observed_round_steps"] != 3 * steps or sum(h[2] for h in history) != 3 * steps:
                raise ValueError("observed accountant steps differ from logical Poisson batches")
            if any(h[0] != sigma or h[1] != 1 / steps or h[2] <= 0 for h in history):
                raise ValueError("observed accountant history differs from calibrated sigma/q")
        mechanisms.append(dict(mechanism,
            source_rows=source_rows,
            independent_accounting=independent_accounting(sigma, 1 / steps, 30 * steps, epsilon),
            features_sha256=hashes[0], targets_sha256=hashes[1],
            observed_round_steps=[r["observed_round_steps"] for r in sorted(rows, key=lambda r: r["round"])]))
    if Counter(m["accounting_population"] for m in mechanisms) != Counter(populations):
        raise ValueError("captured site populations differ from the public split")
    return mechanisms


def planned_cells():
    primary = [(d, "full", e, s) for d in ("breast", "busbra") for e in EPSILONS for s in SEEDS]
    small = [("busbra", "small192", e, s) for e in EPSILONS for s in SEEDS]
    extensions = [("busbra", v, 8, s) for v in ("bce", "heterogeneous") for s in SEEDS]
    return primary + small + extensions


def load_replicate(directory, dataset, variant, epsilon, seed, provenance=None, batch_size=16):
    if (directory / "INVALIDATED.json").exists():
        raise ValueError("invalid: pipeline defect under investigation")
    execution = {}
    execution_path = directory / "execution-status.json"
    if execution_path.exists():
        execution = read_json(execution_path)
        if execution.get("status") != "executed":
            raise ValueError("Execution incomplete or failed in " + execution.get("phase", "unrecorded phase"))
    status = read_json(directory / "federation-status.json")
    if status.get("status") == "failed" or status.get("cleanup_ok") is not True:
        raise ValueError(status.get("error", "federation or cleanup failed"))
    split_path = directory / "effective-split.json"
    source_path = directory / "source-split.json"
    split = validate_split_provenance(directory, dataset, variant, seed, status, provenance)
    channel = read_json(directory / "channel-b.json")
    twins = read_json(directory / "twins" / "twins.json")
    digest = sha256(split_path)
    if split.get("seed") != seed or split.get("variant", "full") != variant:
        raise ValueError("effective split differs from requested cell")
    train, test = split["train"], split["test"]
    sites = split["sites"]
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise ValueError("public train/test subjects must be unique and disjoint")
    if len(sites) != 3 or Counter(sum(sites, [])) != Counter(train):
        raise ValueError("three disjoint sites must cover every training subject once")
    if variant == "small192" and len(train) != 192:
        raise ValueError("small cohort must contain the preregistered 192 subjects")
    expected_train = 192 if variant == "small192" else {"breast": 205, "busbra": 852}[dataset]
    if len(train) != expected_train or len(test) != {"breast": 51, "busbra": 212}[dataset]:
        raise ValueError("public populations differ from preregistered cohort sizes")
    for document in (channel, twins):
        if document.get("status") != "executed" or document["seed"] != seed or document["split_sha256"] != digest:
            raise ValueError("score/twin provenance differs from effective split")
    if twins["epsilon"] != epsilon or twins["n_train"] != len(train) or twins["n_per_site"] != list(map(len, sites)):
        raise ValueError("twin epsilon or public populations differ from federation")
    if twins["trivial"] != channel["trivial"]:
        raise ValueError("twins and channel B scored different reference masks")
    artifact = sha256(released_artifact(directory, status))
    if artifact != channel["artifact_sha256"] or artifact != status["model_sha256"]:
        raise ValueError("scored artifact checksum differs from released federation artifact")
    capture = directory / "public-capture"
    initial = read_json(capture / "public-initial.json")
    if initial["seed"] != seed:
        raise ValueError("initialization seed differs from preregistered split")
    with np.load(capture / "public-initial-arrays.npz", allow_pickle=False) as arrays:
        tensors = [arrays[str(i)] for i in range(len(arrays.files))]
    if [hashlib.sha256(a.tobytes()).hexdigest() for a in tensors] != initial["tensor_sha256"]:
        raise ValueError("public initial arrays differ from their captured digest")
    initial_hash = hashlib.sha256(b"".join(a.tobytes() for a in tensors)).hexdigest()
    config = initial["config"]
    expected = {"batch-size": batch_size, "local-epochs": 3, "num-server-rounds": 10,
                "learning-rate": .001, "optimizer-name": "adam", "scheduler-name": "none",
                "segmentation-alpha": 1. if variant == "bce" else .5}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("captured optimization differs from preregistration")
    captures = [read_json(p) for p in sorted(capture.glob("accountant-*.json"))]
    mechanisms = validate_captures(captures, list(map(len, sites)), epsilon, batch_size=batch_size)
    if [{k: v for k, v in m.items() if k != "independent_accounting"} for m in mechanisms] != [
            {k: v for k, v in m.items() if k != "independent_accounting"}
            for m in twins["federated_accounting"]]:
        raise ValueError("twins did not validate the same effective tensor/accountant captures")
    pooled = twins["pooled_dp"]["mechanism"]
    pooled_steps = math.ceil(len(train) / batch_size)
    pooled_expected = {"accounting_population": len(train), "adjacency": "replace_one",
                       "clipping_norm": 1., "steps_per_epoch": pooled_steps,
                       "sample_rate": 1 / pooled_steps, "expected_batch_size": len(train) // pooled_steps,
                       "total_epochs": 30, "total_steps": pooled_steps * 30}
    if any(pooled.get(key) != value for key, value in pooled_expected.items()):
        raise ValueError("pooled DP mechanism differs from preregistered subject schedule")
    pooled = dict(pooled, independent_accounting=independent_accounting(
        pooled["noise_multiplier"], pooled["sample_rate"], pooled["total_steps"], epsilon))
    branch_metrics = {name: twins[name]["metrics"] for name in
                      ("pooled_dp", "pooled_nonprivate", "federated_nonprivate")}
    for name in branch_metrics:
        if twins[name]["initial_tensor_sha256"] != initial_hash:
            raise ValueError("twin initialization differs from captured federation initialization")
        if sha256(directory / "twins" / (name + ".npz")) != twins[name]["artifact_sha256"]:
            raise ValueError("twin artifact checksum differs from scored artifact")
    elapsed = status.get("elapsed_s")
    if elapsed is None or elapsed < 0:
        raise ValueError("federation elapsed_s must be recorded by execution tooling")
    peaks = [row.get("peak_cuda_bytes") for row in captures]
    if any(value is None or value < 0 for value in peaks):
        raise ValueError("node GPU memory must be recorded by execution tooling")
    return dict(branch_metrics, seed=seed, epsilon=epsilon, delta=1e-5,
        dataset=dataset, variant=variant, n_train=len(train), n_test=len(test),
        n_per_site=list(map(len, sites)), split_sha256=digest,
        source_split_sha256=sha256(source_path),
        site_mechanisms=mechanisms, pooled_mechanism=pooled,
        federated_dp=channel["metrics"], trivial=channel["trivial"],
        artifact_sha256=artifact, cleanup_ok=True,
        execution_history=[{k: r[k] for k in ("status", "phase", "exit_code", "elapsed_s", "started_at", "recovery") if k in r}
                           for r in (execution.get("previous_execution", {}), execution) if r],
        elapsed_s=elapsed + twins["elapsed_s"],
        timing={"federation_s": elapsed, "twins_s": twins["elapsed_s"]},
        peak_cuda_bytes=max(peaks + [twins["peak_cuda_bytes"]]),
        peak_cuda_scope="maximum observed per process, not simultaneous device total",
        twin_matching={"split_sha256": digest,
                       "initial_parameter_sha256": initial["tensor_sha256"],
                       "config": {key: value for key, value in config.items() if key in expected or
                                  key.startswith("segmentation-") or key in
                                  ("vision-extractor-profile", "image-size", "backbone", "loss-name",
                                   "mask-vocabulary", "model-spec-b64")},
                       "effective_tensors": [{k: m[k] for k in ("features_sha256", "targets_sha256")} for m in mechanisms]},
        twin_artifacts={name: twins[name]["artifact_sha256"] for name in branch_metrics})


def envelope_row(replicate):
    return {"seed": replicate["seed"], "epsilon": replicate["epsilon"],
            "n_train": replicate["n_train"], "n_per_site": replicate["n_per_site"],
            "central_dice": replicate["pooled_nonprivate"]["all"]["dice"],
            "federated_dice": replicate["federated_dp"]["all"]["dice"],
            "trivial_dice": replicate["trivial"]["strongest_dice"]}


def summarize(replicates):
    summaries = {}
    for epsilon in sorted({r["epsilon"] for r in replicates}):
        rows = [r for r in replicates if r["epsilon"] == epsilon]
        if sorted(r["seed"] for r in rows) != list(SEEDS):
            raise ValueError("summary requires all three preregistered seeds")
        summaries[epsilon] = {}
        for branch in ("federated_dp", "pooled_dp", "pooled_nonprivate", "federated_nonprivate"):
            summaries[epsilon][branch] = {}
            for stratum in ("all", "foreground_positive", "empty_reference"):
                values = [r[branch][stratum] for r in rows]
                summaries[epsilon][branch][stratum] = {
                    metric: mean_interval([v[metric] for v in values]) for metric in ("dice", "iou")
                } if all(v is not None for v in values) else None
    return summaries


def assemble(runs, provenance, runtime, protocol, batch_size=16):
    planned = planned_cells()
    directories = {}
    paths = set(runs.rglob("federation-status.json")) | set(runs.rglob("execution-status.json"))
    for directory in sorted({path.parent for path in paths}):
        path = directory / "execution-status.json"
        if not path.exists():
            path = directory / "federation-status.json"
        status = read_json(path)
        if status.get("synthetic") is True:
            continue
        dataset = status.get("dataset", path.parent.name.split("-")[0])
        key = (dataset, status.get("variant", "full"), status["epsilon"], status["seed"])
        if key not in planned:
            raise ValueError("unexpected run outside preregistered matrix: " + str(path))
        if key in directories:
            raise ValueError("duplicate run for preregistered cell: " + str(key))
        directories[key] = (directory, status)
    cells, completed = [], []
    for key in planned:
        cell = dict(zip(("dataset", "variant", "epsilon", "seed"), key))
        if key not in directories:
            cell.update(status="not_executed", reason="No execution or federation status found")
        else:
            try:
                directory, execution = directories[key]
                if execution.get("status") == "failed":
                    federation_path = directory / "federation-status.json"
                    if federation_path.exists():
                        federation = read_json(federation_path)
                        cell["failure_detail"] = {k: federation[k] for k in ("status", "error", "cleanup_ok", "elapsed_s") if k in federation}
                    diagnostic = directory / "public-stalled-control.json"
                    if diagnostic.exists():
                        cell["operational_cancellation"] = read_json(diagnostic)
                    cell["accountant_records"] = len(list((directory / "public-capture").glob("accountant-*.json")))
                    raise ValueError(execution.get("error") or
                        "Execution failed in %s (exit code %s)" %
                        (execution.get("phase", "federation"), execution.get("exit_code", "unrecorded")))
                if execution.get("status") == "running":
                    raise ValueError("Execution incomplete or interrupted in " + execution.get("phase", "unrecorded phase"))
                replicate = load_replicate(directory, *key, provenance=provenance, batch_size=batch_size)
                completed.append(replicate)
                cell["status"] = "executed"
            except (ValueError, KeyError, FileNotFoundError) as error:
                cell.update(status="failed", reason=str(error))
        cells.append(cell)
    base = {"schema": "dsflower-segmentation-evidence-v1", "contract": "pytorch_resnet18_segmentation",
            "task": "segmentation", "nominal_batch_size": batch_size, "protocol_sha256": sha256(protocol),
            "executed_at": datetime.now(timezone.utc).isoformat(), "runtime": runtime}
    documents = {}
    releases = {r["filename"]: r for r in read_json(provenance / "release-manifest.json")["sources"]}
    for dataset in ("breast", "busbra"):
        rows = [r for r in completed if r["dataset"] == dataset and r["variant"] == "full"]
        small = [r for r in completed if r["dataset"] == dataset and r["variant"] == "small192"]
        audit_path = provenance / (dataset + "-audit.json")
        audit = read_json(audit_path)
        document = dict(base, dataset=dataset, cells=[c for c in cells if c["dataset"] == dataset])
        document["provenance"] = {
            "release_sha256": releases["BUSBRA.zip" if dataset == "busbra" else "BrEaST.zip"]["sha256"],
            "license_sha256": releases["CC-BY-4.0.txt"]["sha256"],
            "checkpoint_sha256": releases["resnet18-f37072fd.pth"]["sha256"],
            "subject_mapping": {"audit_sha256": sha256(audit_path),
                                "public_subjects": audit["public_subjects"],
                                "public_source_images": audit["public_source_images"],
                                "patient_mapping": audit["patient_mapping"],
                                "samples_sha256": audit["samples_sha256"],
                                "audit_file": audit_path.name},
            "split_sha256": [r["split_sha256"] for r in rows]}
        if len(rows) == 9:
            checks = envelopes([envelope_row(r) for r in rows],
                               [envelope_row(r) for r in small] if len(small) == 9 else None)
            checks["utility_floor"]["preregistered_primary_cohort"] = dataset == "busbra"
            document.update(status="executed", replicates=rows, envelopes=checks,
                            summaries=summarize(rows))
        else:
            document.update(status="failed" if any(c["status"] == "failed" for c in document["cells"]) else "not_executed",
                            reason="Primary matrix incomplete; envelope evaluation requires all nine cells")
            # Preserve observed results without presenting a completed replicate matrix.
            document["completed_cells"] = rows
        document["additional_replicates"] = [r for r in completed if r["dataset"] == dataset and r["variant"] != "full"]
        document["additional_summaries"] = {}
        document["additional_near_central"] = {}
        for variant, expected_count in (("small192", 9), ("bce", 3), ("heterogeneous", 3)):
            extra = [r for r in document["additional_replicates"] if r["variant"] == variant]
            if extra:
                document["additional_near_central"][variant] = near_central_flags([envelope_row(r) for r in extra])
            if len(extra) == expected_count:
                document["additional_summaries"][variant] = summarize(extra)
        documents[dataset + "-evidence.json"] = document
    all_executed = all(c["status"] == "executed" for c in cells)
    documents["campaign-status.json"] = dict(base, status="executed" if all_executed else
        ("failed" if any(c["status"] == "failed" for c in cells) else "not_executed"), cells=cells,
        reason="All 33 preregistered cells executed" if all_executed else "Retain failed/missing cells; see cohort evidence",
        schema="dsflower-segmentation-campaign-summary-v1",
        promotion="Reviewer decision; execution status does not assert validated utility")
    return documents


def main():
    parser = argparse.ArgumentParser()
    for name in ("runs", "provenance", "runtime", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(16, 64), default=16)
    args = parser.parse_args()
    runtime = read_json(args.runtime)
    for key in ("server_commit", "client_commit", "runner_sha256", "dependencies", "device", "determinism"):
        if key not in runtime:
            raise ValueError("runtime metadata missing: " + key)
    documents = assemble(args.runs, args.provenance, runtime, args.protocol, args.batch_size)
    import jsonschema
    schema = read_json(args.protocol.parent / "evidence-schema.json")
    summary_schema = read_json(args.protocol.parent / "campaign-summary-schema.json")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        jsonschema.validate(document, summary_schema if name == "campaign-status.json" else schema)
        (args.out / name).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: document["status"] for name, document in documents.items()}))


if __name__ == "__main__":
    main()
