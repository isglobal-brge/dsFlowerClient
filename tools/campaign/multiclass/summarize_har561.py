#!/usr/bin/env python3
"""Validate HAR561 evidence without rescoring; retain the original CTG summary."""
import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


def mean_sd(values):
    assert all(math.isfinite(value) for value in values)
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values))


def verify_statistics(recorded, recomputed):
    assert recorded.keys() == recomputed.keys()
    for key, value in recomputed.items():
        if isinstance(value, dict):
            verify_statistics(recorded[key], value)
        elif isinstance(value, bool):
            assert recorded[key] is value, f"Incorrect diagnostic: {key}"
        else:
            assert math.isclose(recorded[key], value, rel_tol=0, abs_tol=1e-14), key


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.utcoffset() is not None, "Timestamp must declare its timezone"
    return parsed


ap = argparse.ArgumentParser()
ap.add_argument("evidence", type=Path)
args = ap.parse_args()
summary_path = args.evidence / "summary.json"
previous = json.loads(summary_path.read_text())
ctg = previous["datasets"][0] if "datasets" in previous else previous
assert ctg["dataset"]["uci_id"] == 193
assert ctg["dataset"]["n_total"] == 2126
assert [row["epsilon"] for row in ctg["cells"]] == [1, 4, 8]
for row in ctg["cells"]:
    path = args.evidence / row["evidence_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["evidence_sha256"]

defaults = dict(momentum=0, nesterov=False, learning_rate=0.1, batch_size=32,
                local_epochs=1, weight_decay=0, l1_penalty=0, optimizer="sgd",
                scheduler="none", hidden_layers=[], n_classes=6)
levels = [str(i) for i in range(1, 7)]
archive_sha256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
inner_archive_sha256 = "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0"
rows = []
paired = None
fixed = None
official_subjects = None
central_fit_seconds = {}
federated_fit_seconds = []
for epsilon in (1, 4, 8):
    path = args.evidence / f"har561_pytorch_multiclass_eps{epsilon}.json"
    cell = json.loads(path.read_text())
    assert cell["schema"] == "dsflower-campaign-v2"
    assert cell["cell_id"] == path.stem
    assert cell["contract"] == "pytorch_multiclass"
    assert cell["versions"]["dsflower"] == cell["versions"]["dsflowerclient"] == "0.5.0"
    dataset = cell["dataset"]
    assert dataset["uci_id"] == 240 and dataset["n_total"] == 10299
    assert dataset["n_train"] == 7352 and dataset["n_test"] == 2947
    features = dataset["features"]
    assert len(features) == len(set(features)) == 561
    assert "subject" not in features and "target" not in features
    assert dataset["download_sha256"] == archive_sha256
    assert dataset["inner_archive_sha256"] == inner_archive_sha256
    protocol = cell["protocol"]
    assert protocol["archive_sha256"] == archive_sha256
    assert protocol["inner_archive_sha256"] == inner_archive_sha256
    scoring = cell["scoring"]
    assert scoring["test_table_loads"] == 1
    assert scoring["central_fits"] == 3 and scoring["federated_fits"] == 9
    assert scoring["central_scores_reused_across_epsilon"] is True
    assert scoring["changed_settings_after_scoring"] is False
    assert (timestamp(protocol["declared_at_utc"]) <= timestamp(scoring["training_completed_at_utc"]) <=
            timestamp(scoring["started_at_utc"]) <= timestamp(cell["generated_at"]))
    privacy = cell["privacy"]
    assert privacy["epsilon"] == epsilon and privacy["delta"] == 1e-6
    assert privacy["unit"] == "patient" and privacy["patient_column"] == "subject"
    assert privacy["clipping_norm"] == 1 and privacy["adjacency"] == "replace_one"
    assert privacy["guarantee_scope"] == "per-training"
    assert cell["rounds"] == 5
    assert cell["model_params_requested"] == {"n_classes": 6}
    assert cell["model_params_resolved"] == defaults
    current_fixed = (dataset, protocol, cell["versions"], cell["tooling_sha256"], scoring)
    if fixed is None:
        fixed = current_fixed
    else:
        assert fixed == current_fixed, "Dataset, protocol, environment, tooling or scoring changed across epsilon"
    tooling = cell["tooling_sha256"]
    assert {"har561.R", "har561_campaign.R", "run_har561.R", "har561_protocol.json"} <= tooling.keys()
    for name, expected_hash in tooling.items():
        source = Path(__file__).resolve().parent / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash, name
    reps = cell["per_replicate"]
    assert [rep["seed"] for rep in reps] == [20260820, 20260821, 20260822]
    current = [(rep["split"], rep["central"], rep["trivial"], rep["feature_bounds"]) for rep in reps]
    if paired is None:
        paired = current
    else:
        assert paired == current, "Split, bounds or baselines changed across epsilon"
    assert dataset["n_per_site"] == reps[0]["split"]["n_per_site"]
    for rep in reps:
        split = rep["split"]
        assert split["n_train"] == 7352 and split["n_test"] == 2947
        assert len(split["n_per_site"]) == 3 and sum(split["n_per_site"]) == 7352
        assert split["training_effective_units"] == [7, 7, 7]
        train_subjects, test_subjects = set(split["train_subjects"]), set(split["test_subjects"])
        assert len(split["train_subjects"]) == len(train_subjects) == 21
        assert len(split["test_subjects"]) == len(test_subjects) == 9
        assert not train_subjects & test_subjects
        assert train_subjects | test_subjects == set(range(1, 31))
        if official_subjects is None:
            official_subjects = (train_subjects, test_subjects)
        else:
            assert official_subjects == (train_subjects, test_subjects)
        sites = split["site_subjects"]
        assert len(sites) == 3 and all(len(site) == len(set(site)) == 7 for site in sites)
        assert set().union(*map(set, sites)) == train_subjects
        assert all(not set(sites[i]) & set(sites[j]) for i in range(3) for j in range(i))
        assert len(split["class_counts_sites"]) == 3
        for counts, size in [(split["class_counts_train"], 7352),
                             (split["class_counts_test"], 2947),
                             *zip(split["class_counts_sites"], split["n_per_site"])]:
            assert set(counts) == set(levels)
            assert all(count > 0 for count in counts.values()) and sum(counts.values()) == size
        assert all(sum(site[level] for site in split["class_counts_sites"]) ==
                   split["class_counts_train"][level] for level in levels)
        assert rep["history"]["n_clients"] == 3
        assert rep["history"]["n_rounds"] == 5 and rep["history"]["n_failures"] == 0
        assert rep["cleanup_ok"] is True and rep["central"]["convergence"] == 0
        elapsed = rep["wall_clock"]
        assert elapsed["central_reused_across_epsilon"] is True
        assert all(math.isfinite(elapsed[key]) and elapsed[key] >= 0
                   for key in ("central_fit_s", "federated_s", "total_s"))
        assert math.isclose(elapsed["total_s"], elapsed["central_fit_s"] + elapsed["federated_s"],
                            rel_tol=0, abs_tol=1e-9)
        if rep["seed"] in central_fit_seconds:
            assert central_fit_seconds[rep["seed"]] == elapsed["central_fit_s"]
        else:
            central_fit_seconds[rep["seed"]] = elapsed["central_fit_s"]
        federated_fit_seconds.append(elapsed["federated_s"])
        bounds = rep["feature_bounds"]
        assert bounds == dict(lower=[-1] * 561, upper=[1] * 561)
        metadata = rep["released_metadata"]
        assert metadata["model"] == "pytorch_multiclass" and metadata["model_params"] == defaults
        assert metadata["target_levels"] == levels and metadata["features"] == features
        assert metadata["feature_lower"] == bounds["lower"] and metadata["feature_upper"] == bounds["upper"]
        assert metadata["available_rounds"] == [1, 2, 3, 4, 5]
        assert metadata["n_clients"] == 3 and metadata["num_rounds"] == metadata["requested_num_rounds"] == 5
        assert metadata["privacy"] == "server-enforced-dp" and metadata["status"] == "success"
        assert len(rep["node_privacy"]) == 3
        for node in rep["node_privacy"].values():
            for policy in (node["reported_policy"], node["contract"]):
                assert policy["per_training_epsilon"] == epsilon and policy["per_training_delta"] == 1e-6
                assert policy["dp_unit"] == "patient" and policy["patient_column"] == "subject"
                assert policy["adjacency"] == "replace_one"
            assert node["clipping_norm"] == 1
            assert node["runner_sha256"] == cell["versions"]["runner_sha256"]
        for branch in ("central", "federated_dp", "trivial"):
            assert 0 <= rep[branch]["macro_auc"] <= 1 and 0 <= rep[branch]["acc"] <= 1
            assert math.isfinite(rep[branch]["logloss"]) and rep[branch]["logloss"] >= 0
        assert rep["trivial"]["macro_auc"] == 0.5
        assert math.isclose(rep["gap_macro_auc"], rep["federated_dp"]["macro_auc"] -
                            rep["central"]["macro_auc"], rel_tol=0, abs_tol=1e-14)
        accuracy_pass = rep["federated_dp"]["acc"] > rep["trivial"]["acc"]
        auc_pass = rep["federated_dp"]["macro_auc"] > 0.5
        assert rep["diagnostic"] == dict(accuracy_above_majority=accuracy_pass,
                                          macro_auc_above_half=auc_pass, passed=accuracy_pass and auc_pass)
    recomputed = {}
    for branch, key in (("central", "central_mean_sd"), ("federated_dp", "federated_mean_sd"),
                        ("trivial", "trivial_mean_sd")):
        recomputed[key] = {metric: mean_sd([rep[branch][metric] for rep in reps])
                           for metric in ("macro_auc", "acc", "logloss")}
    recomputed["gap_mean_sd"] = dict(macro_auc=mean_sd([
        rep["federated_dp"]["macro_auc"] - rep["central"]["macro_auc"] for rep in reps]))
    recomputed["diagnostic"] = dict(
        applicable_epsilon=8,
        mean_accuracy_above_majority=statistics.mean([
            rep["federated_dp"]["acc"] - rep["trivial"]["acc"] for rep in reps]) > 0,
        mean_macro_auc_above_half=recomputed["federated_mean_sd"]["macro_auc"]["mean"] > 0.5,
        all_replicates_pass=all(rep["diagnostic"]["passed"] for rep in reps))
    verify_statistics(cell["summary"], recomputed)
    rows.append(dict(epsilon=epsilon, evidence_file=path.name,
                     evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), summary=recomputed,
                     wall_clock_total_s=sum(rep["wall_clock"]["total_s"] for rep in reps),
                     wall_clock_note="Attributed total includes the same three central fits in every epsilon row; "
                     "use training_runtime for the sum with central fits counted once."))

acceptance = dict(rows[-1]["summary"]["diagnostic"])
acceptance["passed"] = acceptance["mean_accuracy_above_majority"] and acceptance["mean_macro_auc_above_half"]
assert len(central_fit_seconds) == 3 and len(federated_fit_seconds) == 9
central_total = sum(central_fit_seconds.values())
federated_total = sum(federated_fit_seconds)
har = dict(schema="dsflower-campaign-v2", track="multiclass", contract="pytorch_multiclass",
           dataset=dataset, host=cell["host"], versions=cell["versions"], protocol=cell["protocol"],
           tooling_sha256=tooling, cells=rows, acceptance_epsilon8=acceptance,
           scoring=scoring,
           training_runtime=dict(central_fits=3, federated_fits=9,
                                 central_fit_total_s=central_total, federated_fit_total_s=federated_total,
                                 unique_training_total_s=central_total + federated_total,
                                 note="Sum of three central and nine federated fit durations, with central fits "
                                 "counted once per seed. Federated durations include federation setup/cleanup; "
                                 "final test scoring and other orchestration time are excluded."),
           validation="All three cells: paired official subject-disjoint splits, 21 train/9 test subjects, "
           "three disjoint sites of seven training subjects, 7352/2947 windows, 561 public [-1,1] bounds, "
           "six classes, registry defaults except n_classes=6, patient privacy on subject, clipping norm 1, "
           "five rounds and zero failures. Official outer/inner archive hashes, one test-table load after all "
           "training, timezone-aware declaration/training/scoring order, node policies, runner/tooling hashes, replicate diagnostics, "
           "means and sample SD independently checked without retraining or rescoring.")
combined = dict(schema="dsflower-campaign-v2", track="multiclass", contract="pytorch_multiclass",
                datasets=[ctg, har],
                interpretation="CTG remains a calibration boundary: ranking retained, argmax lost. "
                "HAR561 is the single user-predeclared alternative; both outcomes are retained.")
summary_path.write_text(json.dumps(combined, indent=2) + "\n")
for row in rows:
    summary = row["summary"]
    print(f"HAR561 epsilon={row['epsilon']} central={summary['central_mean_sd']['macro_auc']['mean']:.6f} "
          f"fed={summary['federated_mean_sd']['macro_auc']['mean']:.6f} "
          f"accuracy={summary['federated_mean_sd']['acc']['mean']:.6f} "
          f"gap={summary['gap_mean_sd']['macro_auc']['mean']:.6f} +/- "
          f"{summary['gap_mean_sd']['macro_auc']['sd']:.6f}")
print(f"HAR561 epsilon-8 diagnostic passed: {acceptance['passed']}")
print("Evidence validation passed; wrote summary.json with the original CTG summary intact")
