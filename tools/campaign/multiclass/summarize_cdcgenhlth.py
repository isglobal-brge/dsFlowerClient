#!/usr/bin/env python3
"""Validate scored CDC GenHlth artifacts and retain the CTG/HAR measurements."""
import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hash(value):
    assert isinstance(value, str) and len(value) == 64
    assert all(character in "0123456789abcdef" for character in value)


def mean_sd(values):
    assert len(values) == 3 and all(math.isfinite(value) for value in values)
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values))


def verify_statistics(recorded, recomputed):
    assert recorded.keys() == recomputed.keys()
    for key, value in recomputed.items():
        if isinstance(value, dict):
            verify_statistics(recorded[key], value)
        elif isinstance(value, bool):
            assert recorded[key] is value, f"Incorrect annotation: {key}"
        else:
            assert math.isclose(recorded[key], value, rel_tol=0, abs_tol=1e-12), key


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.utcoffset() is not None, "Timestamp must declare its timezone"
    return parsed


ap = argparse.ArgumentParser()
ap.add_argument("evidence", type=Path)
args = ap.parse_args()
source_dir = Path(__file__).resolve().parent
declared = json.loads((source_dir / "cdcgenhlth_protocol.json").read_text())
summary_path = args.evidence / "summary.json"
previous = json.loads(summary_path.read_text())
retained = previous["datasets"][:2]
assert [entry["dataset"]["uci_id"] for entry in retained] == [193, 240]
assert [entry["dataset"]["n_total"] for entry in retained] == [2126, 10299]
old_hashes = {}
for entry in retained:
    assert [row["epsilon"] for row in entry["cells"]] == [1, 4, 8]
    for row in entry["cells"]:
        path = args.evidence / row["evidence_file"]
        assert sha256(path) == row["evidence_sha256"], path.name
        old_hashes[path.name] = row["evidence_sha256"]
assert len(old_hashes) == 6

defaults = dict(momentum=0, nesterov=False, learning_rate=0.1, batch_size=32,
                local_epochs=1, weight_decay=0, l1_penalty=0, optimizer="sgd",
                scheduler="none", hidden_layers=[], n_classes=5)
levels = [str(i) for i in range(1, 6)]
features = declared["dataset"]["features"]
assert len(features) == len(set(features)) == 20
assert not {"GenHlth", "Diabetes_binary", "ID", "target", "subject"} & set(features)
public_limits = {name: (0, 1) for name in features}
public_limits.update(BMI=(0, 100), MentHlth=(0, 30), PhysHlth=(0, 30),
                     Age=(1, 13), Education=(1, 6), Income=(1, 8))
public_bounds = dict(lower=[public_limits[name][0] for name in features],
                     upper=[public_limits[name][1] for name in features])
assert declared["training"]["execution_epsilon_order"] == [1, 4, 8]
assert declared["split"]["seeds"] == [20260820, 20260821, 20260822]
rows = []
paired = None
fixed = None
central_fit_seconds = {}
federated_fit_seconds = []
for epsilon in (1, 4, 8):
    path = args.evidence / f"cdcgenhlth_pytorch_multiclass_eps{epsilon}.json"
    cell = json.loads(path.read_text())
    assert cell["schema"] == "dsflower-campaign-v2" and cell["cell_id"] == path.stem
    assert cell["contract"] == "pytorch_multiclass"
    versions = cell["versions"]
    assert versions["dsflower"] == versions["dsflowerclient"] == "0.5.0"
    for key in ("runner_sha256", "dsflower_source_commit", "dsflowerclient_release_commit"):
        assert versions[key] == retained[0]["versions"][key] == retained[1]["versions"][key], key
    dataset = cell["dataset"]
    assert dataset["uci_id"] == 891 and dataset["n_total"] == 45000
    assert dataset["cohort_name"] == "cdc45k" and dataset["features"] == features
    assert dataset["thesis_citation_key"] == "uci_cdc_diabetes_health_indicators"
    assert dataset["download_sha256"] == declared["download_sha256"]
    assert dataset["legacy_logistic_prepared_sha256"] == declared["legacy_logistic_prepared_sha256"]
    assert dataset["n_train"] + dataset["n_test"] == 45000
    assert abs(dataset["n_test"] - 9000) <= 2
    protocol = cell["protocol"]
    assert protocol == declared and cell["split"] == declared["split"]
    preparation = cell["preparation"]
    assert preparation["schema"] == "dsflower-cdcgenhlth-preparation-v1"
    assert preparation["cohort_name"] == "cdc45k" and preparation["n_total"] == 45000
    assert preparation["cohort_seed"] == 20260819
    assert preparation["split_seeds"] == declared["split"]["seeds"]
    assert preparation["raw_sha256"] == declared["download_sha256"]
    assert preparation["legacy_prepared_sha256"] == declared["legacy_logistic_prepared_sha256"]
    for key in ("cohort_index_sha256", "prepared_sha256", "training_sha256", "sealed_test_sha256"):
        check_hash(preparation[key])
    scoring = cell["scoring"]
    assert scoring["test_bundle_loads"] == 1 and scoring["held_out_partitions"] == 3
    assert scoring["central_fits"] == 3 and scoring["federated_fits"] == 9
    assert scoring["central_scores_reused_across_epsilon"] is True
    assert scoring["changed_settings_after_scoring"] is False
    assert (timestamp(protocol["declared_at_utc"]) <= timestamp(preparation["prepared_at_utc"]) <=
            timestamp(scoring["training_started_at_utc"]) <= timestamp(scoring["training_completed_at_utc"]) <=
            timestamp(scoring["started_at_utc"]) <= timestamp(cell["generated_at"]))
    privacy = cell["privacy"]
    assert privacy["epsilon"] == epsilon and privacy["delta"] == 1e-6
    assert privacy["unit"] == "row" and privacy["patient_column"] is None
    assert privacy["clipping_norm"] == 1 and privacy["adjacency"] == "replace_one"
    assert privacy["guarantee_scope"] == "per-training"
    assert privacy["bounds_policy"] == declared["bounds_policy"]
    assert cell["rounds"] == 5 and cell["model_params_requested"] == {"n_classes": 5}
    assert cell["model_params_resolved"] == defaults
    tooling = cell["tooling_sha256"]
    assert {"cdcgenhlth.R", "cdcgenhlth_campaign.R", "run_cdcgenhlth.R", "prepare_cdcgenhlth.R",
            "check_cdcgenhlth.R", "run_cdcgenhlth.sh", "cdcgenhlth_protocol.json"} <= tooling.keys()
    for name, expected_hash in tooling.items():
        assert sha256(source_dir / name) == expected_hash, name
    current_fixed = (dataset, protocol, preparation, versions, tooling, scoring)
    if fixed is None:
        fixed = current_fixed
    else:
        assert fixed == current_fixed, "Dataset, protocol, environment, tooling or scoring changed across epsilon"
    reps = cell["per_replicate"]
    assert [rep["seed"] for rep in reps] == declared["split"]["seeds"]
    current = [(rep["split"], rep["central"], rep["trivial"], rep["feature_bounds"]) for rep in reps]
    if paired is None:
        paired = current
    else:
        assert paired == current, "Split, bounds or baselines changed across epsilon"
    assert dataset["n_per_site"] == reps[0]["split"]["n_per_site"]
    assert len({rep["split"]["train_index_sha256"] for rep in reps}) == 3
    assert len({rep["split"]["test_index_sha256"] for rep in reps}) == 3
    for rep in reps:
        split = rep["split"]
        assert split["n_train"] == dataset["n_train"] and split["n_test"] == dataset["n_test"]
        assert len(split["n_per_site"]) == 3 and sum(split["n_per_site"]) == split["n_train"]
        assert split["training_effective_units"] == split["n_per_site"]
        assert not {"train_subjects", "test_subjects", "site_subjects"} & split.keys()
        assert len(split["site_index_sha256"]) == 3
        hashes = [split["train_index_sha256"], split["test_index_sha256"], *split["site_index_sha256"]]
        assert len(set(hashes)) == 5
        for value in hashes:
            check_hash(value)
        assert len(split["class_counts_sites"]) == 3
        for counts, size in [(split["class_counts_train"], split["n_train"]),
                             (split["class_counts_test"], split["n_test"]),
                             *zip(split["class_counts_sites"], split["n_per_site"])]:
            assert set(counts) == set(levels)
            assert all(isinstance(count, int) and count > 0 for count in counts.values())
            assert sum(counts.values()) == size
        for level in levels:
            train_count = split["class_counts_train"][level]
            test_count = split["class_counts_test"][level]
            assert test_count == round((train_count + test_count) * 0.2)
            site_counts = [counts[level] for counts in split["class_counts_sites"]]
            assert sum(site_counts) == train_count and max(site_counts) - min(site_counts) <= 1
        history = rep["history"]
        assert history["n_clients"] == 3 and history["n_rounds"] == 5 and history["n_failures"] == 0
        assert [row["round"] for row in history["rounds"]] == [1, 2, 3, 4, 5]
        assert all(row["n_failures"] == 0 and row["available"] is True for row in history["rounds"])
        assert rep["cleanup_ok"] is True and rep["central"]["convergence"] == 0
        check_hash(rep["model_sha256"])
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
        assert rep["feature_bounds"] == public_bounds
        metadata = rep["released_metadata"]
        assert metadata["model"] == "pytorch_multiclass" and metadata["model_params"] == defaults
        assert metadata["target_levels"] == levels and metadata["features"] == features
        assert metadata["feature_lower"] == public_bounds["lower"]
        assert metadata["feature_upper"] == public_bounds["upper"]
        assert metadata["available_rounds"] == [1, 2, 3, 4, 5]
        assert metadata["n_clients"] == 3 and metadata["num_rounds"] == metadata["requested_num_rounds"] == 5
        assert metadata["privacy"] == "server-enforced-dp" and metadata["status"] == "success"
        assert len(rep["node_privacy"]) == 3
        for node in rep["node_privacy"].values():
            for policy in (node["reported_policy"], node["contract"]):
                assert policy["per_training_epsilon"] == epsilon and policy["per_training_delta"] == 1e-6
                assert policy["dp_unit"] == "row" and policy["patient_column"] is None
                assert policy["adjacency"] == "replace_one"
            assert node["clipping_norm"] == 1 and node["runner_sha256"] == versions["runner_sha256"]
        for branch in ("central", "federated_dp", "trivial"):
            metrics = rep[branch]
            assert 0 <= metrics["macro_auc"] <= 1 and 0 <= metrics["acc"] <= 1
            assert math.isfinite(metrics["logloss"]) and metrics["logloss"] >= 0
            assert set(metrics["auc_by_class"]) == set(levels)
            assert all(0 <= value <= 1 for value in metrics["auc_by_class"].values())
            assert math.isclose(metrics["macro_auc"], statistics.mean(metrics["auc_by_class"].values()),
                                rel_tol=0, abs_tol=1e-12)
        trivial = rep["trivial"]
        prevalence = [split["class_counts_train"][level] / split["n_train"] for level in levels]
        assert len(trivial["training_prevalence"]) == 5
        assert all(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
                   for actual, expected in zip(trivial["training_prevalence"], prevalence))
        majority = levels[max(range(5), key=prevalence.__getitem__)]
        assert str(trivial["majority_class"]) == majority and trivial["macro_auc"] == 0.5
        assert math.isclose(trivial["acc"], split["class_counts_test"][majority] / split["n_test"],
                            rel_tol=0, abs_tol=1e-12)
        assert math.isclose(rep["gap_macro_auc"], rep["federated_dp"]["macro_auc"] -
                            rep["central"]["macro_auc"], rel_tol=0, abs_tol=1e-12)
        assert rep["diagnostic"] == dict(
            accuracy_above_majority=rep["federated_dp"]["acc"] > trivial["acc"],
            macro_auc_above_half=rep["federated_dp"]["macro_auc"] > 0.5)
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
        all_replicates_above_majority=all(rep["diagnostic"]["accuracy_above_majority"] for rep in reps),
        all_replicates_above_half=all(rep["diagnostic"]["macro_auc_above_half"] for rep in reps))
    verify_statistics(cell["summary"], recomputed)
    rows.append(dict(epsilon=epsilon, evidence_file=path.name, evidence_sha256=sha256(path),
                     summary=recomputed,
                     wall_clock_total_s=sum(rep["wall_clock"]["total_s"] for rep in reps),
                     wall_clock_note="Attributed total includes the same three central fits in every epsilon row; "
                     "use training_runtime for the sum with central fits counted once."))

assert len(central_fit_seconds) == 3 and len(federated_fit_seconds) == 9
central_total = sum(central_fit_seconds.values())
federated_total = sum(federated_fit_seconds)
diagnostic = rows[-1]["summary"]["diagnostic"]
cdc = dict(schema="dsflower-campaign-v2", track="multiclass", contract="pytorch_multiclass",
           dataset=dataset, host=cell["host"], versions=versions, protocol=protocol,
           tooling_sha256=tooling, preparation=preparation, cells=rows,
           diagnostic_epsilon8=diagnostic, scoring=scoring,
           training_runtime=dict(central_fits=3, federated_fits=9,
                                 central_fit_total_s=central_total, federated_fit_total_s=federated_total,
                                 unique_training_total_s=central_total + federated_total,
                                 note="Three central and nine federated fit durations; central fits counted once "
                                 "per seed. Federated durations include setup/cleanup. Final scoring excluded."),
           validation="Paired GenHlth-stratified cdc45k splits and three stratified sites, 20 public feature "
           "bounds, five nominal classes, registry defaults except n_classes=5, row privacy, clipping norm 1, "
           "five rounds and zero failures. Raw and legacy logistic cohort hashes, frozen protocol/tooling, "
           "release runner provenance, split index hashes, node policies, class counts, central convergence, "
           "one sealed test-bundle load after three central and nine federated fits, timezone-aware scoring "
           "order, baseline pairing, replicate annotations, means and sample SD independently checked from "
           "recorded artifacts without retraining or rescoring.")
epsilon8 = rows[-1]["summary"]
interpretations = dict(
    ctg="Ranking retained, argmax calibration lost under class imbalance at 1701 training rows. "
        "The original CTG measurements are retained unchanged.",
    har561="Subject-level privacy with seven units per site is outside the useful regime at every budget "
           "in the grid; this is a utility-law boundary, not a defect. The original HAR561 measurements "
           "are retained unchanged.",
    cdcgenhlth=f"Pre-declared cdc45k alternative: 45000 survey respondents, one row per respondent, "
               f"five nominal GenHlth classes. At epsilon 8, mean macro-AUC is "
               f"{epsilon8['federated_mean_sd']['macro_auc']['mean']:.6f} versus central "
               f"{epsilon8['central_mean_sd']['macro_auc']['mean']:.6f}; mean accuracy is "
               f"{epsilon8['federated_mean_sd']['acc']['mean']:.6f} versus majority "
               f"{epsilon8['trivial_mean_sd']['acc']['mean']:.6f}. Diagnostics are annotations only. "
               "This scored outcome is final; no further alternative.")
combined = dict(schema="dsflower-campaign-v2", track="multiclass", contract="pytorch_multiclass",
                datasets=[*retained, cdc], interpretations=interpretations,
                interpretation="CTG and HAR561 remain documented boundary measurements. CDC GenHlth is "
                "the pre-declared large-cohort row-private alternative. All outcomes are retained; "
                "epsilon-8 diagnostics are descriptive annotations, not acceptance criteria.")
for name, expected_hash in old_hashes.items():
    assert sha256(args.evidence / name) == expected_hash, name
summary_path.write_text(json.dumps(combined, indent=2) + "\n")
for row in rows:
    summary = row["summary"]
    print(f"CDC GenHlth epsilon={row['epsilon']} central={summary['central_mean_sd']['macro_auc']['mean']:.6f} "
          f"fed={summary['federated_mean_sd']['macro_auc']['mean']:.6f} "
          f"accuracy={summary['federated_mean_sd']['acc']['mean']:.6f} "
          f"majority={summary['trivial_mean_sd']['acc']['mean']:.6f} "
          f"gap={summary['gap_mean_sd']['macro_auc']['mean']:.6f} +/- "
          f"{summary['gap_mean_sd']['macro_auc']['sd']:.6f}")
print("CDC GenHlth epsilon-8 annotations:", json.dumps(diagnostic, sort_keys=True))
print("Evidence validation complete; retained CTG/HAR561 summaries and six scored JSON hashes")
