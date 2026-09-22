"""Audit the two frozen regression cells and write their combined summary."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


SEEDS = [20260820, 20260821, 20260822]
EPSILONS = [1, 4, 8]
METRICS = ("rmse", "mae", "r2")
ORIGINAL_HASHES = {
    1: "300d9a91acfcac23a9958fedc96567d6e8cd0ee41e6cb0008f4ac9a5104e099f",
    4: "aaceba52c95f92f497a70cbf31fdc639754c10ea7c1675130ef85ca3e89fcba5",
    8: "b6f6be14f238c8dfd8da9173a13f103cb1bcc82640ca825e4e32afcdc9c98fb4",
}
DEFAULTS = dict(learning_rate=0.01, batch_size=32, local_epochs=1,
                weight_decay=0, l1_penalty=0, optimizer="sgd",
                scheduler="none", hidden_layers=[])


def read(path):
    return json.loads(path.read_text())


def reference(path):
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), (actual, expected)


def mean_sd(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values))


def check_diagnostics(record, rmse_below_trivial, r2_positive):
    if "diagnostic_pass" in record:  # Preserved Parkinson artifact schema.
        assert record["diagnostic_pass"] == (rmse_below_trivial and r2_positive)
    else:
        assert record["diagnostics"] == dict(
            rmse_below_trivial=rmse_below_trivial, r2_positive=r2_positive,
            role="annotation_only")


def check_summary(replicates, summary):
    for branch, key in (("central", "central_mean_sd"),
                        ("federated_dp", "federated_mean_sd"),
                        ("trivial", "trivial_mean_sd")):
        for metric in METRICS:
            expected = mean_sd([rep[branch][metric] for rep in replicates])
            for stat in ("mean", "sd"):
                close(summary[key][metric][stat], expected[stat])
    for rep in replicates:
        close(rep["gap_rmse"], rep["federated_dp"]["rmse"] - rep["central"]["rmse"])
        check_diagnostics(rep, rep["federated_dp"]["rmse"] < rep["trivial"]["rmse"],
                          rep["federated_dp"]["r2"] > 0)
    expected = mean_sd([rep["gap_rmse"] for rep in replicates])
    for stat in ("mean", "sd"):
        close(summary["gap_rmse_mean_sd"][stat], expected[stat])
    check_diagnostics(summary,
                      summary["federated_mean_sd"]["rmse"]["mean"] < summary["trivial_mean_sd"]["rmse"]["mean"],
                      summary["federated_mean_sd"]["r2"]["mean"] > 0)


def diagnostics(evidence):
    summary = evidence["summary"]
    return dict(
        role="Annotations only; not acceptance criteria or rerun triggers",
        federated_rmse_below_trivial=(summary["federated_mean_sd"]["rmse"]["mean"]
                                     < summary["trivial_mean_sd"]["rmse"]["mean"]),
        federated_r2_positive=summary["federated_mean_sd"]["r2"]["mean"] > 0,
        per_seed=[dict(seed=rep["seed"],
                       federated_rmse_below_trivial=rep["federated_dp"]["rmse"] < rep["trivial"]["rmse"],
                       federated_r2_positive=rep["federated_dp"]["r2"] > 0)
                  for rep in evidence["per_replicate"]])


def check_cell(evidence, unit, n):
    assert evidence["schema"] == "dsflower-campaign-v2"
    assert evidence["contract"] == "pytorch_linear_regression"
    assert evidence["versions"]["dsflower"] == evidence["versions"]["dsflowerclient"] == "0.5.0"
    assert evidence["versions"]["runner_sha256"] == "2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724"
    assert evidence["model_params"] == DEFAULTS
    assert not evidence["model_params_overrides"]
    assert evidence["seeds"] == SEEDS
    assert [rep["seed"] for rep in evidence["per_replicate"]] == SEEDS
    assert evidence["rounds"] == 5
    privacy = evidence["privacy"]
    assert privacy["epsilon"] in EPSILONS
    assert privacy["delta"] == 1e-6 and privacy["unit"] == unit and privacy["clipping_norm"] == 1
    for rep in evidence["per_replicate"]:
        split = rep["split"]
        assert split["n_train"] + split["n_test"] == n
        assert len(split["n_per_site"]) == 3 and sum(split["n_per_site"]) == split["n_train"]
        assert rep["history"] == dict(n_clients=3, n_failures=0, n_rounds=5, cleanup_ok=True)
        assert len(rep["node_privacy"]) == 3
        for site, node in enumerate(rep["node_privacy"]):
            policy = node["policy"]
            assert policy["dp_unit"] == unit
            assert policy["per_training_epsilon"] == privacy["epsilon"]
            assert policy["per_training_delta"] == 1e-6
            assert node["clipping_norm"] == 1 and node["staged_configurations"]
            n_units = split["n_subjects_per_site"][site] if unit == "patient" else split["n_per_site"][site]
            for config in node["staged_configurations"]:
                assert config["dp-unit"] == unit and config["n_units"] == n_units
                assert config["n_samples"] == split["n_per_site"][site]
                assert config["privacy-epsilon"] == privacy["epsilon"]
                assert config["privacy-delta"] == 1e-6 and config["privacy-clipping_norm"] == 1
                assert config["num-server-rounds"] == 5 and config["dp_enabled"] is True
                for key, default in (("learning-rate", 0.01), ("batch-size", 32),
                                     ("local-epochs", 1), ("weight-decay", 0),
                                     ("l1-penalty", 0), ("optimizer-name", "sgd"),
                                     ("scheduler-name", "none")):
                    assert config[key] == default
        if unit == "patient":
            groups = [set(ids) for ids in split["site_subjects"]]
            train = set.union(*groups)
            assert len(train) == sum(map(len, groups)) == 34
            assert train == set(split["train_subjects"])
            assert len(set(split["test_subjects"])) == 8
            assert not train & set(split["test_subjects"])
        else:
            assert rep["central_fit"]["observation_unit"] == "row"
            assert rep["central_fit"]["n_training_rows"] == split["n_train"]
    check_summary(evidence["per_replicate"], evidence["summary"])


def check_baselines(evidence, baselines):
    for rep in evidence["per_replicate"]:
        baseline = {key: rep[key] for key in ("central", "trivial", "split", "training_mean")}
        assert baselines.setdefault(rep["seed"], baseline) == baseline


def parkinsons(directory):
    path = directory / "parkinsons_corrected_central_twin.json"
    corrected = read(path)
    assert corrected["schema"] == "dsflower-regression-central-correction-v1"
    assert [rep["seed"] for rep in corrected["baseline_replicates"]] == SEEDS
    assert sorted(cell["epsilon"] for cell in corrected["cells"]) == EPSILONS
    baselines = {}
    budgets = []
    for cell in sorted(corrected["cells"], key=lambda item: item["epsilon"]):
        epsilon = cell["epsilon"]
        original_path = directory / f"pilot_parkinsons_pytorch_linear_regression_eps{epsilon}.json"
        original_ref = reference(original_path)
        assert original_ref["sha256"] == ORIGINAL_HASHES[epsilon]
        assert cell["original_file"] == original_path.name
        assert cell["original_sha256"] == original_ref["sha256"]
        original = read(original_path)
        check_cell(original, "patient", 5875)
        check_baselines(original, baselines)
        assert cell["original_central_mean_sd"] == original["summary"]["central_mean_sd"]
        assert [rep["seed"] for rep in cell["per_replicate"]] == SEEDS
        for rep, old, baseline in zip(cell["per_replicate"], original["per_replicate"], corrected["baseline_replicates"]):
            assert rep["central"] == baseline["central"]
            assert baseline["split"] == old["split"]
            assert baseline["original_central"] == old["central"]
            assert baseline["original_central_fit"] == old["central_fit"]
            assert baseline["central_fit"]["n_training_rows"] == old["split"]["n_train"]
            for metric in METRICS:
                close(baseline["trivial"][metric], old["trivial"][metric])
            close(baseline["training_mean"], old["training_mean"])
            for key in ("federated_dp", "trivial", "diagnostic_pass", "model_sha256"):
                assert rep[key] == old[key]
            assert rep["original_gap_rmse"] == old["gap_rmse"]
        check_summary(cell["per_replicate"], cell["summary"])
        budgets.append(dict(epsilon=epsilon, original_evidence=original_ref,
                            summary=cell["summary"], diagnostics=diagnostics(cell)))
    return dict(
        id="parkinsons", contract="pytorch_linear_regression",
        dataset="UCI Parkinsons Telemonitoring", uci_id=189,
        n=5875, n_privacy_units=42, privacy_unit="subject",
        training_privacy_units_per_site=[12, 11, 11], delta=1e-6,
        corrected_central_evidence=reference(path),
        central_scope_note=corrected["scope_note"],
        original_central_annotation="Twin-computation error: OLS fitted to 34 subject means. Original artifacts are retained byte-for-byte; the corrected reference fits every training recording.",
        original_central_mean_sd=corrected["cells"][0]["original_central_mean_sd"],
        interpretation="Boundary measurement: subject-level privacy with tens of units per site is outside the useful regime observed in this epsilon grid. The 42-subject cohort nominally gives 14 subjects per site; the held-out-subject split leaves 12/11/11 training units. Frozen federated results are unchanged.",
        budgets=budgets)


def check_row_split(metadata, split, cohort):
    assert metadata["index_base"] == 1 and metadata["stratification"] == "Diabetes_binary"
    for key in ("n_train", "n_test", "n_per_site"):
        assert metadata[key] == split[key]
    assert len(metadata["sites"]) == 3
    for key in ("cohort_indices", "source_rows", "uci_ids"):
        train = metadata["train"][key]
        test = metadata["test"][key]
        sites = [site[key] for site in metadata["sites"]]
        assert len(train) == len(set(train)) == split["n_train"]
        assert len(test) == len(set(test)) == split["n_test"]
        assert not set(train) & set(test)
        assert [len(site) for site in sites] == split["n_per_site"]
        combined = [value for site in sites for value in site]
        assert len(combined) == len(set(combined)) and set(combined) == set(train)
        if key == "cohort_indices":
            assert set(train + test) == set(range(1, 45001))
    # Check the three identity columns remain aligned, including at sites.
    identities = {}
    for group in (metadata["train"], metadata["test"]):
        identities.update(zip(group["cohort_indices"], zip(group["source_rows"], group["uci_ids"])))
    assert all(1 <= source_row <= 253680 for source_row, _ in identities.values())
    for site in metadata["sites"]:
        for index, source_row, uci_id in zip(site["cohort_indices"], site["source_rows"], site["uci_ids"]):
            assert identities[index] == (source_row, uci_id)
    assert cohort.setdefault("identities", identities) == identities


def cdcbmi(directory):
    protocol_path = Path(__file__).with_name("cdcbmi_protocol.json")
    protocol = read(protocol_path)
    provenance_path = directory / "cdcbmi_provenance.json"
    provenance = read(provenance_path)
    checksum_lines = (directory / "cdcbmi_CHECKSUMS.sha256").read_text().splitlines()
    assert {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in checksum_lines} == provenance["checksums"]
    assert provenance["cohort"] == "cdc45k" and provenance["n_total"] == 45000
    assert provenance["n_source"] == 253680 and provenance["cohort_seed"] == 20260819
    assert provenance["logistic_prepared_sha256"] == "5f521899386021fd6670d166f4f5ef9a4dcaa80e11d8df10942e60bd18724d3b"
    assert provenance["protocol_sha256"] == reference(protocol_path)["sha256"]
    assert protocol["target"] == "BMI" and protocol["target_bounds"] == dict(lower=12, upper=98)
    assert len(protocol["features"]) == len(set(protocol["features"])) == 20
    assert not {"BMI", "ID", "Diabetes_binary", "Diabetes_012"} & set(protocol["features"])
    baselines, cohort = {}, {}
    budgets = []
    for epsilon in EPSILONS:
        path = directory / f"cdcbmi_pytorch_linear_regression_eps{epsilon}.json"
        evidence = read(path)
        check_cell(evidence, "row", 45000)
        check_baselines(evidence, baselines)
        assert evidence["privacy"]["epsilon"] == epsilon
        assert evidence["dataset"] == provenance
        assert evidence["protocol"] == protocol
        assert evidence["protocol_sha256"] == provenance["protocol_sha256"]
        assert evidence["target_bounds"] == protocol["target_bounds"]
        for rep in evidence["per_replicate"]:
            seed, split = rep["seed"], rep["split"]
            assert (split["n_train"], split["n_test"], split["n_per_site"]) == (36000, 9000, [12000] * 3)
            assert split["membership_file"] == f"cdcbmi_split_seed{seed}.json"
            metadata_path = directory / split["membership_file"]
            checksum = reference(metadata_path)["sha256"]
            assert checksum == split["membership_sha256"]
            assert checksum == provenance["checksums"][f"cdcbmi_seed{seed}/split.json"]
            metadata = read(metadata_path)
            assert metadata["seed"] == seed
            check_row_split(metadata, split, cohort)
            baseline = rep["baseline_provenance"]
            assert baseline["fit_completed_before_test_load"] is True
            assert baseline["protocol_sha256"] == provenance["protocol_sha256"]
            for branch in ("train", "test"):
                assert baseline[f"{branch}_sha256"] == provenance["checksums"][f"cdcbmi_seed{seed}/{branch}.csv"]
            for node in rep["node_privacy"]:
                for config in node["staged_configurations"]:
                    assert config["feature_columns"] == protocol["features"]
                    assert config["target_column"] == "BMI"
                    assert config["feature-bounds"] == protocol["feature_bounds"]
                    assert config["target-bounds"] == protocol["target_bounds"]
        budgets.append(dict(epsilon=epsilon, evidence=reference(path),
                            summary=evidence["summary"], diagnostics=diagnostics(evidence)))
    diagnostic = budgets[-1]["diagnostics"]
    rmse_note = "below" if diagnostic["federated_rmse_below_trivial"] else "not below"
    r2_note = "positive" if diagnostic["federated_r2_positive"] else "nonpositive"
    return dict(
        id="cdcbmi", contract="pytorch_linear_regression",
        dataset="CDC Diabetes Health Indicators: BMI regression", uci_id=891,
        cohort="cdc45k", n=45000, n_privacy_units=45000, privacy_unit="respondent row",
        training_privacy_units_per_site=[12000] * 3, delta=1e-6,
        provenance=reference(provenance_path), target="BMI", target_bounds=protocol["target_bounds"],
        metric_units="Original BMI units for RMSE/MAE; dimensionless R2",
        central_scope_note="OLS with intercept on all 36000 training rows, using the runner's public feature transform and target clipping. The RMSE gap includes optimizer/convergence differences as well as differential privacy.",
        interpretation=(f"The predeclared one-row-per-respondent alternative uses the exact fixed logistic cdc45k cohort. "
                        f"At epsilon 8, mean federated RMSE is {rmse_note} the trivial RMSE and mean R2 is {r2_note}. "
                        "These are annotations only; settings were frozen before scoring and no further alternative was run."),
        budgets=budgets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    cells = [parkinsons(args.directory), cdcbmi(args.directory)]
    summary = dict(
        schema="dsflower-regression-track-v1", track="regression", cells=cells,
        rounds=5, replicates=3, seeds=SEEDS, epsilons=EPSILONS,
        diagnostics_epsilon=8, diagnostics_role="Annotations only; no acceptance gate",
        policy="Original Parkinson federated scores are retained unchanged. Only its central reference and associated gaps are corrected. CDC settings were predeclared and each replicate was scored once; no further alternative.")
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for cell in cells:
        for budget in cell["budgets"]:
            result = budget["summary"]
            print(f"{cell['id']} epsilon={budget['epsilon']}: "
                  f"central={result['central_mean_sd']['rmse']['mean']:.6f}, "
                  f"federated={result['federated_mean_sd']['rmse']['mean']:.6f}, "
                  f"trivial={result['trivial_mean_sd']['rmse']['mean']:.6f}, "
                  f"R2={result['federated_mean_sd']['r2']['mean']:.6f}, "
                  f"gap={result['gap_rmse_mean_sd']['mean']:.6f} +/- {result['gap_rmse_mean_sd']['sd']:.6f}")
    print("Validated two datasets, three budgets each; wrote", args.directory / "summary.json")


if __name__ == "__main__":
    main()
