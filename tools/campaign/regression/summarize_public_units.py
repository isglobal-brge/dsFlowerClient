"""Audit the frozen regression track, including the declared public-unit cell."""
import argparse
import json
import math
from pathlib import Path

from summarize_regression import (
    DEFAULTS, EPSILONS, METRICS, SEEDS, cdcbmi, check_baselines, check_cell,
    check_row_split, close, diagnostics, parkinsons, read, reference,
)


RAW_BMI_HASHES = {
    1: "c5a29ea8b3f61a58f46d203b5375e2913b68ec39d753e99ad83ddb4a89a6c405",
    4: "a3a4d0d321797161da63192caf57fda71ae782226f56d0b64c7e933f5472a87f",
    8: "64d4a989f9f1581fef20b02c9cc1c6d73b4858e013a6f30e7047539af3f575a7",
}
TARGET_TRANSFORM = dict(
    center=55, scale=43, forward="(clip(BMI,12,98)-55)/43",
    inverse="55+43*prediction",
)


def public_units(directory):
    protocol_path = Path(__file__).with_name("cdcbmi_public_units_protocol.json")
    protocol = read(protocol_path)
    protocol_sha = reference(protocol_path)["sha256"]
    raw_protocol = read(protocol_path.with_name("cdcbmi_protocol.json"))
    provenance_path = directory / "cdcbmi_provenance.json"
    provenance = read(provenance_path)
    assert protocol["contract"] == "pytorch_linear_regression"
    assert protocol["target"] == "BMI"
    assert protocol["target_bounds"] == dict(lower=-1, upper=1)
    assert protocol["original_target_bounds"] == dict(lower=12, upper=98)
    assert protocol["public_target_transform"] == TARGET_TRANSFORM
    assert protocol["features"] == raw_protocol["features"]
    assert protocol["feature_bounds"] == raw_protocol["feature_bounds"]
    assert protocol["split"] == raw_protocol["split"]
    assert protocol["seeds"] == SEEDS and protocol["epsilon_order"] == EPSILONS
    assert protocol["sites"] == protocol["replicates"] == 3
    assert protocol["rounds"] == 5 and protocol["registry_defaults"] == DEFAULTS
    assert not protocol["model_params_overrides"]
    assert protocol["dataset"]["thesis_citation_key"] == "uci_cdc_diabetes_health_indicators"
    baselines, cohort, budgets = {}, {}, []
    for epsilon in EPSILONS:
        path = directory / f"cdcbmi_public_units_pytorch_linear_regression_eps{epsilon}.json"
        evidence = read(path)
        raw = read(directory / f"cdcbmi_pytorch_linear_regression_eps{epsilon}.json")
        check_cell(evidence, "row", 45000)
        check_baselines(evidence, baselines)
        assert evidence["cell_id"] == path.stem
        assert evidence["privacy"]["epsilon"] == epsilon
        assert evidence["dataset"] == provenance
        assert evidence["protocol"] == protocol
        assert evidence["protocol_sha256"] == protocol_sha
        assert evidence["target_bounds"] == protocol["target_bounds"]
        assert evidence["feature_bounds"] == raw["feature_bounds"]
        assert evidence["split"] == raw["split"]
        assert evidence["predeclaration_commit"] == protocol_path.with_name(
            "cdcbmi_public_units_declaration_commit.txt").read_text().strip()
        assert evidence["tooling_sha256"][protocol_path.name] == protocol_sha
        for name, checksum in evidence["tooling_sha256"].items():
            assert reference(protocol_path.with_name(name))["sha256"] == checksum
        for rep, old in zip(evidence["per_replicate"], raw["per_replicate"]):
            seed, split = rep["seed"], rep["split"]
            assert split == old["split"]
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
            assert baseline["protocol_sha256"] == protocol_sha
            for branch in ("train", "test"):
                assert baseline[f"{branch}_sha256"] == provenance["checksums"][f"cdcbmi_seed{seed}/{branch}.csv"]
            assert math.isclose(rep["central"]["rmse"], old["central"]["rmse"],
                                rel_tol=0, abs_tol=1e-5)
            for metric in METRICS:
                close(rep["trivial"][metric], old["trivial"][metric])
            close(rep["training_mean"], old["training_mean"])
            decomposition = rep["federated_dp"]["decomposition"]
            assert all(math.isfinite(decomposition[key]) for key in (
                "mean_residual", "residual_variance", "mean_prediction", "target_mean"))
            assert decomposition["residual_variance"] >= 0
            close(rep["federated_dp"]["rmse"] ** 2,
                  decomposition["mean_residual"] ** 2 + decomposition["residual_variance"])
            close(decomposition["mean_residual"],
                  decomposition["mean_prediction"] - decomposition["target_mean"])
            for node in rep["node_privacy"]:
                for config in node["staged_configurations"]:
                    assert config["feature_columns"] == protocol["features"]
                    assert config["target_column"] == "BMI"
                    assert config["feature-bounds"] == protocol["feature_bounds"]
                    assert config["target-bounds"] == protocol["target_bounds"]
        budgets.append(dict(epsilon=epsilon, evidence=reference(path),
                            summary=evidence["summary"], diagnostics=diagnostics(evidence)))
    return dict(
        id="cdcbmi_public_units", contract="pytorch_linear_regression",
        dataset="CDC Diabetes Health Indicators: BMI regression in public units", uci_id=891,
        cohort="cdc45k", n=45000, n_privacy_units=45000, privacy_unit="respondent row",
        training_privacy_units_per_site=[12000] * 3, delta=1e-6,
        provenance=reference(provenance_path), target="BMI",
        target_bounds=protocol["target_bounds"],
        original_target_bounds=protocol["original_target_bounds"],
        public_target_transform=TARGET_TRANSFORM,
        metric_units="Original BMI units for RMSE/MAE; dimensionless R2",
        central_scope_note="OLS with intercept on the same 36000 training rows in the same public feature and target units, with predictions mapped back to BMI units.",
        interpretation="The declared public-unit cell is the operational privacy-cost measurement against OLS. Its federated-DP minus central RMSE gap also contains residual optimization, clipping and federation effects; it is not an isolated causal estimate of privacy noise. Diagnostic outcomes are annotations only and do not trigger another cell.",
        budgets=budgets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for epsilon, checksum in RAW_BMI_HASHES.items():
        path = args.directory / f"cdcbmi_pytorch_linear_regression_eps{epsilon}.json"
        assert reference(path)["sha256"] == checksum
    cells = [parkinsons(args.directory), cdcbmi(args.directory), public_units(args.directory)]
    cells[0]["interpretation"] = (
        "Subject-level privacy with 12/11/11 training units per site is outside the useful regime observed here. "
        "Its approximately epsilon-independent raw-target error is also consistent with the diagnosed scale and optimization limit. "
        "The corrected recording-level OLS reference differs from the subject-mean DP fitting unit.")
    cells[1]["interpretation"] = (
        "Raw BMI units produce an optimization-limited, approximately epsilon-independent error. "
        "This gap is not a privacy-cost measurement; the declared public-unit cell addresses the target-scale artifact.")
    summary = dict(
        schema="dsflower-regression-track-v1", track="regression", cells=cells,
        rounds=5, replicates=3, seeds=SEEDS, epsilons=EPSILONS,
        diagnostics_epsilon=8, diagnostics_role="Annotations only; no acceptance gate",
        policy="Original Parkinson and raw-BMI evidence is retained unchanged, with the existing Parkinson central correction preserved. The public-unit cell was declared before training and each epsilon/seed was scored once; no scored cell was rerun and no further alternative was selected.")
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
    print("Validated three cells, three budgets each; wrote", args.directory / "summary.json")


if __name__ == "__main__":
    main()
