"""Append the requested all-recording OLS reference without rerunning DP models."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

import pandas as pd

from central_rows import fit_rows, score_rows, sha256


def mean_sd(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values))


def branch_summary(replicates, branch):
    return {metric: mean_sd([rep[branch][metric] for rep in replicates])
            for metric in ("rmse", "mae", "r2")}


def source_record(path, ranges):
    lines = path.read_text().splitlines()
    return dict(path=str(path), sha256=sha256(path), excerpts=[
        dict(start_line=start, end_line=end, text="\n".join(lines[start - 1:end]))
        for start, end in ranges])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Correction already exists; refusing to overwrite it.")
    protocol = json.loads(args.protocol.read_text())
    originals = []
    for epsilon in (1, 4, 8):
        path = args.evidence / f"pilot_parkinsons_pytorch_linear_regression_eps{epsilon}.json"
        evidence = json.loads(path.read_text())
        assert evidence["privacy"]["epsilon"] == epsilon
        assert evidence["protocol_sha256"] == sha256(args.protocol)
        originals.append((path, evidence, sha256(path)))
    source_path = args.root / "data_cache" / "parkinsons_updrs.data"
    assert sha256(source_path) == originals[0][1]["dataset"]["checksums"][source_path.name]
    source = pd.read_csv(source_path)
    patient_column = protocol["patient_column"]
    columns = [patient_column] + protocol["features"] + [protocol["target"]]
    baseline_replicates = []
    for index, seed in enumerate(protocol["seeds"]):
        original_rep = originals[0][1]["per_replicate"][index]
        assert original_rep["seed"] == seed
        split = original_rep["split"]
        assert sum(split["n_subjects_per_site"]) == 34
        csv_paths = []
        for _, evidence, _ in originals:
            rep = evidence["per_replicate"][index]
            assert rep["seed"] == seed and rep["split"] == split
            assert rep["central"] == original_rep["central"]
            assert rep["trivial"] == original_rep["trivial"]
            work = args.root / "runs" / evidence["cell_id"] / f"rep{index + 1}"
            csv_paths.append((work / "train.csv", work / "test.csv"))
        assert len({sha256(train) for train, _ in csv_paths}) == 1
        assert len({sha256(test) for _, test in csv_paths}) == 1
        train_path, test_path = csv_paths[0]
        train = pd.read_csv(train_path)
        assert len(train) == split["n_train"]
        assert sorted(train[patient_column].unique().tolist()) == split["train_subjects"]
        expected_train = source.loc[source[patient_column].isin(split["train_subjects"]), columns]
        pd.testing.assert_frame_equal(train, expected_train.reset_index(drop=True),
                                      check_exact=False, rtol=1e-12, atol=1e-12)
        coefficients, fit, training_mean = fit_rows(train, protocol)
        test = pd.read_csv(test_path)
        assert len(test) == split["n_test"]
        assert sorted(test[patient_column].unique().tolist()) == split["test_subjects"]
        expected_test = source.loc[source[patient_column].isin(split["test_subjects"]), columns]
        pd.testing.assert_frame_equal(test, expected_test.reset_index(drop=True),
                                      check_exact=False, rtol=1e-12, atol=1e-12)
        assert not set(train[patient_column]) & set(test[patient_column])
        baseline = score_rows(test, protocol, coefficients, fit, training_mean)
        for metric in ("rmse", "mae", "r2"):
            assert math.isclose(baseline["trivial"][metric], original_rep["trivial"][metric],
                                rel_tol=1e-12, abs_tol=1e-12)
        baseline_replicates.append(dict(
            seed=seed, split=split, original_central=original_rep["central"],
            original_central_fit=original_rep["central_fit"],
            train_sha256=sha256(train_path), test_sha256=sha256(test_path),
            **baseline))
    corrections = []
    for path, evidence, original_hash in originals:
        corrected = []
        for rep, baseline in zip(evidence["per_replicate"], baseline_replicates):
            corrected.append(dict(
                seed=rep["seed"], central=baseline["central"],
                federated_dp=rep["federated_dp"], trivial=rep["trivial"],
                gap_rmse=rep["federated_dp"]["rmse"] - baseline["central"]["rmse"],
                original_gap_rmse=rep["gap_rmse"], diagnostic_pass=rep["diagnostic_pass"],
                model_sha256=rep["model_sha256"]))
        summary = {f"{label}_mean_sd": branch_summary(corrected, branch)
                   for label, branch in (("central", "central"),
                                         ("federated", "federated_dp"),
                                         ("trivial", "trivial"))}
        summary["gap_rmse_mean_sd"] = mean_sd([rep["gap_rmse"] for rep in corrected])
        summary["diagnostic_pass"] = evidence["summary"]["diagnostic_pass"]
        assert sha256(path) == original_hash
        corrections.append(dict(
            cell_id=evidence["cell_id"], epsilon=evidence["privacy"]["epsilon"],
            original_file=path.name, original_sha256=original_hash,
            original_central_mean_sd=evidence["summary"]["central_mean_sd"],
            per_replicate=corrected, summary=summary))
    runner = args.root / "dsFlower" / "inst" / "flower_app" / "dsflower_runner"
    result = dict(
        schema="dsflower-regression-central-correction-v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        contract=protocol["contract"], dataset=protocol["dataset"]["name"],
        n_recordings=5875, n_subjects=42,
        correction_type="twin-computation error relative to requested all-training-recording OLS reference",
        original_status="Retained byte-for-byte; original subject-mean OLS is superseded only as the requested recording-level comparator.",
        correction="OLS with intercept on every training recording using the original public feature transform and target clipping, exact saved splits and seeds; no DP model retrained or rescored.",
        scope_note="The release patient-mode runner actually averages each subject's features and continuous target before DP-SGD. The original pooled OLS matched that preprocessing; the corrected all-row OLS is a recording-level reference, not an exact patient-pooling twin. The corrected gap includes this change of fitting unit and the optimizer/convergence difference.",
        privacy_unit_note="42 subjects total; 34 training and 8 held out per replicate; actual training sites contain 12, 11 and 11 privacy units, not 14 each.",
        source_sha256=sha256(source_path), protocol_sha256=sha256(args.protocol),
        script_sha256=sha256(__file__),
        baseline_script_sha256=sha256(Path(__file__).with_name("central_rows.py")),
        transform_source_sha256=sha256(Path(__file__).with_name("central_train.py")),
        runner_source_evidence=dict(
            patient_pooling=source_record(runner / "client_app.py", [(584, 610), (638, 642), (739, 752)]),
            target_clipping=source_record(runner / "task.py", [(95, 127)])),
        baseline_replicates=baseline_replicates, cells=corrections)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"central_mean_sd": corrections[0]["summary"]["central_mean_sd"],
                      "gaps": {cell["epsilon"]: cell["summary"]["gap_rmse_mean_sd"]
                               for cell in corrections}}, indent=2))


if __name__ == "__main__":
    main()
