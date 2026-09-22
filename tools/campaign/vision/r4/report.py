#!/usr/bin/env python3
"""Assemble R4 aggregate evidence without training or accessing cohort records."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics


BRANCHES = ("central", "nonprivate_federated", "federated_dp", "pooled_dp", "trivial")
METRICS = ("auc", "accuracy", "brier", "log_loss")
DIAGNOSIS_FILES = ("audit.json", "candidates-before-sweep.json", "nonprivate-sweep.json",
                   "shortlist.json", "dp-confirmation.json", "selection.json")
LIMITATIONS = [
    "Primary evaluation is per image; partitioning, head training and privacy are per patient. Patient-mean-feature evaluation is secondary.",
    "All three R4 seeds use the same 852/212 patient split (split seed 20260919). SD describes three training realizations, not split or population uncertainty.",
    "The converged central fit is deterministic and reused across all seeds and epsilons; its zero SD is not an uncertainty estimate.",
    "The nonprivate federated twin removes both clipping and noise, retaining the finite schedule, Poisson sampling, expected-batch divisor and FedAvg structure.",
    "The central-to-private AUC gap combines finite optimization, federation, clipping and noise; it is not a pure noise effect.",
    "The derived private-training gap (federated-DP minus nonprivate federated) includes clipping, noise and their resulting training trajectories; it is not a pure noise effect.",
    "Selection used one 681/171 patient inner split, one initialization, and two actual epsilon-8 federations after nonprivate pruning; selection uncertainty is not estimated.",
    "The original cell had already scored this cohort. R4 did not access held-out records before the locked final scoring pass; no claim of a previously unused external test cohort is made.",
    "Epsilon is the per-training five-round guarantee. No campaign-wide privacy composition claim is made across diagnosis, cells, twins or repeated public-cohort fits.",
    "Node-owned secret randomness is not published; the public seed alone does not reproduce DP sampling or noise.",
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def aggregate(values):
    assert len(values) == 3 and all(math.isfinite(v) for v in values)
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values), n=len(values))


def aggregates(records, key):
    return {branch: {metric: aggregate([row[key][branch][metric] for row in records])
                     for metric in METRICS} for branch in BRANCHES}


@lru_cache(None)
def independent_accounting(sigma, sample_rate, steps, epsilon):
    from opacus.accountants import PRVAccountant
    accountant = PRVAccountant()
    accountant.history = [(sigma, sample_rate, steps)]
    delta0 = 1e-6 / (1 + math.exp(epsilon / 2))
    epsilon0 = accountant.get_epsilon(delta=delta0)
    achieved_delta = delta0 * (1 + math.exp(epsilon0))
    assert 2 * epsilon0 <= epsilon and achieved_delta <= 1e-6
    return dict(accountant="PRVAccountant", full_horizon_steps=steps,
        sample_rate=sample_rate, noise_multiplier=sigma,
        epsilon_replace_one=2 * epsilon0, delta_replace_one=achieved_delta)


def publication_check(value):
    """Reject raw subject IDs and data/prediction/secret payloads in public JSON."""
    if isinstance(value, dict):
        forbidden = {"ids", "patient_ids", "subject_ids", "train_ids", "test_ids",
                     "predictions", "probabilities", "secret", "secret_key",
                     "arrays", "private_key", "targets", "labels"}
        assert not forbidden.intersection(value), forbidden.intersection(value)
        for item in value.values():
            publication_check(item)
    elif isinstance(value, list):
        for item in value:
            publication_check(item)
    elif isinstance(value, str):
        assert not value.startswith("busbra:"), "Raw patient identifier in public evidence"


def replicate(root, protocol, score):
    epsilon, seed = score["epsilon"], score["seed"]
    run = root / "r4/runs" / f"pytorch_resnet18-eps{epsilon}-seed{seed}"
    assert score == read(run / "scores.json")
    assert score["status"] == "scored" and score["scoring_unit"] == "image"
    assert score["split_seed"] == protocol["split_seed"] == 20260919
    assert score["split_sha256"] == protocol["split"]["sha256_by_seed"]["20260919"]
    assert score["n_train_patients"] == 852 and score["n_test_patients"] == 212
    assert set(score["metrics"]) == set(score["patient_metrics"]) == set(BRANCHES)
    for key, gap in (("metrics", "gap"), ("patient_metrics", "patient_gap")):
        assert math.isclose(score[gap], score[key]["federated_dp"]["auc"] - score[key]["central"]["auc"], abs_tol=1e-12)
    federation = read(run / "federation-status.json")
    twins = read(run / "twins-status.json")
    assert federation["status"] == "trained_unscored" and federation["cleanup_ok"]
    assert federation["test_accessed"] is False and federation["epsilon"] == epsilon
    assert federation["seed"] == seed and federation["n_patients_per_site"] == [284] * 3
    captures = [read(path) for path in sorted((run / "public-capture").glob("accountant-*.json"))]
    manifests = [read(path) for path in sorted((run / "public-capture").glob("manifest-*.json"))]
    assert len(captures) == len(manifests) == 15
    assert not list((run / "public-capture").glob("failure-*.json"))
    schedule = protocol["model_params"]
    seen = set()
    for capture in captures:
        mechanism, privacy, pins = capture["mechanism"], capture["privacy_config"], capture["training_pins"]
        assert mechanism["accounting_population"] == 284 and mechanism["adjacency"] == "replace_one"
        assert privacy["epsilon"] == epsilon and privacy["delta"] == 1e-6 and privacy["clipping_norm"] == 1
        assert pins["num_rounds"] == 5 and pins["round_index"] == capture["round"]
        for key in ("learning_rate", "local_epochs", "batch_size"):
            assert pins[key] == schedule[key]
        assert pins["optimizer"]["name"] == schedule["optimizer"]
        assert pins["optimizer"]["weight_decay"] == schedule.get("weight_decay", 0)
        steps = mechanism["steps_per_epoch"] * schedule["local_epochs"]
        assert capture["observed_round_steps"] == steps and mechanism["total_steps"] == 5 * steps
        assert capture["accountant_history"] == [[mechanism["noise_multiplier"], mechanism["sample_rate"], steps]]
        coordinate = (capture["features_sha256"], capture["targets_sha256"], capture["round"])
        assert coordinate not in seen
        seen.add(coordinate)
        capture["independent_full_horizon_composition"] = score["independent_accounting"]
    sites = {(x, y) for x, y, _ in seen}
    assert len(sites) == 3 and seen == {(x, y, r) for x, y in sites for r in range(1, 6)}
    for item in manifests:
        manifest = item["manifest"]
        assert manifest["dp-unit"] == "patient" and manifest["patient_column"] == "subject_id"
        assert manifest["n_units"] == 284 and manifest["image-size"] == 224
        assert manifest["privacy-epsilon"] == epsilon and manifest["privacy-delta"] == 1e-6
    for branch in ("central", "nonprivate_federated", "pooled_dp"):
        assert twins[branch]["status"] == "trained_unscored" and twins[branch]["test_accessed"] is False
    pooled = twins["pooled_dp"]["mechanism"]
    assert pooled["accounting_population"] == 852
    twins["pooled_dp"]["independent_full_horizon_composition"] = independent_accounting(
        pooled["noise_multiplier"], pooled["sample_rate"], pooled["total_steps"], epsilon)
    row = dict(score, **score["metrics"], gap_auc=score["gap"],
        federation_status=federation, twins_status=twins,
        node_reported_contract=read(run / "node-contract.json"),
        node_training_manifests=manifests, node_accountant_captures=captures,
        public_initialization=read(run / "public-capture/public-initial.json"),
        execution_timing=read(run / "execution-timing.json"),
        training_tensor_parity_verified=True)
    for key, prefix, total in (("metrics", "", "gap"), ("patient_metrics", "patient_", "patient_gap")):
        values = score[key]
        optimization = values["nonprivate_federated"]["auc"] - values["central"]["auc"]
        private = values["federated_dp"]["auc"] - values["nonprivate_federated"]["auc"]
        assert math.isclose(optimization + private, score[total], abs_tol=1e-12)
        row[prefix + "optimization_gap_auc"] = optimization
        row[prefix + "private_training_gap_auc"] = private
    publication_check(row)
    return row


def formatted(value):
    return f"{value['mean']:.3f} ± {value['sd']:.3f}"


def markdown(summary, audit, sweep, confirmation):
    chosen = summary["selection"]["selected"]
    candidate = chosen["candidate"]
    lines = ["# BUS-BRA vision R4 diagnosis and corrected cell", "",
        f"Training-only five-fold converged logistic AUC: **{audit['logistic_cv_mean_auc']:.4f} ± {audit['logistic_cv_sd_auc']:.4f}** (fold mean ± sample SD); pooled out-of-fold AUC **{audit['logistic_oof']['auc']:.4f}**.", "",
        f"Selected `{candidate['id']}`: {candidate['optimizer']}, learning rate {candidate['learning_rate']}, {candidate['local_epochs']} local epochs, batch {candidate['batch_size']}, weight decay {candidate['weight_decay']}, five rounds. Actual epsilon-8 federated-DP patient inner-validation AUC **{chosen['inner_validation']['auc']:.4f}**, accuracy **{chosen['inner_validation']['accuracy']:.4f}**, Brier **{chosen['inner_validation']['brier']:.4f}**, log-loss **{chosen['inner_validation']['log_loss']:.4f}** on 171 validation patients after fitting 681 training patients.", "",
        f"The {len(sweep)}-candidate nonprivate pruning sweep and {len(confirmation)} actual federated confirmations used training patients only. Patient identifiers are published only as hashes. All settings and comparators were declared before final training and the held-out scoring lock.", "",
        "| Inner candidate | Nonprivate AUC | DP ε=8 AUC | DP accuracy | DP Brier | DP log-loss | Selected |",
        "|---|---:|---:|---:|---:|---:|:---:|"]
    pruning = {row["candidate"]["id"]: row for row in sweep}
    for row in confirmation:
        name, values = row["candidate"]["id"], row["inner_validation"]
        lines.append(f"| {name} | {pruning[name]['metrics']['auc']:.4f} | {values['auc']:.4f} | {values['accuracy']:.4f} | {values['brier']:.4f} | {values['log_loss']:.4f} | {'yes' if name == candidate['id'] else ''} |")
    lines.extend(["", "Corrected cell, primary per-image scoring; mean ± sample SD over three training seeds on one fixed patient split.", "",
        "| ε | Central AUC | Nonprivate federated AUC | Federated-DP AUC | Pooled-DP AUC | Trivial AUC | DP accuracy | Majority accuracy | AUC gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for cell in summary["cells"]:
        a = cell["aggregate"]
        values = [str(cell["epsilon"])] + [formatted(a[b]["auc"]) for b in BRANCHES]
        values += [formatted(a["federated_dp"]["accuracy"]), formatted(cell["majority_accuracy"]), formatted(cell["gap_auc"])]
        lines.append("| " + " | ".join(values) + " |")
    for key, label in (("aggregate", "Primary image metrics"), ("patient_aggregate", "Secondary patient-mean-feature metrics")):
        lines.extend(["", f"{label} (mean ± sample SD):", "",
            "| ε | Comparator | AUC | Accuracy | Brier | Log-loss |", "|---:|---|---:|---:|---:|---:|"])
        for cell in summary["cells"]:
            for branch in BRANCHES:
                lines.append("| " + " | ".join([str(cell["epsilon"]), branch] +
                    [formatted(cell[key][branch][metric]) for metric in METRICS]) + " |")
    lines.extend(["", "Derived AUC decomposition; the two mean components sum to the total mean gap. Component SDs do not add.", "",
        "| Unit | ε | Nonprivate federated − central | Federated-DP − nonprivate federated | Total gap |",
        "|---|---:|---:|---:|---:|"])
    for prefix, label in (("", "image"), ("patient_", "patient")):
        for cell in summary["cells"]:
            lines.append("| " + " | ".join([label, str(cell["epsilon"])] +
                [formatted(cell[prefix + key]) for key in ("optimization_gap_auc", "private_training_gap_auc", "gap_auc")]) + " |")
    lines.extend(["", "Central is the converged C=1 L2 logistic head on all 852 patient feature vectors, with an unpenalized intercept. The nonprivate federated twin uses the selected schedule without clipping or noise. Pooled-DP uses the released private fitter on all 852 patients with its own calibrated geometry. The trivial probability is training-image prevalence and its class prediction is the training majority.", "",
        "Interpretation and limits:", ""])
    for cell in summary["cells"]:
        lines.append(f"- ε={cell['epsilon']}: {cell['interpretation']}")
    lines.extend("- " + item for item in LIMITATIONS)
    lines.extend(["", "The original registry-default cell remains byte-for-byte preserved and annotated separately as schedule-limited (finite-schedule central AUC 0.596 ± 0.042). R4 is the single declared corrected cell; no alternative was scored.", "",
        "BUS-BRA: Gómez-Flores, Gregorio-Calas and Pereira (2024), Medical Physics 51:3110–3123, [doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812). Dataset [doi:10.5281/zenodo.8231412](https://doi.org/10.5281/zenodo.8231412). Thesis citation key: `gomezflores_busbra_2024`.", ""])
    return "\n".join(lines)


def main(root, out):
    tools = Path(__file__).resolve().parent
    protocol_path = tools / "protocol.json"
    protocol, diagnosis = read(protocol_path), root / "r4/diagnosis"
    assert protocol["status"] == "declared_before_corrected_training"
    assert protocol["sites"] == 3 and protocol["rounds"] == 5
    assert protocol["delta"] == 1e-6 and protocol["clipping_norm"] == 1
    assert protocol["split_seed"] == 20260919 and sorted(protocol["epsilon_order"]) == [1, 4, 8]
    records_file = root / "r4/scores/results.json"
    results, lock = read(records_file), read(root / "r4/scoring-lock.json")
    assert results["status"] == "scored" and len(results["records"]) == 9
    assert results["protocol_sha256"] == lock["protocol_sha256"] == sha(protocol_path)
    assert results["scoring_lock_sha256"] == sha(root / "r4/scoring-lock.json")
    audit, selection = read(diagnosis / "audit.json"), read(diagnosis / "selection.json")
    sweep, confirmation = read(diagnosis / "nonprivate-sweep.json"), read(diagnosis / "dp-confirmation.json")
    assert audit["test_accessed"] is selection["test_accessed"] is False
    assert all(row["test_accessed"] is False and row["actual_federation"] for row in confirmation)
    chosen = selection["selected"]
    assert chosen == sorted(confirmation, key=lambda row: (-row["inner_validation"]["auc"], row["candidate"]["id"]))[0]
    for key, value in chosen["candidate"].items():
        if key != "id":
            assert protocol["model_params"][key] == value
    coordinates = {(row["epsilon"], row["seed"]) for row in results["records"]}
    assert coordinates == {(e, s) for e in (1, 4, 8) for s in protocol["seeds"]}
    dataset = dict(name="BUS-BRA", version="1.0", total_patients=1064, total_images=1875,
        source_url="https://zenodo.org/records/8231412", dataset_doi="10.5281/zenodo.8231412",
        citation_url="https://doi.org/10.1002/mp.16812", citation_key="gomezflores_busbra_2024",
        citation_reference="Gómez-Flores W, Gregorio-Calas MJ, Pereira WCA (2024). BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems. Medical Physics 51:3110–3123.",
        archive_sha256="ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff",
        licence="CC BY 4.0; cite the dataset paper.")
    summary = dict(schema="dsflower-vision-r4-summary-v1", status="executed",
        generated_at=datetime.now(timezone.utc).isoformat(), execution_token=protocol["execution_token"],
        contract=protocol["contract"], dataset=dataset, release=protocol["release"], host=protocol["host"],
        protocol_sha256=sha(protocol_path), scoring_lock_sha256=sha(root / "r4/scoring-lock.json"),
        scores_sha256=sha(records_file), scored_replicates=9, model_params=protocol["model_params"],
        split_seed=protocol["split_seed"], no_r4_test_access_before_scoring=True, selection=selection,
        logistic_cv_mean_auc=audit["logistic_cv_mean_auc"], logistic_cv_sd_auc=audit["logistic_cv_sd_auc"],
        logistic_oof=audit["logistic_oof"], limitations=LIMITATIONS, cells=[])
    for epsilon in (1, 4, 8):
        rows = [replicate(root, protocol, row) for row in sorted(results["records"], key=lambda r: r["seed"]) if row["epsilon"] == epsilon]
        means, patients = aggregates(rows, "metrics"), aggregates(rows, "patient_metrics")
        gap = aggregate([row["gap"] for row in rows])
        majority = aggregate([row["test_majority_rate"] for row in rows])
        passed = means["federated_dp"]["auc"]["mean"] > .5 and means["federated_dp"]["accuracy"]["mean"] > majority["mean"] + 1e-12
        interpretation = (f"Selected-schedule federated-DP AUC {means['federated_dp']['auc']['mean']:.4f}; "
            f"central gap {gap['mean']:+.4f}; accuracy {means['federated_dp']['accuracy']['mean']:.4f} "
            f"versus majority {majority['mean']:.4f}. "
            f"Mean diagnostic {'passes' if passed else 'does not pass'}; annotation only, with no effect on executed status.")
        diagnostic = dict(role="annotation_only", affects_execution_status=False,
            assessed_at_epsilon8=epsilon == 8, mean_passed=passed,
            per_seed_passed=[row["acceptance_diagnostic"] for row in rows],
            definition="Federated-DP AUC > 0.5 and accuracy > held-out majority rate.")
        cell = dict(epsilon=epsilon, file=f"busbra_r4_pytorch_resnet18_eps{epsilon}.json", status="executed",
            aggregate=means, patient_aggregate=patients, gap_auc=gap,
            patient_gap_auc=aggregate([row["patient_gap"] for row in rows]), majority_accuracy=majority,
            diagnostic=diagnostic, interpretation=interpretation, scored_replicates=3)
        for prefix in ("", "patient_"):
            for component in ("optimization_gap_auc", "private_training_gap_auc"):
                key = prefix + component
                cell[key] = aggregate([row[key] for row in rows])
            assert math.isclose(cell[prefix + "optimization_gap_auc"]["mean"] +
                cell[prefix + "private_training_gap_auc"]["mean"], cell[prefix + "gap_auc"]["mean"], abs_tol=1e-12)
        record = dict(cell, schema="dsflower-vision-r4-cell-v1", generated_at=summary["generated_at"],
            execution_token=protocol["execution_token"], contract=protocol["contract"], dataset=dataset,
            release=protocol["release"], host=protocol["host"], protocol=protocol,
            protocol_sha256=summary["protocol_sha256"], scoring_lock_sha256=summary["scoring_lock_sha256"],
            delta=1e-6, clipping_norm=1, privacy_unit="patient", sites=3, rounds=5,
            seeds=protocol["seeds"], split_seed=20260919, model_params=protocol["model_params"],
            declared_model=protocol["model"], scoring_unit="image", secondary_scoring_unit="patient-mean features and modal label",
            training_only_selection=selection, per_replicate=rows, limitations=LIMITATIONS,
            gap_definitions=dict(gap_auc="federated-DP minus converged central AUC",
                optimization_gap_auc="nonprivate federated finite-schedule twin minus converged central AUC",
                private_training_gap_auc="federated-DP minus nonprivate federated AUC; includes clipping, noise and their resulting training trajectories; not a pure noise effect"),
            alternative=dict(attempted=False, reason="One corrected cell declared; no post-score alternative."))
        publication_check(record)
        save(out / cell["file"], record)
        summary["cells"].append(cell)
    publication_check(summary)
    save(out / "summary-r4.json", summary)
    (out / "report-r4.md").write_text(markdown(summary, audit, sweep, confirmation))
    snapshots = [(protocol_path, Path("protocol.json")), (records_file, Path("scores-results.json"))]
    snapshots += [(diagnosis / name, Path("diagnosis") / name) for name in DIAGNOSIS_FILES]
    for name in ("scoring-lock.json", "matrix-start.json", "matrix-complete.json", "training-staging.json", "runtime-preflight.json", "input-verification.json", "matrix-import-failure.json"):
        path = root / "r4" / name
        if path.exists():
            snapshots.append((path, Path(name)))
    for name in ("training-staging.json",):
        if (diagnosis / name).exists():
            snapshots.append((diagnosis / name, Path("diagnosis") / name))
    inner_runs = [(run, Path("diagnosis/federations") / run.name)
                  for run in sorted((diagnosis / "federations").glob("*")) if run.is_dir()]
    failed_startup = diagnosis / "failed-startup"
    if not failed_startup.exists():
        failed_startup = tools / "diagnosis/failed-startup"
    if failed_startup.exists():
        inner_runs.append((failed_startup, Path("diagnosis/failed-startup")))
    for run, relative in inner_runs:
        for name in ("federation-status.json", "job.json", "inner-validation.json", "node-contract.json",
                     "startup-wait.json", "public-node-diagnostics.json"):
            if (run / name).exists():
                snapshots.append((run / name, relative / name))
        snapshots += [(path, relative / "public-capture" / path.name)
                      for path in sorted((run / "public-capture").glob("*.json"))]
    snapshot_hashes = []
    for source, relative in snapshots:
        contents = source.read_bytes()
        publication_check(json.loads(contents))
        destination = out / "r4" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        digest = hashlib.sha256(contents).hexdigest()
        assert sha(destination) == digest
        snapshot_hashes.append(dict(file=str(Path("r4") / relative), sha256=digest, bytes=len(contents)))
    save(out / "r4/snapshot-hashes.json", dict(byte_identical_to_source=True, files=snapshot_hashes))
    print(json.dumps(summary["cells"], indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.out)
