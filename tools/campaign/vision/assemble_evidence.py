#!/usr/bin/env python3
"""Assemble measured records without rerunning training or test scoring."""
import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import socket
import statistics
import subprocess

from opacus.accountants import PRVAccountant


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values), n=len(values))


@lru_cache(maxsize=None)
def independent_accounting(sigma, q, steps, epsilon):
    accountant = PRVAccountant()
    accountant.history = [(sigma, q, steps)]
    delta_add_remove = 1e-6 / (1 + math.exp(epsilon / 2))
    achieved_add_remove = accountant.get_epsilon(delta=delta_add_remove)
    achieved = 2 * achieved_add_remove
    delta_replace_one = delta_add_remove * (1 + math.exp(achieved_add_remove))
    assert achieved <= epsilon and delta_replace_one <= 1e-6
    return dict(accountant="PRVAccountant", epsilon_replace_one=achieved,
                delta_replace_one=delta_replace_one, delta_add_remove=delta_add_remove)


def main(root, out, epsilons):
    tools = Path(__file__).resolve().parent
    protocol = read(tools / "protocol.json")
    runtime = read(root / "runtime_preflight.json")
    assert runtime["status"] == "verified"
    assert sha(tools / "protocol.json") == read(root / "scoring-lock.json")["protocol_sha256"]
    audit = read(root / "prepared/busbra/audit.json")
    source = read(root / "data/release-manifest.json")
    archive = next(r for r in source["sources"] if r["filename"] == "BUSBRA.zip")
    dataset = dict(name="BUS-BRA", version="1.0", source_url="https://zenodo.org/records/8231412",
        archive_url=archive["url"], archive_bytes=archive["bytes"], archive_sha256=archive["sha256"],
        archive_verification="Fresh download matched pinned SHA-256 and publisher MD5 before preparation.",
        citation_reference="Gómez-Flores W, Gregorio-Calas MJ, Pereira WCA (2024). BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems. Medical Physics 51:3110–3123. doi:10.1002/mp.16812.",
        citation_url="https://doi.org/10.1002/mp.16812", dataset_doi="10.5281/zenodo.8231412",
        licence="CC BY 4.0 in Zenodo; archive attribution licence also requires citation.",
        archive_licence_sha256=sha(root / "prepared/busbra/BUS-BRA-LICENSE.txt"),
        total_images=1875, total_patients=1064, patient_mapping=audit["patient_mapping"])
    host = dict(protocol["host"], hostname=socket.gethostname(), gpus=runtime["gpus"],
                state_at_end="left_running", python=platform.python_version())
    dependencies = subprocess.check_output(["Rscript", "-e",
        'cat(jsonlite::toJSON(as.list(sapply(c("dsFlower","dsFlowerClient","dsImaging","dsHPC","DSI","DSLite"), function(p) as.character(packageVersion(p)))), auto_unbox=TRUE))'], text=True)
    release = dict(protocol["release"], installed_versions=json.loads(dependencies), r_version=runtime["r_version"],
                   installed_runner_sha256=runtime["installed_runner_sha256"],
                   torch=json.loads(runtime["torch_probe"]["stdout"]))
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(schema="dsflower-vision-summary-v1", status="executed", contract=protocol["contract"],
        dataset=dataset, release=release, host=host, protocol_sha256=sha(tools / "protocol.json"),
        no_test_tuning=True, alternative=protocol["evaluation"]["alternative"], cells=[])
    for epsilon in epsilons:
        replicates = []
        for seed in protocol["seeds"]:
            run = root / "runs" / f"pytorch_resnet18-eps{epsilon}-seed{seed}"
            scores, fed, twins = read(run / "scores.json"), read(run / "federation-status.json"), read(run / "twins-status.json")
            captures = [read(p) for p in sorted((run / "public-capture").glob("accountant-*.json"))]
            manifests = [read(p) for p in sorted((run / "public-capture").glob("manifest-*.json"))]
            assert len(captures) == len(manifests) == 15
            for capture in captures:
                pcfg, mechanism = capture["privacy_config"], capture["mechanism"]
                assert pcfg["epsilon"] == epsilon and pcfg["delta"] == 1e-6 and pcfg["clipping_norm"] == 1
                capture["independent_composition"] = independent_accounting(
                    mechanism["noise_multiplier"], mechanism["sample_rate"], mechanism["total_steps"], epsilon)
            for item in manifests:
                assert item["manifest"]["dp-unit"] == "patient"
                assert item["manifest"]["patient_column"] == "subject_id"
                assert item["manifest"]["n_units"] == 284
            sizes = []
            for site in range(1, 4):
                path = root / "prepared/vision" / str(seed) / f"site{site}/samples.csv"
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                sizes.append(dict(site=site, images=len(rows), patients=len({r["subject_id"] for r in rows})))
            replicates.append(dict(seed=seed, status="executed", split_sha256=fed["split_sha256"],
                n_train_patients=852, n_test_patients=212, n_train_images=sum(s["images"] for s in sizes),
                n_test_images=scores["n_test_images"], site_sizes=sizes,
                **scores["metrics"], gap_auc=scores["gap"],
                heldout_majority_rate=scores["test_majority_rate"], training_prevalence=scores["training_prevalence"],
                acceptance_diagnostic=scores["acceptance_diagnostic"],
                node_reported_contract=read(run / "node-contract.json"),
                node_training_manifests=manifests, node_accountant_captures=captures,
                central_training=twins["central"], pooled_dp_training=twins["pooled_dp"],
                federated_model_sha256=fed["model_sha256"],
                wall_clock_s=dict(federated=fed["elapsed_s"], central=twins["central"]["elapsed_s"],
                    pooled_dp=twins["pooled_dp"]["elapsed_s"], scoring=scores["scoring_elapsed_s"],
                    driver=read(run / "execution-timing.json")), cleanup_ok=fed["cleanup_ok"]))
        means = {branch: {metric: aggregate([r[branch][metric] for r in replicates])
                           for metric in protocol["evaluation"]["metrics"]}
                 for branch in ("central", "federated_dp", "pooled_dp", "trivial")}
        gap = aggregate([r["gap_auc"] for r in replicates])
        majority = aggregate([r["heldout_majority_rate"] for r in replicates])
        passed = means["federated_dp"]["auc"]["mean"] > .5 and means["federated_dp"]["accuracy"]["mean"] > majority["mean"]
        diagnostic = dict(predeclared=protocol["evaluation"]["diagnostic"], epsilon=8,
            assessed=epsilon == 8, passed=passed if epsilon == 8 else None,
            per_seed_passed=[r["acceptance_diagnostic"] for r in replicates], majority_rate=majority)
        record = dict(schema="dsflower-vision-cell-v1", status="executed", generated_at=datetime.now(timezone.utc).isoformat(),
            contract=protocol["contract"], epsilon=epsilon, delta=1e-6, sites=3, rounds=5,
            dataset=dataset, release=release, host=host, seeds=protocol["seeds"], split_policy=protocol["split"],
            declared_model=protocol["model"], model_params=protocol["model_params"],
            protocol=protocol, protocol_sha256=sha(tools / "protocol.json"),
            scoring_lock_sha256=sha(root / "scoring-lock.json"), per_replicate=replicates,
            aggregate=means, gap_auc=gap, acceptance_diagnostic=diagnostic,
            alternative=dict(attempted=False, reason=protocol["evaluation"]["alternative"]),
            limitations=["Image-level evaluation; patient-level privacy and partitioning.",
                        "Central is the matched finite-schedule noiseless reference, not an optimized upper bound.",
                        "Three seeds; DP randomness additionally depends on node-owned secrets."])
        name = f"pilot_busbra_pytorch_resnet18_eps{epsilon}.json"
        (out / name).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        summary["cells"].append(dict(epsilon=epsilon, file=name, aggregate=means, gap_auc=gap,
                                     acceptance_diagnostic=diagnostic))
    (out / "runtime_preflight.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["cells"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epsilons", type=int, nargs="+", default=[1, 8, 4])
    args = parser.parse_args()
    main(args.root, args.out, args.epsilons)
