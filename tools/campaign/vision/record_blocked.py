#!/usr/bin/env python3
"""Record the observed first-attempt runtime blocker without scoring any data."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, out):
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = read(protocol_path)
    runtime = read(root / "runtime_preflight.json")
    check = read(root / "import-check.json")
    run = root / "runs/pytorch_resnet18-eps1-seed20260919"
    failed = read(run / "federation-status.json")
    assert runtime["status"] == "verified" and failed["status"] == "failed"
    assert check["returncode"] == 99 and check["benchmark_observer_enabled"] is False
    assert not (root / "scoring-lock.json").exists()
    assert not list((run / "public-capture").glob("accountant-*.json"))
    assert not (run / "artifact/model.pt").exists()
    out.mkdir(parents=True, exist_ok=True)
    write(out / "runtime_preflight.json", runtime)
    if (root / "provisioning.json").exists():
        write(out / "provisioning.json", read(root / "provisioning.json"))
    write(out / "import_check.json", check)
    first = dict(federation=failed, node_reported_contract=read(run / "node-contract.json"),
        node_diagnostics=read(run / "public-node-diagnostics.json"),
        public_initial=read(run / "public-capture/public-initial.json"),
        observer_install=read(root / "observer-install.json"),
        clientapp_import_check=(read(root / "clientapp-import-check.json")
                               if (root / "clientapp-import-check.json").exists() else None),
        unguarded_public_model_import_trace=((root / "logs/model-build-diagnosis.log").read_text()
                               if (root / "logs/model-build-diagnosis.log").exists() else None),
        note="The unguarded trace was a data-free diagnostic, with no training or privacy operation.")
    write(out / "first_attempt.json", first)
    audit = read(root / "prepared/busbra/audit.json")
    provenance = read(root / "data/release-manifest.json")
    archive = next(r for r in provenance["sources"] if r["filename"] == "BUSBRA.zip")
    checkpoint = next(r for r in provenance["sources"] if r["filename"] == "resnet18-f37072fd.pth")
    dataset = dict(name="BUS-BRA", version="1.0", source_url="https://zenodo.org/records/8231412",
        archive_url=archive["url"], archive_bytes=archive["bytes"], archive_sha256=archive["sha256"],
        archive_verification="Fresh archive matched pinned SHA-256 and publisher MD5.",
        citation_reference="Gómez-Flores W, Gregorio-Calas MJ, Pereira WCA (2024). BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems. Medical Physics 51:3110–3123. doi:10.1002/mp.16812.",
        citation_url="https://doi.org/10.1002/mp.16812", dataset_doi="10.5281/zenodo.8231412",
        licence="Zenodo declares CC BY 4.0; archive attribution licence also requires citation.",
        licence_sha256=sha(root / "prepared/busbra/BUS-BRA-LICENSE.txt"),
        total_images=1875, total_patients=1064, patient_mapping=audit["patient_mapping"])
    blocker = dict(kind="runtime_integrity_default_deny", stage="public_model_construction",
        error="DSFLOWER SECURITY: package '_remote_module_non_scriptable' is not in pinned_packages.json (default-deny).\nAborting process.",
        client_error=failed["error"], affected_nodes=3, child_exit_code=99,
        reproduced_without_benchmark_observer=True, package_or_gate_change_attempted=False,
        alternate_dependency_stack_tested=False, diagnostics="first_attempt.json",
        diagnostics_sha256=sha(out / "first_attempt.json"), independent_check="import_check.json")
    host = dict(protocol["host"], hostname=runtime["hostname"], gpus=runtime["gpus"], state_at_end="left_running")
    release = dict(protocol["release"], installed_versions=runtime["installed_versions"],
        installed_runner_sha256=runtime["installed_runner_sha256"], r_version=runtime["r_version"],
        torch=json.loads(runtime["torch_probe"]["stdout"]), pretrained_checkpoint=checkpoint)
    cells = []
    for epsilon in protocol["epsilon_order"]:
        replicates = []
        for seed in protocol["seeds"]:
            sizes = []
            for site in range(1, 4):
                with (root / "prepared/vision" / str(seed) / f"site{site}/samples.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                sizes.append(dict(site=site, images=len(rows), patients=len({r["subject_id"] for r in rows})))
            attempted = epsilon == 1 and seed == 20260919
            replicates.append(dict(seed=seed, status="failed_before_training" if attempted else "not_started",
                split_sha256=audit["split_hashes"][str(seed)], site_sizes=sizes,
                n_train_patients=852, n_test_patients=212,
                n_train_images=sum(s["images"] for s in sizes),
                n_test_images=1875-sum(s["images"] for s in sizes),
                sizes_source="Prepared collection census; not a utility evaluation.",
                federated_dp=None, central=None, pooled_dp=None, trivial=None, gap_auc=None,
                wall_clock_s=failed["elapsed_s"] if attempted else None,
                node_reported_contract=first["node_reported_contract"] if attempted else None,
                observed_optimizer_steps=0 if attempted else None, completed_rounds=0 if attempted else None))
        record = dict(schema="dsflower-vision-cell-v1", status="blocked", generated_at=datetime.now(timezone.utc).isoformat(),
            contract=protocol["contract"], epsilon=epsilon, delta=1e-6, sites=3, rounds=5,
            seeds=protocol["seeds"], dataset=dataset, release=release, host=host, blocker=blocker,
            protocol=protocol, protocol_sha256=sha(protocol_path), split_policy=protocol["split"],
            declared_model=protocol["model"], model_params=protocol["model_params"],
            requested_privacy=dict(epsilon=epsilon, delta=1e-6, unit="patient", clipping_norm=1., adjacency="replace_one"),
            per_replicate=replicates, aggregate=None, gap_auc=dict(mean=None, sd=None, n=0),
            acceptance_diagnostic=dict(epsilon=8, auc_above=.5, accuracy_above="held-out majority rate",
                                       assessed=False, passed=None),
            test_scoring_started=False, schedule_search_performed=False,
            alternative=dict(attempted=False, reason="Execution stopped at the unchanged integrity gate before any scored primary run."))
        name = f"pilot_busbra_pytorch_resnet18_eps{epsilon}.json"
        write(out / name, record)
        cells.append(dict(epsilon=epsilon, status="blocked", file=name, scored_replicates=0, gap_auc=record["gap_auc"]))
    write(out / "summary.json", dict(schema="dsflower-vision-summary-v1", status="blocked",
        generated_at=datetime.now(timezone.utc).isoformat(), contract=protocol["contract"], dataset=dataset,
        release=release, host=host, blocker=blocker, cells=cells, federation_attempts=1,
        scored_replicates=0, optimizer_steps_observed=0, central_trained=False, pooled_dp_trained=False,
        test_scoring_started=False, acceptance_diagnostic_assessed=False, pod_left_running=True,
        previous_volume_blocker="Superseded: fresh pod storage, release install, archive checks and dsImaging admission passed."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.out)
