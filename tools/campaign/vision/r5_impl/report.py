#!/usr/bin/env python3
"""Assemble R5 aggregate evidence without training or accessing cohort records."""
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
    "All three R5 seeds use the same 852/212 patient split (split seed 20260919). SD describes three training realizations, not split or population uncertainty.",
    "The converged central fit is deterministic and reused across all seeds and epsilons; its zero SD is not an uncertainty estimate.",
    "The nonprivate federated twin removes both clipping and noise, retaining the finite schedule, Poisson sampling, expected-batch divisor and FedAvg structure.",
    "Nonprivate-twin Poisson draws use independent benchmark randomness; initialization and sampling law are matched, not individual sampling draws.",
    "Canonical federated image prediction retains the released numerical defaults; direct twins and secondary patient inference disable TF32 and use deterministic controls. Training tensor parity is verified; test-feature bitwise parity across these inference routes is not asserted.",
    "The central-to-private AUC gap combines finite optimization, federation, clipping and noise; it is not a pure noise effect.",
    "The derived private-training gap (federated-DP minus nonprivate federated) includes clipping, noise and their resulting training trajectories; it is not a pure noise effect.",
    "Selection used one 681/171 patient inner split, three emulated noise seeds over 736 candidates, then three actual epsilon-8 confirmations; selection uncertainty is not estimated.",
    "The original cell had already scored this cohort. R5 did not access held-out records before the locked final scoring pass; no claim of a previously unused external test cohort is made.",
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
    run = root / "r5_impl/runs" / f"pytorch_resnet18-eps{epsilon}-seed{seed}"
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
        assert pins["optimizer"]["momentum"] == schedule["momentum"]
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


def markdown(summary):
    lines = ['# BUS-BRA R5 corrected cell', '',
        'Declared SGD lr 3, momentum 0.9, two local epochs, full-site batch (284/site; 852 pooled), five rounds. Selected on training patients under the real DP contract. Fixed 852/212 patient split; three training seeds. Primary per-image metrics; mean ± sample SD.', '',
        '| ε | Central AUC | Nonprivate federated AUC | Federated-DP AUC | Pooled-DP AUC | Trivial AUC | DP accuracy | Majority accuracy | AUC gap |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cell in summary['cells']:
        a = cell['aggregate']
        values = [str(cell['epsilon'])] + [formatted(a[b]['auc']) for b in BRANCHES]
        values += [formatted(a['federated_dp']['accuracy']), formatted(cell['majority_accuracy']), formatted(cell['gap_auc'])]
        lines.append('| ' + ' | '.join(values) + ' |')
    for key, label in [('aggregate', 'Primary image metrics'), ('patient_aggregate', 'Secondary patient metrics')]:
        lines += ['', label + ':', '', '| ε | Arm | AUC | Accuracy | Brier | Log-loss |', '|---:|---|---:|---:|---:|---:|']
        for cell in summary['cells']:
            for branch in BRANCHES:
                lines.append('| ' + ' | '.join([str(cell['epsilon']), branch] + [formatted(cell[key][branch][m]) for m in METRICS]) + ' |')
    lines += ['', 'Interpretations:', ''] + ['- ' + c['interpretation'] for c in summary['cells']]
    lines += ['', 'Limits:', ''] + ['- ' + x for x in LIMITATIONS]
    lines += ['', 'R3 was schedule-limited at registry defaults. R4 demonstrated that non-private pruning does not transfer under DP; its final scoring was stopped before test access. All original records remain unchanged.', '',
        'BUS-BRA: Gómez-Flores, Gregorio-Calas and Pereira (2024), Medical Physics 51:3110–3123. Paper DOI 10.1002/mp.16812; dataset DOI 10.5281/zenodo.8231412; thesis key `gomezflores_busbra_2024`.', '']
    return '\n'.join(lines)


def main(root, out):
    tools = Path(__file__).resolve().parent
    protocol_path = tools / 'protocol.json'
    protocol = read(protocol_path)
    work = root / 'r5_impl'
    results, lock = read(work / 'scores/results.json'), read(work / 'scoring-lock.json')
    assert results['status'] == 'scored' and len(results['records']) == 9
    assert results['protocol_sha256'] == lock['protocol_sha256'] == sha(protocol_path)
    assert results['scoring_lock_sha256'] == sha(work / 'scoring-lock.json')
    assert {(r['epsilon'], r['seed']) for r in results['records']} == {(e,s) for e in [8,4,1] for s in protocol['seeds']}
    selection = read(root / 'r5/selection.json')
    assert selection['test_accessed'] is False
    assert sha(root / 'r5/selection.json') == protocol['diagnosis']['selection_sha256']
    assert selection['selected'] == protocol['diagnosis']['selected']
    preflight = read(work / 'runtime-preflight.json')
    assert preflight['status'] == 'verified'
    dataset = dict(name='BUS-BRA', version='1.0', total_patients=1064, total_images=1875, **protocol['provenance'])
    release = dict(protocol['release'], installed_versions=preflight['installed_versions'],
        installed_runner_sha256=preflight['installed_runner_sha256'], r_version=preflight['r_version'],
        torch=json.loads(preflight['torch_probe']['stdout']))
    host = dict(protocol['host'], hostname=preflight['hostname'], gpus=preflight['gpus'], state_at_end='left_running')
    wall = dict(matrix_start=read(work / 'matrix-start.json'), matrix_complete=read(work / 'matrix-complete.json'),
        scoring_started_at=lock['started_at'], scoring_finished_at=results['finished_at'], scoring_elapsed_s=results['elapsed_s'])
    summary = dict(schema='dsflower-vision-r5-summary-v1', status='executed',
        generated_at=datetime.now(timezone.utc).isoformat(), contract=protocol['contract'],
        dataset=dataset, release=release, host=host, protocol_sha256=sha(protocol_path),
        scoring_lock_sha256=sha(work / 'scoring-lock.json'), scores_sha256=sha(work / 'scores/results.json'),
        scored_replicates=9, model_params=protocol['model_params'], split_seed=20260919,
        no_test_access_before_scoring=True, selection=selection, limitations=LIMITATIONS, wall_clock=wall, cells=[])
    for epsilon in [8,4,1]:
        rows = [replicate(root, protocol, row) for row in sorted(results['records'], key=lambda r:r['seed']) if row['epsilon']==epsilon]
        a, pa = aggregates(rows, 'metrics'), aggregates(rows, 'patient_metrics')
        gap = aggregate([r['gap'] for r in rows]); majority=aggregate([r['test_majority_rate'] for r in rows])
        passed = a['federated_dp']['auc']['mean']>.5 and a['federated_dp']['accuracy']['mean']>majority['mean']+1e-12
        interpretation=(f"DP-aware selected schedule: federated-DP AUC {a['federated_dp']['auc']['mean']:.6f}; central gap {gap['mean']:+.6f}; "
            f"accuracy {a['federated_dp']['accuracy']['mean']:.6f} versus majority {majority['mean']:.6f}. "
            f"Diagnostic {'passes' if passed else 'does not pass'}; annotation only. "
            f"Brier {a['federated_dp']['brier']['mean']:.6f}, log-loss {a['federated_dp']['log_loss']['mean']:.6f}; discrimination and calibration must be interpreted separately.")
        cell = dict(epsilon=epsilon, file=f'busbra_r5_pytorch_resnet18_eps{epsilon}.json', status='executed',
            phase='dp_aware_selected_schedule', aggregate=a, patient_aggregate=pa, gap_auc=gap,
            patient_gap_auc=aggregate([r['patient_gap'] for r in rows]), majority_accuracy=majority,
            diagnostic=dict(role='annotation_only',affects_execution_status=False,mean_passed=passed,
                per_seed_passed=[r['acceptance_diagnostic'] for r in rows]), interpretation=interpretation, scored_replicates=3)
        for prefix in ['', 'patient_']:
            for component in ['optimization_gap_auc','private_training_gap_auc']:
                cell[prefix+component]=aggregate([r[prefix+component] for r in rows])
        record=dict(cell,schema='dsflower-vision-r5-cell-v1', generated_at=summary['generated_at'],
            contract=protocol['contract'],dataset=dataset,release=release,host=host,protocol=protocol,
            protocol_sha256=sha(protocol_path),scoring_lock_sha256=summary['scoring_lock_sha256'],
            delta=1e-6,clipping_norm=1,privacy_unit='patient',sites=3,rounds=5,seeds=protocol['seeds'],
            split_seed=20260919,model_params=protocol['model_params'],declared_model=protocol['model'],
            scoring_unit='image',secondary_scoring_unit='patient-mean features and modal label',
            training_only_selection=selection,per_replicate=rows,limitations=LIMITATIONS,wall_clock=wall,
            gap_definitions=dict(gap_auc='federated-DP minus converged central AUC',
                optimization_gap_auc='nonprivate federated minus converged central AUC',
                private_training_gap_auc='federated-DP minus nonprivate federated; includes clipping and noise'),
            alternative=dict(attempted=False,reason='One corrected cell, no post-score alternative'))
        publication_check(record);save(out/cell['file'],record);summary['cells'].append(cell)
    publication_check(summary);save(out/'busbra_r5_summary.json',summary)
    (out/'busbra_r5_report.md').write_text(markdown(summary))
    snapshots = [protocol_path, work/'scores/results.json']
    snapshots += [work/name for name in ['scoring-lock.json','matrix-start.json','matrix-complete.json','training-staging.json','runtime-preflight.json']]
    for source in snapshots:
        value=read(source);publication_check(value)
        target=out/'r5_impl'/source.name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes());assert sha(target)==sha(source)
    print(markdown(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.out)
