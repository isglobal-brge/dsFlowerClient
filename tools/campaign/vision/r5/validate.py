#!/usr/bin/env python3
"""Verify the analytical private head against the unchanged runner, train only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from dsflower_runner import client_app, dp_harness, params, seeding

import emulate


def read(path):
    return json.loads(path.read_text())


def captured(run):
    initial = read(run/'public-capture/public-initial.json')
    with np.load(run/'public-capture/public-initial-arrays.npz') as stored:
        arrays = [stored[str(i)] for i in range(len(stored.files))]
    captures = [read(path) for path in sorted((run/'public-capture').glob('accountant-*.json'))]
    assert len(captures) == 15
    return initial['config'], arrays, captures


def schedule(capture):
    pins = capture['training_pins']
    return emulate.candidate(pins['optimizer']['name'], pins['learning_rate'],
        pins['local_epochs'], pins['batch_size'], pins['optimizer'].get('momentum', 0),
        pins['optimizer']['weight_decay'])


def runner_round(arrays, x, y, capture, cfg, master, round_index):
    model = params.load_user_model(cfg, 512, 'cross_entropy')
    params.set_torch_params(model, arrays)
    pins = dict(capture['training_pins'], round_index=round_index)
    pcfg = dict(capture['privacy_config'])
    pcfg['n_samples'] = len(x)
    result, count = client_app._dp_fit(model, x, y, pcfg, pins, len(x), cfg,
        master, capture['mechanism']['noise_multiplier'])
    assert count == len(x)
    return result


def parity(name, cfg, initial, captures, xs, ys, vx):
    """Five rounds, with independent site streams shared by both implementations."""
    capture = captures[0]
    candidate = schedule(capture)
    mechanism = dp_harness.effective_dpsgd_mechanism(8, 1e-6, 1, len(xs[0]),
        candidate['batch_size'], candidate['local_epochs'], 5)
    assert all(row['mechanism'] == mechanism for row in captures)
    expected_hashes = {(hashlib.sha256(x.tobytes()).hexdigest(),
                       hashlib.sha256(y.tobytes()).hexdigest()) for x, y in zip(xs, ys)}
    observed_hashes = {(row['features_sha256'], row['targets_sha256']) for row in captures}
    assert expected_hashes == observed_hashes
    actual = [array.copy() for array in initial]
    analytical = [array.copy() for array in initial]
    cpu = [array.copy() for array in initial]
    rounds = []
    for round_index in range(1, 6):
        actual_nodes, analytical_nodes, cpu_nodes = [], [], []
        for site, (x, y) in enumerate(zip(xs, ys)):
            master = emulate.master_key(20260926, round_index, site)
            actual_nodes.append(runner_round(actual, x, y, capture, cfg, master, round_index))
            analytical_nodes.append(emulate.fit_round(analytical, x, y, candidate,
                mechanism, master, device='cuda'))
            cpu_nodes.append(emulate.fit_round(cpu, x, y, candidate,
                mechanism, master, device='cpu'))
        actual = [np.add.reduce([node[j] for node in actual_nodes])/3 for j in range(2)]
        analytical = [np.add.reduce([node[j] for node in analytical_nodes])/3 for j in range(2)]
        cpu = [np.add.reduce([node[j] for node in cpu_nodes])/3 for j in range(2)]
        actual_p = emulate.probability(actual, vx)
        row = dict(round=round_index,
            cuda_parameter_max_abs=max(float(np.max(np.abs(a-b))) for a, b in zip(actual, analytical)),
            cpu_parameter_max_abs=max(float(np.max(np.abs(a-b))) for a, b in zip(actual, cpu)),
            cuda_probability_max_abs=float(np.max(np.abs(actual_p-emulate.probability(analytical, vx)))),
            cpu_probability_max_abs=float(np.max(np.abs(actual_p-emulate.probability(cpu, vx)))))
        rounds.append(row)
        print('PARITY', name, json.dumps(row), flush=True)
    passed = all(max(row[key] for key in row if key != 'round') < 1e-4 for row in rounds)
    return dict(name=name, candidate=candidate, mechanism=mechanism,
        feature_hash_parity=True, rounds=rounds, tolerance_absolute=1e-4, passed=passed,
        randomness='Identical fresh master per site and round; unchanged SecureNumpyRng sampling and noise.')


def distribution(initial, xs, ys, vx, vy, candidate, mechanism, historical_metrics):
    seeds = list(range(20261001, 20261013))
    scores = [emulate.metrics(vy, emulate.probability(emulate.fit(initial, xs, ys,
        candidate, mechanism, seed), vx)) for seed in seeds]
    aucs = np.asarray([score['auc'] for score in scores])
    lower, upper = np.quantile(aucs, [.025, .975])
    return dict(candidate=candidate, mechanism=mechanism, seeds=seeds, metrics=scores,
        mean_auc=float(aucs.mean()), sd_auc=float(aucs.std(ddof=1)),
        empirical_auc_range=[float(aucs.min()), float(aucs.max())],
        empirical_auc_95_percentiles=[float(lower), float(upper)],
        historical_metrics=historical_metrics,
        historical_auc_within_empirical_range=bool(aucs.min() <= historical_metrics['auc'] <= aucs.max()),
        interpretation='Independent-noise distribution comparison, not replay of historic custodial noise.')


def main(root):
    emulate.controls()
    assert torch.cuda.is_available(), 'Parity validation requires the actual CUDA runner device'
    out = root/'r5'
    out.mkdir(exist_ok=True)
    data = emulate.load_data(root)
    start = time.monotonic()
    r3 = root/'runs/pytorch_resnet18-eps8-seed20260919'
    cfg3, initial3, captures3 = captured(r3)
    default = parity('r3_registry_default', cfg3, initial3, captures3,
        data['outer_xs'], data['outer_ys'], data['x'])
    checks = [default]
    history = []
    previous = read(root/'r4/diagnosis/dp-confirmation.json')
    for record in previous:
        name = record['candidate']['id']
        run = root/'r4/diagnosis/federations'/name
        if not (run/'federation-status.json').exists() or read(run/'federation-status.json')['status'] != 'trained_unscored':
            run = run.with_name(name+'-startup-retry')
        cfg, initial, captures = captured(run)
        check = parity(name, cfg, initial, captures, data['xs'], data['ys'], data['vx'])
        checks.append(check)
        comparison = distribution(initial, data['xs'], data['ys'], data['vx'], data['vy'],
            check['candidate'], check['mechanism'], record['inner_validation'])
        comparison['evaluation_scope'] = 'The unchanged 171-patient inner-validation subset of training'
        history.append(comparison)
        print('HISTORICAL_R4', name, comparison['mean_auc'], comparison['empirical_auc_range'], flush=True)
    status3 = read(r3/'federation-status.json')
    model3 = params.load_user_model(cfg3, 512, 'cross_entropy')
    model3.load_state_dict(torch.load(Path(status3['output_dir'])/'model.pt',
        map_location='cpu', weights_only=True))
    train3 = emulate.metrics(data['y'], emulate.probability(params.get_torch_params(model3), data['x']))
    comparison3 = distribution(initial3, data['outer_xs'], data['outer_ys'], data['x'], data['y'],
        default['candidate'], default['mechanism'], train3)
    comparison3['evaluation_scope'] = '852 training patients only; released R3 model, no test records or predictions'
    comparison3['published_test_auc_context_only'] = .533
    comparison3['published_test_auc_recomputed'] = False
    result = dict(passed=all(check['passed'] for check in checks), numerical_parity=checks,
        historic_r4_inner_validation=history, historic_r3_training_only=comparison3,
        gate='All three mechanisms and feature hashes match captures; all five-round CUDA/CPU numerical differences < 1e-4.',
        statistical_comparisons_gate_selection=False,
        test_accessed=False, elapsed_s=time.monotonic()-start,
        imported_source_sha256={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in [('client_app', client_app), ('dp_harness', dp_harness), ('seeding', seeding)]})
    emulate.save(out/'validation.json', result)
    print('VALIDATION', result['passed'], 'elapsed', result['elapsed_s'], flush=True)
    assert result['passed'], 'Emulator does not match the unchanged runner'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
