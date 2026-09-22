#!/usr/bin/env python3
"""Verify shortlisted schedules and AUC ranks against the unchanged DP runner."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata
import torch
from dsflower_runner import dp_harness, params, validation

import emulate
from validate import captured, read, runner_round


def actual_probability(cfg, arrays, x):
    model = params.load_user_model(cfg, 512, 'cross_entropy')
    params.set_torch_params(model, arrays)
    return np.asarray(validation.neural_predictions(model, x, 'cross_entropy'))[:, 1]


def comparison(actual, approximate, actual_p, approximate_p, y):
    actual_metrics = emulate.metrics(y, actual_p)
    approximate_metrics = emulate.metrics(y, approximate_p)
    actual_ranks, approximate_ranks = rankdata(actual_p), rankdata(approximate_p)
    constant = np.std(actual_ranks) == 0 or np.std(approximate_ranks) == 0
    correlation = None if constant else float(np.corrcoef(actual_ranks, approximate_ranks)[0, 1])
    actual_order = np.sign(actual_p[:, None]-actual_p[None, :])
    approximate_order = np.sign(approximate_p[:, None]-approximate_p[None, :])
    upper = np.triu_indices(len(y), 1)
    cross_class = (y[:, None] != y[None, :])[upper]
    disagreements = (actual_order != approximate_order)[upper]
    row = dict(parameter_max_abs=max(float(np.max(np.abs(a-b))) for a, b in zip(actual, approximate)),
        probability_max_abs=float(np.max(np.abs(actual_p-approximate_p))),
        log_probability_max_abs=float(np.max(np.abs(np.log(np.maximum(actual_p.astype(float), 1e-300))
            -np.log(np.maximum(approximate_p.astype(float), 1e-300))))),
        actual_metrics=actual_metrics, emulator_metrics=approximate_metrics,
        auc_abs_difference=abs(actual_metrics['auc']-approximate_metrics['auc']),
        spearman_rank_correlation=correlation,
        pair_order_disagreements=int(disagreements.sum()),
        cross_class_pair_order_disagreements=int(disagreements[cross_class].sum()),
        actual_probability_range=[float(actual_p.min()), float(actual_p.max())],
        emulator_probability_range=[float(approximate_p.min()), float(approximate_p.max())])
    row['passed'] = (row['parameter_max_abs'] < 1e-4 and row['probability_max_abs'] < 1e-4
        and row['auc_abs_difference'] <= .001)
    return row


def main(root):
    emulate.controls()
    assert torch.cuda.is_available()
    started = time.monotonic()
    data = emulate.load_data(root)
    out = root/'r5'
    candidates = read(out/'shortlist.json')
    assert len(candidates) == 3
    seeds = read(out/'search-declaration.json')['noise_seeds']
    cfg, initial, inner_captures = captured(root/'r4/diagnosis/federations/adam_lr0.003_e20_b32-startup-retry')
    _, _, outer_captures = captured(root/'runs/pytorch_resnet18-eps8-seed20260919')
    expected = {(hashlib.sha256(x.tobytes()).hexdigest(), hashlib.sha256(y.tobytes()).hexdigest(), r)
        for x, y in zip(data['xs'], data['ys']) for r in range(1, 6)}
    observed = {(row['features_sha256'], row['targets_sha256'], row['round']) for row in inner_captures}
    assert len(observed) == 15 and observed == expected
    assert all(np.array_equal(a, b) for a, b in zip(initial, data['initial']))
    checks = []
    for candidate in candidates:
        mechanism = dp_harness.effective_dpsgd_mechanism(8, 1e-6, 1, 227,
            candidate['batch_size'], candidate['local_epochs'], 5)
        template = inner_captures[0] if candidate['optimizer'] == 'adam' else outer_captures[0]
        capture = copy.deepcopy(template)
        capture['mechanism'] = mechanism
        pins = capture['training_pins']
        for key in ('learning_rate', 'batch_size', 'local_epochs'):
            pins[key] = candidate[key]
        pins['num_rounds'] = 5
        pins['optimizer']['name'] = candidate['optimizer']
        pins['optimizer']['weight_decay'] = candidate['weight_decay']
        pins['optimizer']['l1_penalty'] = 0
        if candidate['optimizer'] == 'sgd':
            pins['optimizer']['momentum'] = candidate['momentum']
            pins['optimizer']['nesterov'] = False
        assert pins['scheduler']['name'] == 'none'
        capture['privacy_config'].update(epsilon=8, delta=1e-6, clipping_norm=1)
        replications = []
        for seed in seeds:
            arrays = {name: [a.copy() for a in initial] for name in ('runner', 'cpu', 'cuda')}
            rounds = []
            for round_index in range(1, 6):
                nodes = {name: [] for name in arrays}
                for site, (x, y) in enumerate(zip(data['xs'], data['ys'])):
                    master = emulate.master_key(seed, round_index, site)
                    nodes['runner'].append(runner_round(arrays['runner'], x, y, capture, cfg, master, round_index))
                    for device in ('cpu', 'cuda'):
                        nodes[device].append(emulate.fit_round(arrays[device], x, y, candidate,
                            mechanism, master, device=device))
                arrays = {name: [np.add.reduce([node[j] for node in models])/3 for j in range(2)]
                    for name, models in nodes.items()}
                runner_p = actual_probability(cfg, arrays['runner'], data['vx'])
                row = dict(round=round_index)
                for device in ('cpu', 'cuda'):
                    row[device] = comparison(arrays['runner'], arrays[device], runner_p,
                        emulate.probability(arrays[device], data['vx']), data['vy'])
                rounds.append(row)
            passed = all(row[device]['passed'] for row in rounds for device in ('cpu', 'cuda'))
            replications.append(dict(seed=seed, rounds=rounds, passed=passed))
            print('SHORTLIST_PARITY', candidate['id'], seed, 'passed', passed,
                'runner_auc', rounds[-1]['cpu']['actual_metrics']['auc'],
                'cpu_auc_delta', rounds[-1]['cpu']['auc_abs_difference'],
                'cuda_auc_delta', rounds[-1]['cuda']['auc_abs_difference'], flush=True)
        checks.append(dict(candidate=candidate, mechanism=mechanism, replications=replications,
            passed=all(replication['passed'] for replication in replications)))
    result = dict(passed=all(check['passed'] for check in checks), checks=checks,
        feature_hash_parity_with_all_15_r4_captures=True, test_accessed=False,
        parameter_probability_absolute_tolerance=1e-4, auc_absolute_tolerance=.001,
        noise_seeds=seeds, elapsed_s=time.monotonic()-started,
        scope='All three shortlisted schedules, all three search noise seeds, all five FedAvg rounds; unchanged CUDA _dp_fit versus CPU and CUDA analytical emulator, with shared secure sampling/noise streams.')
    emulate.save(out/'shortlist-validation.json', result)
    assert result['passed'], 'Shortlisted schedule fails runner/emulator numerical or AUC parity'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
