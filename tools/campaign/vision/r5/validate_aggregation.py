#!/usr/bin/env python3
"""Check shortlisted AUCs using the installed Flower aggregation arithmetic."""
import argparse
import copy
import hashlib
import inspect
import itertools
from pathlib import Path
import time

import flwr
from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
import numpy as np
import torch
from dsflower_runner import dp_harness

import emulate
from validate import captured, read, runner_round
from validate_shortlist import actual_probability, comparison


def flower_mean(nodes, order=(0, 1, 2)):
    records = [RecordDict({'arrays': ArrayRecord(numpy_ndarrays=nodes[site]),
        'metrics': MetricRecord({'num-examples': 1})}) for site in order]
    return aggregate_arrayrecords(records, 'num-examples').to_numpy_ndarrays()


def main(root):
    emulate.controls()
    assert torch.cuda.is_available()
    started = time.monotonic()
    data = emulate.load_data(root)
    out = root/'r5'
    candidates = read(out/'shortlist.json')
    seeds = read(out/'search-declaration.json')['noise_seeds']
    assert len(candidates) == len(seeds) == 3
    cfg, initial, inner_captures = captured(root/'r4/diagnosis/federations/adam_lr0.003_e20_b32-startup-retry')
    _, _, outer_captures = captured(root/'runs/pytorch_resnet18-eps8-seed20260919')
    expected = {(hashlib.sha256(x.tobytes()).hexdigest(), hashlib.sha256(y.tobytes()).hexdigest(), r)
        for x, y in zip(data['xs'], data['ys']) for r in range(1, 6)}
    assert expected == {(row['features_sha256'], row['targets_sha256'], row['round']) for row in inner_captures}
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
            actual, analytical = [a.copy() for a in initial], [a.copy() for a in initial]
            rounds = []
            for round_index in range(1, 6):
                actual_nodes, analytical_nodes = [], []
                for site, (x, y) in enumerate(zip(data['xs'], data['ys'])):
                    master = emulate.master_key(seed, round_index, site)
                    actual_nodes.append(runner_round(actual, x, y, capture, cfg, master, round_index))
                    analytical_nodes.append(emulate.fit_round(analytical, x, y, candidate,
                        mechanism, master, device='cpu'))
                actual = flower_mean(actual_nodes)
                analytical = [np.add.reduce([node[j] for node in analytical_nodes])/3 for j in range(2)]
                row = comparison(actual, analytical, actual_probability(cfg, actual, data['vx']),
                    emulate.probability(analytical, data['vx']), data['vy'])
                rounds.append(dict(round=round_index, **row))
            # Assess all six arrival orders on the same final-round node models.
            # Earlier rounds remain in the documented canonical site order.
            orders = []
            canonical_p = actual_probability(cfg, actual, data['vx'])
            for order in itertools.permutations(range(3)):
                reordered = flower_mean(actual_nodes, order)
                row = comparison(actual, reordered, canonical_p,
                    actual_probability(cfg, reordered, data['vx']), data['vy'])
                orders.append(dict(site_order=list(order), **row))
            passed = all(row['passed'] for row in rounds+orders)
            replications.append(dict(seed=seed, rounds=rounds, final_round_reply_orders=orders, passed=passed))
            print('FLOWER_AGGREGATION_PARITY', candidate['id'], seed,
                'passed', passed, 'runner_auc', rounds[-1]['actual_metrics']['auc'],
                'emulator_auc', rounds[-1]['emulator_metrics']['auc'],
                'auc_delta', rounds[-1]['auc_abs_difference'],
                'parameter_max_abs', rounds[-1]['parameter_max_abs'], flush=True)
        checks.append(dict(candidate=candidate, mechanism=mechanism, replications=replications,
            passed=all(replication['passed'] for replication in replications)))
    source = Path(inspect.getsourcefile(aggregate_arrayrecords))
    result = dict(passed=all(check['passed'] for check in checks), checks=checks,
        feature_hash_parity_with_all_15_r4_captures=True, test_accessed=False,
        flower_version=flwr.__version__, aggregation_source=str(source),
        aggregation_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        installed_arithmetic='Normalize unit weights to 1/3; scale each node tensor before sequential addition in reply order.',
        emulated_arithmetic='Sum three node tensors in site order, then divide by 3.',
        parameter_probability_absolute_tolerance=1e-4, auc_absolute_tolerance=.001,
        noise_seeds=seeds, elapsed_s=time.monotonic()-started,
        scope='All three shortlisted schedules and three search noise seeds, five full rounds with unchanged CUDA _dp_fit and installed Flower aggregation versus CPU analytical emulator; identical diagnostic secure streams.',
        reply_order_scope='Canonical site order for five-round comparison; all six orders evaluated separately for the final aggregation of the same node models. This does not replay historical custodial randomness or exhaust all arrival-order sequences.')
    emulate.save(out/'aggregation-validation.json', result)
    assert result['passed'], 'Installed Flower aggregation changes shortlisted numerical or AUC parity'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
