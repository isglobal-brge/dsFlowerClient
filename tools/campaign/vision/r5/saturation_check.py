#!/usr/bin/env python3
"""Exercise output saturation, SGD momentum and weight decay against the runner."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from dsflower_runner import dp_harness

import emulate
from validate import captured, runner_round


def main(root):
    emulate.controls()
    data = emulate.load_data(root)
    cfg, initial, adam_captures = captured(root/'r4/diagnosis/federations/adam_lr0.003_e20_b32-startup-retry')
    _, _, sgd_captures = captured(root/'runs/pytorch_resnet18-eps8-seed20260919')
    cases = [
        ('finite_clamp', emulate.candidate('sgd', 1, 1, 227), sgd_captures[0], True),
        ('sgd_momentum_weight_decay', emulate.candidate('sgd', .3, 2, 64, .9, .0001), sgd_captures[0], False),
        ('adam_weight_decay', emulate.candidate('adam', .003, 2, 128, 0, .0001), adam_captures[0], False),
    ]
    rows = []
    for name, candidate, template, saturation in cases:
        arrays = [array.copy() for array in initial]
        if saturation:
            # One logit is strictly outside the clamp and the other strictly
            # inside for every record; incorrect clamp derivatives cannot hide.
            arrays[0].fill(0)
            arrays[1][:] = [-31, 0]
        capture = copy.deepcopy(template)
        mechanism = dp_harness.effective_dpsgd_mechanism(8, 1e-6, 1, 227,
            candidate['batch_size'], candidate['local_epochs'], 5)
        capture['mechanism'] = mechanism
        pins = capture['training_pins']
        for key in ('learning_rate', 'local_epochs', 'batch_size'):
            pins[key] = candidate[key]
        pins['optimizer']['weight_decay'] = candidate['weight_decay']
        if candidate['optimizer'] == 'sgd':
            pins['optimizer']['momentum'] = candidate['momentum']
        master = emulate.master_key(20260927, 1, 0)
        actual = runner_round(arrays, data['xs'][0], data['ys'][0], capture, cfg, master, 1)
        row = dict(name=name, candidate=candidate, mechanism=mechanism,
            saturation_fraction_at_initialization=.5 if saturation else None)
        for device in ('cpu', 'cuda'):
            approximate = emulate.fit_round(arrays, data['xs'][0], data['ys'][0],
                candidate, mechanism, master, device=device)
            row[device+'_parameter_max_abs'] = max(float(np.max(np.abs(a-b))) for a, b in zip(actual, approximate))
            row[device+'_probability_max_abs'] = float(np.max(np.abs(
                emulate.probability(actual, data['vx'])-emulate.probability(approximate, data['vx']))))
        row['passed'] = all(row[device+'_'+quantity+'_max_abs'] < 1e-4
            for device in ('cpu', 'cuda') for quantity in ('parameter', 'probability'))
        rows.append(row)
        print('ADDITIONAL_PARITY', name, row, flush=True)
    result = dict(passed=all(row['passed'] for row in rows), cases=rows,
        tolerance_absolute=1e-4, test_accessed=False,
        scope='One full local round per case, identical secure sampling/noise; unchanged CUDA _dp_fit compared with CPU and CUDA emulator.')
    emulate.save(root/'r5/finiteclamp-validation.json', result)
    assert result['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
