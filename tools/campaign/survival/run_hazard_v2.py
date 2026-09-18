#!/usr/bin/env python3
"""One-shot preregistered development, selection, confirmation and summary."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

SEEDS = (1101, 1102, 1103)
SETTINGS = (('support2', 'full'), ('support2', 'small600'),
            ('support2', 'heterogeneous'), ('lung1', 'full'))
GRID = [dict(id=f'h{i+1:02}', protocol_version=2, rounds=r, local_epochs=e,
             batch_size=64, learning_rate=lr, K=10)
        for i, (r, e, lr) in enumerate((r, e, lr) for r, e in
                                      ((10, 2), (20, 2), (10, 4)) for lr in (.02, .05))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    # Exclusive creation: a second launch cannot overwrite a prior run.
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def prepare_split(source, dest, protocol, development):
    """Development never opens outer test.csv; retain original site membership."""
    meta = json.loads((source/'split.json').read_text())
    dest.mkdir(parents=True)
    meta['outer_manifest_sha256'] = sha(source/'split.json')
    meta['outer_train_sha256'] = sha(source/'train.csv')
    assert meta['outer_train_sha256'] == meta['train_sha256']
    original = pd.read_csv(source/'train.csv', dtype={'subject_id': str}, float_precision='round_trip')
    parts, heldout, sites = [], [], []
    for site in range(1, 4):
        path = source/f'site{site}.csv'
        assert sha(path) == meta['sites'][site-1]['split_sha256']
        frame = pd.read_csv(path, dtype={'subject_id': str}, float_precision='round_trip')
        if development:
            keys = frame.subject_id.map(lambda subject: hashlib.sha256(
                f'hazard-v2-inner:{meta["seed"]}:{subject}'.encode()).hexdigest())
            frame = frame.loc[keys.sort_values().index]
            cut = int(.8*len(frame))
            heldout.append(frame.iloc[cut:])
            frame = frame.iloc[:cut]
        parts.append(frame)
        frame.to_csv(dest/path.name, index=False)
        sites.append(dict(site=site, source_rows=len(frame), n_subjects=len(frame),
                          split_sha256=sha(dest/path.name)))
    train = pd.concat(parts, ignore_index=True) if development else original
    if development:
        test = pd.concat(heldout, ignore_index=True)
        assert set(train.subject_id).isdisjoint(test.subject_id)
        assert set(train.subject_id) | set(test.subject_id) == set(original.subject_id)
        assert len(train)+len(test) == len(original)
        train.to_csv(dest/'train.csv', index=False)
        test.to_csv(dest/'test.csv', index=False)
    else:
        assert sha(source/'test.csv') == meta['test_sha256']
        shutil.copyfile(source/'train.csv', dest/'train.csv')
        shutil.copyfile(source/'test.csv', dest/'test.csv')
        test = pd.read_csv(dest/'test.csv', dtype={'subject_id': str})
        assert set(train.subject_id).isdisjoint(test.subject_id)
    meta.update(protocol_version=2, protocol_sha256=sha(protocol),
                evaluation_role='inner validation' if development else 'reused outer holdout confirmation',
                split_rule='Within original sites SHA256(hazard-v2-inner:seed:id), floor(0.8*N)' if development else meta['split_rule'],
                sites=sites, n_train=len(train), n_test=len(test),
                train_sha256=sha(dest/'train.csv'), test_sha256=sha(dest/'test.csv'))
    write(dest/'split.json', meta)
    return dest


def select(scores):
    ranked = []
    for cfg in GRID:
        cells = scores[cfg['id']]
        expected = {f'{d}-{s}-{seed}' for d, s in SETTINGS for seed in SEEDS}
        if set(cells) != expected or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in cells.values()):
            raise ValueError('Selection requires all twelve finite scores per candidate')
        ranked.append(dict(config=cfg, scores=cells, mean=sum(cells[k] for k in sorted(cells))/len(cells)))
    ranked.sort(key=lambda row: (-row['mean'], row['config']['rounds']*row['config']['local_epochs'],
                                row['config']['K'], row['config']['id']))
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace', type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    tools = root/'dsFlowerClient/tools/campaign/survival'
    protocol = tools/'PROTOCOL_F_SURVIVAL.md'
    base = root/'runtime/hazard_v2'
    base.mkdir()  # Refuse relaunch/retry against an existing experiment.
    ext = root/'dsFlowerClient/inst/extdata/campaign'
    archives = {phase: ext/name for phase, name in
                [('development', 'survival_hazard_v2_development'), ('confirmation', 'survival_hazard_v2')]}
    for path in archives.values():
        path.mkdir()
    for name in ('configs', 'logs', 'runs'):
        (base/name).mkdir()
    for cfg in GRID:
        write(base/'configs'/f'{cfg["id"]}.json', cfg)
    subprocess.run([sys.executable, str(root/'dsFlowerClient/tools/check-runner-sync.py'),
                    '--server', str(root/'dsFlower')], check=True, cwd=root)
    from validate_completion import validate_cell
    for variant in ('weibull', 'lognormal', 'hazard'):
        pilot = json.loads((ext/'survival'/f'cell-synthetic-{variant}.json').read_text())
        validate_cell(pilot)
        assert pilot['status'] == 'executed'
    write(base/'preregistration.json', dict(grid=GRID, protocol_sha256=sha(protocol),
          driver_sha256=sha(__file__), campaign_tools_commit=subprocess.check_output(
              ['git', '-C', str(root/'dsFlowerClient'), 'rev-parse', 'HEAD'], text=True).strip(),
          started_utc=datetime.now(timezone.utc).isoformat()))
    outer = root/'data/survival/splits'
    names = [f'{d}-{s}-{seed}' for d, s in SETTINGS for seed in SEEDS]
    inner = {name: prepare_split(outer/name, base/'splits/development'/name, protocol, True) for name in names}

    def run(item):
        phase, name, epsilon, cfg, split = item
        identity = f'{phase}-{cfg["id"]}-{name}-eps{epsilon}'
        out = base/'runs'/identity
        env = os.environ.copy()
        env.update(R_LIBS_USER=str(root/'runtime/rlib'), TMPDIR=str(root/'runtime/tmp'),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (base/'logs'/f'{identity}.log').open('x') as log:
            result = subprocess.run(['Rscript', str(tools/'run_cell.R'), str(root), str(split),
                'hazard', str(epsilon), str(out), str(cfg['rounds']), str(cfg['local_epochs']),
                str(base/'configs'/f'{cfg["id"]}.json')], env=env, cwd=root,
                stdout=log, stderr=subprocess.STDOUT)
        evidence = out/'evidence.json'
        if evidence.exists():
            record = json.loads(evidence.read_text())
        else:
            record = dict(record_type='cell', status='failed', protocol_version=2,
                          dataset=json.loads((split/'split.json').read_text()), variant='hazard',
                          epsilon=epsilon, error=f'Rscript exit {result.returncode} before evidence; see {identity}.log')
        record.update(protocol_version=2, hazard_v2_phase=phase, hazard_v2_config=cfg,
                      driver_returncode=result.returncode)
        write(archives[phase]/f'cell-{identity}.json', record)
        print(identity, record['status'], flush=True)
        return cfg['id'], name, record, result.returncode

    # Two synchronous worker slots; each cell owns three actual custodian processes.
    development = [('development', name, 8, cfg, inner[name]) for cfg in GRID for name in names]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, development))
    scores = {cfg['id']: {} for cfg in GRID}
    for cid, name, record, code in results:
        if code or record['status'] != 'executed':
            raise RuntimeError('Development failure: no selection or confirmation; inspect retained evidence')
        scores[cid][name] = record['results']['federated_dp']['c_index']
    ranked = select(scores)
    winner = ranked[0]['config']
    selection_path = base/'hazard_v2_selection.json'
    write(selection_path, dict(protocol_version=2, selected=winner, ranked=ranked,
          rule='highest mean inner C; fewer epochs; smaller K; ascending ID',
          protocol_sha256=sha(protocol), selected_utc=datetime.now(timezone.utc).isoformat(),
          development_evidence_sha256={p.name: sha(p) for p in sorted(archives['development'].glob('*.json'))},
          inner_manifest_sha256={name: sha(path/'split.json') for name, path in inner.items()}))
    # Only now may outer test CSVs be opened.
    confirm = {name: prepare_split(outer/name, base/'splits/confirmation'/name, protocol, False) for name in names}
    matrix = [('confirmation', f'{d}-{s}-{seed}', eps, winner, confirm[f'{d}-{s}-{seed}'])
              for d, s in SETTINGS for eps in ([8] if s == 'heterogeneous' else [1, 4, 8]) for seed in SEEDS]
    assert len(matrix) == 30
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, matrix))
    subprocess.run([sys.executable, str(tools/'summarize.py'), str(archives['confirmation']), '--hazard-v2'], check=True)
    failed = any(code or record['status'] != 'executed' for _, _, record, code in results)
    write(base/'driver_complete.json', dict(status='failed' if failed else 'executed',
          selection_sha256=sha(selection_path), finished_utc=datetime.now(timezone.utc).isoformat()))
    if failed:
        raise SystemExit('Confirmatory failures retained; summary is incomplete')


if __name__ == '__main__':
    main()
