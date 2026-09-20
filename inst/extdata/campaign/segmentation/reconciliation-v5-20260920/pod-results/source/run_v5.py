#!/usr/bin/env python3
"""Detached Remedy 2: frozen public pretraining, gates, development, 66 cells."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from assemble_evidence import (assemble, load_replicate, pinned_source_split, read_json,
                               released_artifact, sha256)
from benchmark_hooks.public_initialization import verify_capture
from protocol_v5 import candidates, floor, inner_split, mechanism_candidate, select, SEEDS
from run_v4 import verify_development, run, write

TOOLS = Path(__file__).resolve().parent
PACKAGE = TOOLS.parents[2]
EVIDENCE = PACKAGE / 'inst/extdata/campaign/segmentation'
PROVENANCE = EVIDENCE / 'provenance'
PROTOCOL = EVIDENCE / 'protocol-v5.md'


def status(root, phase, **details):
    value = dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **details)
    write(root / 'driver-status.json', value)
    (root / 'STATUS_R2.md').write_text(
        '# Segmentation Remedy 2 — protocol v5\n\n'
        + 'Phase: ' + phase + '\n\n'
        + 'Public BUSI decoder pretraining is NONPRIVATE. BUS-BRA DP fine-tuning is separately accounted.\n'
        + 'V3/v4 evidence and the failed per-site boundary are retained unchanged. vetted=FALSE.\n\n'
        + '```json\n' + json.dumps(value, indent=2) + '\n```\n')
    print(json.dumps(value), flush=True)


def activate(root, candidate, env):
    path = root / 'active-candidate.json'
    write(path, mechanism_candidate(candidate))
    bindings = root / 'public-pretraining' / f"epochs{candidate['pretraining_epochs']}"
    env.update(F_SEG_V4_CONFIG=str(path), F_SEG_V5_BINDINGS=str(bindings))
    os.environ.update(F_SEG_V4_CONFIG=str(path), F_SEG_V5_BINDINGS=str(bindings))
    return bindings


def check_registration():
    registration = read_json(EVIDENCE / 'preregistration-v5.json')
    if registration['protocol_sha256'] != sha256(PROTOCOL) or registration['grid'] != candidates():
        raise ValueError('v5 protocol/grid differs from preregistration')
    for name, digest in registration['files'].items():
        if sha256(PACKAGE / name) != digest:
            raise ValueError('preregistered tooling/provenance changed: ' + name)
    return registration


def confirmation_floor(root, batch):
    rows = []
    errors = []
    for seed in SEEDS:
        work = root / f'runs-batch{batch}/busbra-full-eps8-seed{seed}'
        try:
            load_replicate(work, 'busbra', 'full', 8, seed, provenance=PROVENANCE, batch_size=batch)
            score = read_json(work / 'channel-b.json')
            rows.append(dict(seed=seed, dice=score['metrics']['all']['dice'],
                foreground_dice=score['metrics']['foreground_positive']['dice'],
                strongest_trivial=score['trivial']['strongest_dice'],
                foreground_strongest_trivial=max(s['foreground_positive']['dice']
                                                for s in score['trivial']['scores'].values())))
        except Exception as error:
            errors.append(dict(seed=seed, error=f'{type(error).__name__}: {error}'))
    if errors:
        return dict(verdict='FAIL', reason='incomplete or invalid epsilon8 evidence; numerical floor undetermined',
                    numerical_floor_pass=None, valid_replicates=rows, errors=errors)
    values = {k: sum(r[k] for r in rows) / 3 for k in rows[0] if k != 'seed'}
    passed = floor(values)
    return dict(verdict='PASS' if passed else 'FAIL', numerical_floor_pass=passed,
                across_seed_means=values, replicates=rows, absolute_floor=.50, trivial_margin=.10)


def study(root):
    registration = check_registration()
    root.mkdir(exist_ok=False)
    base = root.parent
    env = dict(os.environ)
    for name in ('prepared', 'features'):
        (root / name).symlink_to(base / name, target_is_directory=True)
    runtime = read_json(base / 'runtime-v4.json')
    if (runtime['protocol_sha256'] != sha256(EVIDENCE / 'protocol-v4.md')
            or sha256(base / 'runtime-v4.json') != registration['retained_runtime_sha256']):
        raise ValueError('retained v4 runtime/protocol changed')
    for name, digest in runtime['runner_files'].items():
        if sha256(base / 'runtime/dsflower_runner' / name) != digest:
            raise ValueError('installed runner differs from retained v4 pin: ' + name)
    # Keep the exact installed runtime identity, distinguishing new benchmark code.
    runtime['v4_protocol_sha256'] = runtime['protocol_sha256']
    runtime['protocol_sha256'] = sha256(PROTOCOL)
    runtime['v5_preregistration_sha256'] = sha256(EVIDENCE / 'preregistration-v5.json')
    runtime['v5_campaign_files'] = registration['files']
    runtime['public_pretraining'] = read_json(PROVENANCE / 'busi-v1/provenance.json')
    write(root / 'runtime-v5.json', runtime)
    splits = []
    for seed in SEEDS:
        source_path = base / f'prepared/busbra/split-{seed}.json'
        value = inner_split(pinned_source_split(source_path, 'busbra', seed, PROVENANCE))
        value['source_sha256'] = sha256(source_path)
        write(root / f'inner-{seed}.json', value)
        splits.append(value)
    gates = read_json(Path(env['F_SEG_GATES_JSON']))
    if any(gates.get(f'segmentation_6_1_{i}') is not True for i in range(1, 8)):
        raise ValueError('all existing mechanism gates required')
    write(root / 'launch-manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
          protocol_sha256=sha256(PROTOCOL), registration=registration, runtime=runtime,
          inner_split_sha256={str(s): sha256(root / f'inner-{s}.json') for s in SEEDS},
          retained_v4_protocol_sha256=sha256(EVIDENCE / 'protocol-v4.md'),
          gates_sha256=sha256(Path(env['F_SEG_GATES_JSON']))))
    status(root, 'public_nonprivate_pretraining', floor='PENDING')
    run([sys.executable, str(TOOLS / 'pretrain_public_v5.py'),
         '--archive', str(base / 'data/busi-v1/BUSI.zip'),
         '--provenance', str(PROVENANCE / 'busi-v1/provenance.json'),
         '--out', str(root / 'public-pretraining'), '--protocol-sha256', sha256(PROTOCOL)],
        env, Path('/workspace/logs/segmentation-v5-pretraining.log'))
    write(root / 'public-pretraining-manifest.json', {
        str(p.relative_to(root)): sha256(p)
        for p in sorted((root / 'public-pretraining').rglob('*'))
        if p.is_file() and p.suffix in ('.json', '.npz')})
    # Real three-node, two-round gates on both trained prefix checkpoints, before scoring.
    status(root, 'synthetic_gates', floor='PENDING')
    for epochs in (20, 60):
        candidate = next(c for c in candidates() if c['pretraining_epochs'] == epochs)
        bindings = activate(root, candidate, env)
        work = root / f'synthetic-epochs{epochs}'
        synthetic_env = dict(env, F_SEG_SYNTHETIC='1', F_SEG_VARIANT='full', F_SEG_BATCH_SIZE='16')
        prepared = base / 'prepared/synthetic'
        log = Path(f'/workspace/logs/segmentation-v5-synthetic-epochs{epochs}.log')
        run([str(TOOLS / 'run_federated.sh'), str(prepared), str(prepared / 'split-20260919.json'),
             '8', '20260919', str(work)], synthetic_env, log)
        run([sys.executable, str(TOOLS / 'verify_synthetic.py'), '--prepared', str(prepared),
             '--run', str(work), '--decoder', 'narrow', '--out', str(work / 'verified.json')], env, log)
        verify_capture(work, read_json(bindings / 'seed20260919.json'))
    status(root, 'development', floor='PENDING')
    results = []
    for candidate in candidates():
        bindings = activate(root, candidate, env)
        for seed, split in zip(SEEDS, splits):
            name = f"pre{candidate['pretraining_epochs']}-batch{candidate['batch_size']}-rounds{candidate['rounds']}-seed{seed}"
            work = root / 'development' / name
            work.mkdir(parents=True)
            log = Path('/workspace/logs') / f'segmentation-v5-development-{name}.log'
            row = dict(candidate=candidate, seed=seed, status='running', log=str(log))
            write(work / 'execution-status.json', row)
            cell_env = dict(env, F_SEG_BATCH_SIZE=str(candidate['batch_size']), F_SEG_VARIANT='full',
                            F_SEG_INNER_SPLIT=str(root / f'inner-{seed}.json'))
            try:
                prepared = base / 'prepared/busbra'
                run([str(TOOLS / 'run_federated.sh'), str(prepared), str(prepared / f'split-{seed}.json'),
                     '8', str(seed), str(work)], cell_env, log)
                verify_capture(work, read_json(bindings / f'seed{seed}.json'))
                run([sys.executable, str(TOOLS / 'score_public.py'), '--features', str(base / 'features/busbra'),
                     '--split', str(work / 'effective-split.json'), '--probabilities', str(work / 'public-probabilities.csv'),
                     '--artifact', str(released_artifact(work)), '--out', str(work / 'channel-b.json')], cell_env, log)
                row.update(verify_development(work, base / 'features/busbra', split, mechanism_candidate(candidate)))
                score = read_json(work / 'channel-b.json')
                row.update(status='executed', foreground_strongest_trivial=max(
                    v['foreground_positive']['dice'] for v in score['trivial']['scores'].values()))
            except Exception as error:
                row.update(status='failed', error=f'{type(error).__name__}: {error}')
            write(work / 'execution-status.json', row)
            results.append(row)
            write(root / 'development-results.json', results)
            print(json.dumps(row), flush=True)
    selection = select(results, splits)
    write(root / 'selection.json', selection)
    candidate = selection['selected']['candidate']
    selected_bindings = activate(root, candidate, env)
    runtime['selected_public_checkpoints'] = {str(seed): read_json(selected_bindings / f'seed{seed}.json')
                                             for seed in SEEDS}
    write(root / 'runtime-v5.json', runtime)
    status(root, 'confirmation', selected=candidate, inner_floor=selection['selected']['floor_passed'], floor='PENDING')
    codes = {}
    for batch in (16, 64):
        try:
            run([sys.executable, str(TOOLS / 'run_matrix.py'), '--root', str(root), '--workers', '1',
                 '--batch-size', str(batch)], env, Path(f'/workspace/logs/segmentation-v5-confirmation-batch{batch}.log'))
            codes[str(batch)] = 0
        except subprocess.CalledProcessError as error:
            codes[str(batch)] = error.returncode
    write(root / 'confirmation-exit-codes.json', codes)
    summary = dict(selection=selection, arms={}, floors={}, retained_versions=['v3', 'v4'],
                   public_pretraining=read_json(PROVENANCE / 'busi-v1/provenance.json'),
                   public_pretraining_audit_sha256=sha256(root / 'public-pretraining/audit.json'),
                   promotion='vetted remains FALSE; utility floor does not assert promotion')
    for batch in (16, 64):
        output = root / f'evidence/batch{batch}'
        output.mkdir(parents=True)
        try:
            documents = assemble(root / f'runs-batch{batch}', PROVENANCE, runtime, PROTOCOL, batch)
            import jsonschema
            for name, document in documents.items():
                schema = 'campaign-summary-schema.json' if name == 'campaign-status.json' else 'evidence-schema.json'
                jsonschema.validate(document, read_json(EVIDENCE / schema))
                write(output / name, document)
            summary['arms'][str(batch)] = documents
        except Exception as error:
            summary['arms'][str(batch)] = dict(status='failed', error=f'{type(error).__name__}: {error}')
        summary['floors'][str(batch)] = confirmation_floor(root, batch)
    # Preserve, hash and label the available prior evidence without writing into v3/v4.
    summary['prior_evidence'] = {str(p): dict(sha256=sha256(p), evidence=read_json(p))
        for version in ('v3', 'v4') for p in sorted((base / version / 'evidence').rglob('*evidence.json'))}
    write(root / 'summary-v5.json', summary)
    complete = (all(c == 0 for c in codes.values()) and all(
        a.get('campaign-status.json', {}).get('status') == 'executed' for a in summary['arms'].values()))
    primary = summary['floors'][str(candidate['batch_size'])]
    status(root, 'completed' if complete else 'completed_with_failures',
           primary_batch=candidate['batch_size'], primary_floor=primary,
           sensitivity_floor=summary['floors'][str(80-candidate['batch_size'])],
           confirmation_exit_codes=codes, all_66_cells_validated=complete,
           development_failures=sum(r['status'] != 'executed' for r in results),
           summary=str(root / 'summary-v5.json'))
    with (root / 'STATUS_R2.md').open('a') as stream:
        stream.write('\nPrimary confirmation floor: **' + primary['verdict'] + '**.\n')
    checksums = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
                 if p.is_file() and not p.is_symlink() and p.suffix in ('.json', '.md', '.npz', '.pt', '.csv')
                 and 'private-state' not in p.parts}
    write(root / 'artifact-checksums.json', checksums)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists():
        parser.error('refusing to overwrite an existing v5 study')
    try:
        study(root)
    except Exception as error:
        if root.exists():
            status(root, 'failed', floor='FAIL: study evidence incomplete; numerical floor undetermined',
                   error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
