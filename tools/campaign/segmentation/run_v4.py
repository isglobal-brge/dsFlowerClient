#!/usr/bin/env python3
"""One immutable public pod study: development, selection, confirmation, summary."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from assemble_evidence import (assemble, read_json, released_artifact, sha256,
                               validate_captures, pinned_source_split)
from protocol_v4 import candidates, inner_split, select, SEEDS

TOOLS = Path(__file__).resolve().parent
PROVENANCE = TOOLS.parents[2] / 'inst/extdata/campaign/segmentation/provenance'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(command, env, log):
    with log.open('a') as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def verify_development(work, features, expected_split, candidate):
    """Validate actual trained release, tensors and full-horizon site accounting."""
    import base64
    import hashlib
    import numpy as np
    from dsflower_runner import segmentation
    from central_twins import load_twin_pins, validate_twin_pins, source_row_counts
    status = read_json(work / 'federation-status.json')
    split = read_json(work / 'effective-split.json')
    if split != expected_split or status.get('cleanup_ok') is not True or status.get('status') == 'failed':
        raise ValueError('development split/federation/cleanup mismatch')
    score = read_json(work / 'channel-b.json')
    if (score['artifact_sha256'] != sha256(released_artifact(work))
            or score['artifact_sha256'] != status['model_sha256']
            or score['split_sha256'] != sha256(work / 'effective-split.json')
            or status['split_sha256'] != expected_split['source_sha256']):
        raise ValueError('development scored release provenance mismatch')
    capture = work / 'public-capture'
    initial = read_json(capture / 'public-initial.json')
    cfg = initial['config']
    if initial['seed'] != split['seed'] or cfg['segmentation-alpha'] != .5:
        raise ValueError('development initialization/loss differs')
    validate_twin_pins(cfg, read_json(features / 'manifest.json'), load_twin_pins(cfg), candidate['batch_size'])
    if json.loads(base64.b64decode(cfg['model-spec-b64'])) != segmentation.decoder_spec(candidate['decoder']):
        raise ValueError('development decoder differs')
    with np.load(capture / 'public-initial-arrays.npz', allow_pickle=False) as arrays:
        if [hashlib.sha256(arrays[str(i)].tobytes()).hexdigest() for i in range(len(arrays.files))] != initial['tensor_sha256']:
            raise ValueError('initial parameter hashes differ')
    with np.load(features / 'public-subject-tensors.npz', allow_pickle=False) as data:
        lookup = {str(s): i for i, s in enumerate(data['subjects'])}
        hashes, census = {}, {}
        source_rows = source_row_counts(features / 'samples.csv', split['sites'])
        for site, count in zip(split['sites'], source_rows):
            rows = [lookup[s] for s in sorted(site)]
            pair = tuple(hashlib.sha256(data[k][rows].tobytes()).hexdigest() for k in ('X', 'y'))
            hashes[pair], census[pair] = len(rows), count
    accounting = validate_captures([read_json(p) for p in capture.glob('accountant-*.json')],
                                  list(map(len, split['sites'])), 8, hashes, census, candidate['batch_size'])
    write(work / 'validated-development.json', dict(accounting=accounting, score=score,
                                                   candidate=candidate, split_sha256=sha256(work / 'effective-split.json')))
    return dict(dice=score['metrics']['all']['dice'],
                foreground_dice=score['metrics']['foreground_positive']['dice'],
                strongest_trivial=score['trivial']['strongest_dice'])


def study(root):
    root.mkdir(exist_ok=False)
    base = root.parent
    config_path = root / 'active-candidate.json'
    env = dict(os.environ, F_SEG_V4_CONFIG=str(config_path))
    os.environ['F_SEG_V4_CONFIG'] = str(config_path)
    protocol = TOOLS.parents[2] / 'inst/extdata/campaign/segmentation/protocol-v4.md'
    runtime = read_json(base / 'runtime-v4.json')
    if runtime['protocol_sha256'] != sha256(protocol):
        raise ValueError('preregistered protocol changed')
    gates = read_json(Path(env['F_SEG_GATES_JSON']))
    if any(gates.get(f'segmentation_6_1_{i}') is not True for i in range(1, 8)):
        raise ValueError('all mechanism gates required')
    for name in ('prepared', 'features'):
        (root / name).symlink_to(base / name, target_is_directory=True)
    splits = []
    for seed in SEEDS:
        source_path = base / f'prepared/busbra/split-{seed}.json'
        source = pinned_source_split(source_path, 'busbra', seed, PROVENANCE)
        value = inner_split(source)
        value['source_sha256'] = sha256(source_path)
        write(root / f'inner-{seed}.json', value)
        splits.append(value)
    write(root / 'launch-manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
          protocol_sha256=sha256(protocol), runtime=runtime, candidates=candidates(),
          split_sha256={str(s): sha256(root / f'inner-{s}.json') for s in SEEDS}))
    results = []
    for candidate in candidates():
        write(config_path, candidate)
        for seed, split in zip(SEEDS, splits):
            name = f"{candidate['decoder']}-batch{candidate['batch_size']}-rounds{candidate['rounds']}-seed{seed}"
            work = root / 'development' / name
            work.mkdir(parents=True)
            log = Path('/workspace/logs') / f'segmentation-v4-development-{name}.log'
            cell_env = dict(env, F_SEG_BATCH_SIZE=str(candidate['batch_size']), F_SEG_VARIANT='full',
                            F_SEG_INNER_SPLIT=str(root / f'inner-{seed}.json'))
            row = dict(candidate=candidate, seed=seed, status='running', log=str(log))
            write(work / 'execution-status.json', row)
            print(json.dumps(row), flush=True)
            try:
                prepared = base / 'prepared/busbra'
                run([str(TOOLS / 'run_federated.sh'), str(prepared), str(prepared / f'split-{seed}.json'),
                     '8', str(seed), str(work)], cell_env, log)
                run([sys.executable, str(TOOLS / 'score_public.py'), '--features', str(base / 'features/busbra'),
                     '--split', str(work / 'effective-split.json'), '--probabilities', str(work / 'public-probabilities.csv'),
                     '--artifact', str(released_artifact(work)), '--out', str(work / 'channel-b.json')], cell_env, log)
                row.update(verify_development(work, base / 'features/busbra', split, candidate), status='executed')
            except Exception as error:
                row.update(status='failed', error=f'{type(error).__name__}: {error}')
            write(work / 'execution-status.json', row)
            results.append(row)
            write(root / 'development-results.json', results)
            print(json.dumps(row), flush=True)
    selection = select(results, splits)
    write(root / 'selection.json', selection)
    candidate = selection['selected']['candidate']
    write(config_path, candidate)
    # All development is complete before the first outer-test confirmation cell.
    codes = {}
    for batch in (16, 64):
        log = Path('/workspace/logs') / f'segmentation-v4-confirmation-batch{batch}.log'
        try:
            run([sys.executable, str(TOOLS / 'run_matrix.py'), '--root', str(root), '--workers', '1',
                 '--batch-size', str(batch)], env, log)
            codes[str(batch)] = 0
        except subprocess.CalledProcessError as error:
            codes[str(batch)] = error.returncode
    write(root / 'confirmation-exit-codes.json', codes)
    summary = dict(selection=selection, v3='FAILED; retained at /workspace/segmentation/v3',
                   promotion='vetted remains FALSE; reviewer decides', arms={})
    for batch in (16, 64):
        output = root / f'evidence/batch{batch}'
        output.mkdir(parents=True)
        try:
            documents = assemble(root / f'runs-batch{batch}', PROVENANCE, runtime, protocol, batch)
            import jsonschema
            for name, document in documents.items():
                schema = 'campaign-summary-schema.json' if name == 'campaign-status.json' else 'evidence-schema.json'
                jsonschema.validate(document, read_json(protocol.parent / schema))
                write(output / name, document)
            summary['arms'][str(batch)] = documents
        except Exception as error:
            summary['arms'][str(batch)] = dict(status='failed', error=f'{type(error).__name__}: {error}')
    # V3 is read only: carry its existing evidence beside v4, never reinterpret it.
    v3_files = sorted((base / 'v3').rglob('*evidence*.json'))
    summary['v3_evidence'] = {str(p): dict(sha256=sha256(p), evidence=read_json(p)) for p in v3_files}
    write(root / 'summary-v4.json', summary)
    lines = ['# Segmentation v4 public-development study', '',
             'V3 floors remain FAILED. Registration remains vetted=FALSE.',
             'Public benchmark confirmation uses previously inspected outer test sets.', '',
             'Selected configuration: ' + json.dumps(candidate),
             'Inner-validation floor: ' + ('PASSED' if selection['selected']['floor_passed'] else 'FAILED'),
             '', 'See summary-v4.json and evidence/batch{16,64} for every cell, stratum, twin and envelope.',
             'Selected batch is primary; the other batch is sensitivity. No batch pooling.',
             'Development failures: ' + str(sum(r['status'] != 'executed' for r in results)),
             'Confirmation process exit codes: ' + json.dumps(codes)]
    lines += ['', '| Version | Batch | ε8 BUS-BRA Dice | Foreground Dice | Floor |',
              '|---|---:|---:|---:|---|']
    for version in ('v3', 'v4'):
        for batch in (16, 64):
            path = base / version / f'evidence/batch{batch}/busbra-evidence.json'
            if path.exists():
                document = read_json(path)
                high = document.get('summaries', {}).get('8', {}).get('federated_dp', {})
                dice = high.get('all', {}).get('dice', {}).get('mean', 'incomplete')
                foreground = (high.get('foreground_positive') or {}).get('dice', {}).get('mean', 'incomplete')
                floor = document.get('envelopes', {}).get('utility_floor', {}).get('pass')
                verdict = 'PASSED' if floor is True else 'FAILED' if floor is False else 'INCOMPLETE'
                lines.append(f'| {version} | {batch} | {dice} | {foreground} | {verdict} |')
            else:
                lines.append(f'| {version} | {batch} | incomplete | incomplete | INCOMPLETE |')
    (root / 'SUMMARY_V4.md').write_text('\n'.join(lines) + '\n')
    write(root / 'driver-status.json', dict(status='completed' if all(code == 0 for code in codes.values())
          and all(a.get('campaign-status.json', {}).get('status') == 'executed' for a in summary['arms'].values())
          else 'completed_with_failures', confirmation_exit_codes=codes,
          finished_at=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.root.exists():
        parser.error("refusing to overwrite an existing v4 study")
    try:
        study(args.root.resolve())
    except Exception as error:
        if args.root.exists() and not (args.root / 'driver-status.json').exists():
            write(args.root / 'driver-status.json', dict(status='failed', error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
