#!/usr/bin/env python3
"""Combine arm execution status without pooling scores or choosing an arm."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import jsonschema
from assemble_evidence import planned_cells, read_json, sha256


def status_for(cells):
    states = {cell['status'] for cell in cells}
    return 'executed' if states == {'executed'} else 'failed' if 'failed' in states else 'not_executed'


def combine(root):
    digest = sha256(root / 'protocol.md')
    schema = read_json(root / 'campaign-summary-schema.json')
    cells, arms = [], {}
    for batch in (16, 64):
        path = root / f'batch{batch}/campaign-status.json'
        arm = read_json(path)
        jsonschema.validate(arm, schema)
        if arm['protocol_sha256'] != digest or arm.get('nominal_batch_size') != batch:
            raise ValueError('arm identity or protocol mismatch')
        keys = [(c['dataset'], c['variant'], c['epsilon'], c['seed']) for c in arm['cells']]
        if len(keys) != len(set(keys)) or set(keys) != set(planned_cells()):
            raise ValueError('arm omits or duplicates planned cells')
        if arm['status'] != status_for(arm['cells']):
            raise ValueError('arm status contradicts cell outcomes')
        cells.extend(dict(cell, nominal_batch_size=batch) for cell in arm['cells'])
        arms[str(batch)] = dict(path=str(path.relative_to(root)), sha256=sha256(path), status=arm['status'])
    return dict(schema='dsflower-segmentation-campaign-combined-v2',
        contract='pytorch_resnet18_segmentation', task='segmentation',
        protocol_sha256=digest, executed_at=datetime.now(timezone.utc).isoformat(),
        status=status_for(cells), cells=cells, arms=arms,
        reason='Both preregistered arms retained separately; inspect all failed or missing cells and each cohort archive',
        promotion='Reviewer decision; execution status does not assert validated utility')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    value = combine(args.root)
    jsonschema.validate(value, read_json(args.root / 'campaign-combined-schema.json'))
    (args.root / 'campaign-status.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':value['status'], 'cells':len(value['cells'])}))


if __name__ == '__main__':
    main()
