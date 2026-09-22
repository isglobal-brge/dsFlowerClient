#!/usr/bin/env python3
"""Validate v3 evidence and summarize only its single locked confirmation."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import mean_ci
from validate_completion import mechanism,require,digest
from protocol import SEEDS,sha


def validate(record):
    require(record['schema_version']==1 and record['protocol_version']==3,'schema/version')
    require(record['delta']==1e-5 and record['clip']==1 and record['privacy_unit']=='patient' and record['adjacency']=='replace_one','privacy pins')
    require(record['status']=='executed' and record['cleanup_ok'],'executed and clean required')
    cfg=record['public_config'];meta=record['dataset'];result=record['results'];fed=record['federation']
    require(record['package_versions']['dsFlower']==record['package_versions']['dsFlowerClient']=='0.5.0','package versions')
    require(fed['n_clients']==record['site_count']==len(meta['sites']),'site count')
    require(fed['n_rounds_run']==cfg['num-server-rounds'] and fed['n_failures']==0,'complete rounds')
    require(record['artifact_checksum']==result['model_sha256']==fed['model_sha256'] and digest(record['artifact_checksum']),'model hash')
    require(record['installed_build']['runner_sha256']==record['runner_sha256'],'runner identity')
    require(sum(s['n_subjects'] for s in meta['sites'])==meta['n_train'],'patient census')
    for site in result['site_mechanisms']:
        n=next(s['n_subjects'] for s in meta['sites'] if s['site']==site['site'])
        mechanism(site,n,cfg,record['epsilon'],record['delta'])
    if record['hazard_v3_phase']!='development':
        mechanism(result['pooled_mechanism'],meta['n_train'],cfg,record['epsilon'],record['delta'])
    for branch in ('federated_dp',) if record['hazard_v3_phase']=='development' else ('federated_dp','central_dp','central','null'):
        score=result[branch]
        require(isinstance(score['c_index'],(int,float)) and 0<=score['c_index']<=1,'finite score')
        require(score['n_evaluated_public_subjects']+score['n_invalid_public_subjects']==meta['n_test'],'evaluation census')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive',type=Path);args=ap.parse_args();root=args.archive
    selection=json.loads((root/'selection.json').read_text())
    begun=json.loads((root/'confirmation_started.json').read_text())
    require(begun['selection_sha256']==sha(root/'selection.json'),'selection lock')
    require(selection['selected_utc']<=begun['started_utc'],'selection order')
    for name,value in selection['development_sha256'].items():
        path=root/'development'/name;require(sha(path)==value,'development evidence hash');validate(json.loads(path.read_text()))
    groups=defaultdict(list);failures=[]
    expected={(arm,eps,seed) for arm in selection['confirmation_arms'] for eps in ([1,4,8] if arm=='full' else [8]) for seed in SEEDS}
    seen=set()
    for path in sorted(root.glob('cell-support2-*.json')):
        record=json.loads(path.read_text());meta=record['dataset'];key=(meta['subset'],record['epsilon'],meta['seed'])
        require(key in expected and key not in seen,'duplicate/unexpected confirmation');seen.add(key)
        require(record['hazard_v3_config']==selection['per_seed_configs'][str(meta['seed'])],'configuration changed')
        if record['status']!='executed':failures.append(dict(file=path.name,error=record['error']));continue
        validate(record)
        require(record['started_utc']>=begun['started_utc'][:19]+'Z','confirmation predates lock')
        groups[key[:2]].append(record)
    output=[]
    for (arm,eps),records in sorted(groups.items()):
        require(sorted(r['dataset']['seed'] for r in records)==list(SEEDS),'incomplete seed group')
        result=dict(arm=arm,epsilon=eps,n_train=records[0]['dataset']['n_train'],
            n_sites=records[0]['site_count'],site_n=[s['n_subjects'] for s in records[0]['dataset']['sites']],
            c_index={b:mean_ci([r['results'][b]['c_index'] for r in records]) for b in ('federated_dp','central_dp','central','null')},
            heldout_nll={b:mean_ci([r['results'][b]['heldout_nll'] for r in records]) for b in ('federated_dp','central_dp','central','null')},
            pooled_dp_minus_federated=mean_ci([r['results']['central_dp']['c_index']-r['results']['federated_dp']['c_index'] for r in records]),
            central_minus_federated=mean_ci([r['results']['central']['c_index']-r['results']['federated_dp']['c_index'] for r in records]))
        output.append(result)
    primary=next((r for r in output if r['arm']=='full' and r['epsilon']==8),None)
    verdict='UNAVAILABLE' if primary is None else ('PASS' if primary['c_index']['federated_dp']['mean']>=.6 and primary['c_index']['federated_dp']['mean']>=primary['c_index']['null']['mean']+.05 else 'FAIL')
    summary=dict(schema_version=1,record_type='summary',task='survival',protocol_version=3,
        status='executed' if seen==expected and not failures else 'incomplete',date_utc=datetime.now(timezone.utc).isoformat(),
        selected=selection['selected'],selection_sha256=sha(root/'selection.json'),confirmation_number=3,
        expected_matrix_cells=len(expected),observed_cells=len(seen),groups=output,failed_attempts=failures,
        development_failed_attempts=[dict(file=str(p.relative_to(root)),error=json.loads(p.read_text()).get('error'))
            for p in sorted((root/'failures').glob('cell-*.json'))],
        three_site_verdict=verdict,absolute_floor=.60,null_margin=.05,
        caveat='Third confirmation on reused outer holdouts; overlapping split replicates, not independent clinical validation; envelopes do not change admission.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('CONFIRMATION',summary['status'],verdict,flush=True)
    for row in output:
        print(row['arm'],row['epsilon'],{k:round(v['mean'],6) for k,v in row['c_index'].items()})


if __name__=='__main__':main()
