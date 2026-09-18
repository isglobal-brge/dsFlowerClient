#!/usr/bin/env python3
"""Aggregate only executed survival evidence; failures are never imputed."""
import argparse
import collections
import datetime
import json
from pathlib import Path
from metrics import envelopes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('evidence',type=Path)
    args=parser.parse_args()
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    failures=[]
    for path in sorted(args.evidence.glob('*.json')):
        record=json.loads(path.read_text())
        if record.get('record_type')=='summary':
            continue
        if record['status']=='failed':
            failures.append({'file':path.name,'reason':record.get('error')})
            continue
        if record['status']!='executed':
            raise ValueError('only executed or failed attempts may be archived')
        meta=record['dataset']
        if meta['dataset']=='synthetic-public':
            continue
        key=(meta['dataset'],meta['subset'],record['variant'])
        groups[key][record['epsilon']].append(record['results'])
    expected={(dataset,subset,variant) for dataset,subsets in [('support2',['full','small600','heterogeneous']),('lung1',['full'])] for subset in subsets for variant in ['weibull','lognormal','hazard']}
    if set(groups)-expected:
        raise ValueError('unexpected matrix group')
    summaries=[]
    for (dataset,subset,variant),by_epsilon in sorted(groups.items()):
        required=[8] if subset=='heterogeneous' else [1,4,8]
        complete=(sorted(by_epsilon)==required and
                  all(sorted(c['seed'] for c in by_epsilon[e])==[1101,1102,1103] for e in required))
        if not complete:
            summaries.append({'dataset':dataset,'subset':subset,'variant':variant,
                              'status':'incomplete','executed_replicates':{str(e):len(c) for e,c in by_epsilon.items()}})
            continue
        summaries.append({'dataset':dataset,'subset':subset,'variant':variant,
                          'status':'executed',
                          'envelopes':envelopes(by_epsilon,primary=dataset=='support2' and subset=='full',small_n=subset=='small600')})
    for group in summaries:
        if group['status']=='executed':
            env=group['envelopes']
            keys=[f"epsilon_envelope_{x['from']}_{x['to']}" for x in env['epsilon_envelope'] if x['flag']]
            if env['small_n_trend'] and env['small_n_trend']['flag']:
                keys.append('small_n_trend')
            keys += [f"near_central_{x['epsilon']}_{x['seed']}" for x in env['near_central'] if x['historic_flag'] or x['minimum_site_companion_flag']]
            group['investigations']={key:{'status':'pending','explanation':'Requires matched artifact hash and independent full-horizon mechanism review.'} for key in keys}
    result={'schema_version':1,'record_type':'summary','task':'survival',
            'status':'executed' if set(groups)==expected and all(s['status']=='executed' for s in summaries) else 'incomplete',
            'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'expected_matrix_cells':90,'groups':summaries,'failed_attempts':failures,
            'envelope_interpretation':'Empirical utility diagnostics, not privacy proofs; G=central-federated reverses historic delta.',
            'promotion':'Reviewer decision; floor failures are retained.'}
    (args.evidence/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],len(summaries),'groups',len(failures),'failed attempts')


if __name__=='__main__':
    main()
