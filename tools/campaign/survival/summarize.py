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
    result={'schema_version':1,'record_type':'summary','task':'survival',
            'status':'executed' if len(summaries)==12 and all(s['status']=='executed' for s in summaries) else 'incomplete',
            'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'expected_matrix_cells':90,'groups':summaries,'failed_attempts':failures,
            'envelope_interpretation':'Empirical utility diagnostics, not privacy proofs; G=central-federated reverses historic delta.',
            'promotion':'Reviewer decision; floor failures are retained.'}
    (args.evidence/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],len(summaries),'groups',len(failures),'failed attempts')


if __name__=='__main__':
    main()
