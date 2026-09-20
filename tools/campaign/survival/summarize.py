#!/usr/bin/env python3
"""Aggregate only executed survival evidence; failures are never imputed."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
from metrics import envelopes
from validate_completion import same


def selected_hazard_records(evidence):
    """Read the frozen selection and only its complete outer confirmation matrix."""
    root=evidence/'hazard-v2'
    selection_path=root/'hazard_v2_selection.json'
    selection=json.loads(selection_path.read_text())
    completed=json.loads((root/'driver_complete.json').read_text())
    if completed['status']!='executed' or completed['selection_sha256']!=hashlib.sha256(selection_path.read_bytes()).hexdigest():
        raise ValueError('hazard v2 completion does not match selection')
    expected={(dataset,subset,epsilon,seed)
              for dataset,subset in [('support2','full'),('support2','small600'),
                                     ('support2','heterogeneous'),('lung1','full')]
              for epsilon in ([8] if subset=='heterogeneous' else [1,4,8])
              for seed in [1101,1102,1103]}
    records=[]
    observed=set()
    for path in sorted(root.glob('confirmation-*/evidence.json')):
        record=json.loads(path.read_text())
        meta=record['dataset']
        identity=(meta['dataset'],meta['subset'],record['epsilon'],meta['seed'])
        if (record['status']!='executed' or record['variant']!='hazard' or
            record['protocol_version']!=2 or record['hazard_v2_config']!=selection['selected'] or
            identity not in expected or identity in observed):
            raise ValueError('invalid or duplicate selected hazard confirmation: '+str(path))
        observed.add(identity)
        records.append((path,record))
    if observed!=expected:
        raise ValueError('selected hazard summary requires all 30 confirmation cells')
    return selection,records


def write_release_markdown(evidence, result):
    selected=result['hazard_v2_selection']['selected']
    lines=['# Survival campaign evidence summary','',
           'Current rows combine frozen v1 AFT evidence with the selected hazard-v2 confirmation. '
           'The 90 v1 cohort cells, three synthetic successes and seven failed attempts remain unchanged; '
           '`v1/summary.json` preserves their original report. The current report uses 60 AFT cells and '
           '30 hazard-v2 cells, with three matched seeds (1101, 1102, 1103) per row.','',
           f"Hazard selection: {selected['id']}, {selected['rounds']} rounds × {selected['local_epochs']} local epochs, "
           f"batch {selected['batch_size']}, SGD learning rate {selected['learning_rate']}, K={selected['K']} equal-width bins over 1825 days. "
           'Selection used the highest mean inner-validation C-index over twelve cells per candidate; '
           'all six candidates and 72 development scores are retained in `hazard-v2/hazard_v2_selection.json`. '
           'Confirmation reuses the v1 outer holdouts and is not independent validation. '
           'The per-run privacy budgets do not account for the whole development sweep.','',
           'C-index and held-out NLL are mean ± sample SD over three public split replicates. '
           'Student-t 95% intervals are in `summary.json`; overlapping splits and only three replicates limit their interpretation. '
           'Compare NLL only within the matching likelihood/grid and against its matching null. '
           'Ranking utility does not establish probability calibration.']
    for title,nll in [('C-index',False),('Held-out NLL',True)]:
        lines+=['',f'## {title} — mean ± SD','',
                '| Cohort / subset | Variant | Protocol | ε | Federated-DP | Pooled-DP | Pooled-nonprivate | Null |',
                '|---|---|---|---:|---:|---:|---:|---:|']
        for group in result['groups']:
            for epsilon,row in group['envelopes']['summaries'].items():
                scores=row['heldout_nll'] if nll else row
                names=['federated_dp' if nll else 'federated','central_dp','central','null']
                values=' | '.join(f"{scores[name]['mean']:.3f} ± {scores[name]['sd']:.3f}" for name in names)
                protocol='v2 / '+selected['id'] if group['variant']=='hazard' else 'v1'
                lines.append(f"| {group['dataset']} / {group['subset']} | {group['variant']} | {protocol} | {epsilon} | {values} |")
    lines+=['','## Preregistered utility diagnostics','',
            'These are empirical diagnostics, not privacy proofs. Shortfall is G = pooled-nonprivate − federated-DP, '
            'the reverse of the historic delta sign. A FLAG is retained for review.','',
            '| Cohort / subset | Variant | Adjacent-ε envelope | ε=8 designated floor | Small-N trend | Near-central flags (historic / minimum-site) |',
            '|---|---|---|---|---|---|']
    for group in result['groups']:
        env=group['envelopes']
        adjacent=', '.join(f"{x['from']}→{x['to']}: {'FLAG' if x['flag'] else 'PASS'}" for x in env['epsilon_envelope']) or 'N/A'
        floor=env['utility_floor']
        verdict=('PASS' if floor['pass'] else 'FAIL') if floor else 'N/A'
        trend=env['small_n_trend']
        small=('FLAG' if trend['flag'] else 'PASS') if trend else 'N/A'
        historic=sum(x['historic_flag'] for x in env['near_central'])
        companion=sum(x['minimum_site_companion_flag'] for x in env['near_central'])
        lines.append(f"| {group['dataset']} / {group['subset']} | {group['variant']} | {adjacent} | {verdict} | {small} | {historic} / {companion} |")
    lines+=['','V1 AFT investigation notes are preserved; new hazard flags remain pending reviewer investigation. '
            'Designated floors require both C-index ≥ 0.60 and C-index ≥ null + 0.05; failures remain failures.','',
            '## Retained failed attempts','', '| Evidence record | Recorded cause |','|---|---|']
    for failed in result['failed_attempts']:
        reason=(failed['reason'] or '').replace('|','\\|').replace('\n',' ')
        lines.append(f"| `{failed['file']}` | {reason} |")
    lines+=['','Raw h06 confirmation cells and preregistration/selection/completion records are under `hazard-v2/`. '
            '`hazard-v2/source-digests.json` records their pod2 source paths and SHA-256 digests; '
            '`v1/source-digests.json` records all original v1 JSON digests. No cells were rerun or scores changed.','']
    (evidence/'SURVIVAL_EVIDENCE_SUMMARY.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('evidence',type=Path)
    parser.add_argument('--hazard-v2',action='store_true')
    parser.add_argument('--selected-hazard-v2',action='store_true',
                        help='combine preserved v1 AFT with packaged hazard-v2 confirmations')
    args=parser.parse_args()
    if args.hazard_v2 and args.selected_hazard_v2:
        parser.error('hazard-only and combined release summaries are separate modes')
    if (args.evidence/'hazard-v2').is_dir() and not args.selected_hazard_v2:
        parser.error('packaged hazard-v2 evidence requires --selected-hazard-v2; v1 summary is preserved under v1/')
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    failures=[]
    records=[(path,json.loads(path.read_text())) for path in sorted(args.evidence.glob('*.json'))]
    if args.selected_hazard_v2:
        previous=json.loads((args.evidence/'v1/summary.json').read_text())
        selection,hazard=selected_hazard_records(args.evidence)
        records=[(path,record) for path,record in records
                 if record.get('variant')!='hazard' or record.get('status')=='failed']+hazard
    for path,record in records:
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
    if args.hazard_v2:
        expected={group for group in expected if group[2]=='hazard'}
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
        for cells in by_epsilon.values():
            cells.sort(key=lambda cell:cell['seed'])
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
            if args.selected_hazard_v2 and group['variant']!='hazard':
                original=next(g for g in previous['groups']
                              if (g['dataset'],g['subset'],g['variant'])==(group['dataset'],group['subset'],group['variant']))
                same(env,original['envelopes'],'preserved AFT envelopes')
                group['investigations']=original['investigations']
    result={'schema_version':1,'record_type':'summary','task':'survival',
            'status':'executed' if set(groups)==expected and all(s['status']=='executed' for s in summaries) else 'incomplete',
            'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'protocol_version':2 if args.hazard_v2 else 1,
            'expected_matrix_cells':30 if args.hazard_v2 else 90,'groups':summaries,'failed_attempts':failures,
            'envelope_interpretation':'Empirical utility diagnostics, not privacy proofs; G=central-federated reverses historic delta.',
            'promotion':'Reviewer decision; floor failures are retained.'}
    if args.selected_hazard_v2:
        result.update(protocol_version='v1-aft+v2-hazard',
                      evidence_scope='60 frozen v1 AFT cells and 30 selected hazard-v2 confirmation cells; v1 hazard retained as history',
                      hazard_v2_selection={key:selection[key] for key in ['selected','rule','protocol_sha256','selected_utc']})
    (args.evidence/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if args.selected_hazard_v2:
        write_release_markdown(args.evidence,result)
    print(result['status'],len(summaries),'groups',len(failures),'failed attempts')


if __name__=='__main__':
    main()
