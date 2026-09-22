#!/usr/bin/env python3
"""Assemble R4 reporting from scored records only; no model/data access."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text())


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(repo,public):
    target=repo/'inst/extdata/campaign/sequence'
    tools=repo/'tools/campaign/sequence/r4'
    incoming=read(public/'evidence/summary.json')
    summary=read(target/'summary.json')
    assert len(summary['cells'])==9
    for cell in summary['cells']:
        assert hashlib.sha256((target/cell['file']).read_bytes()).hexdigest()==cell['sha256']
    for cell in incoming['cells']:
        shutil.copy2(public/'evidence'/cell['file'],target/cell['file'])
        assert hashlib.sha256((target/cell['file']).read_bytes()).hexdigest()==cell['sha256']
    decomposition=read(tools/'diagnosis/decomposition.json')
    annotation=('R4 TRAIN-only one-seed diagnosis: finite-schedule -0.070398, net federation-path '
        '+0.034791, clipping -0.076769, marginal-noise estimate -0.086579 macro-AUC; total -0.198954. '
        'The Poisson/normalization bridge is +0.042903, followed by a remaining federation contrast '
        '-0.008112. Ordered controls do not reallocate the historical held-out gap, and sampling '
        'variation remains in the noise contrast. Full central-to-DP gap is not pure privacy cost.')
    for cell in summary['cells']:
        if cell['file'].startswith('har_window_'):
            cell['r4_diagnosis_annotation']=annotation
            cell['r4_training_only_decomposition']=decomposition
    summary['cells'].extend(incoming['cells'])
    summary['n_scored_cells']=12
    summary['interpretations']['r3_window_diagnosis']=annotation
    summary['interpretations']['r4_window']=incoming['cells'][0]['interpretation']
    summary['r4']=dict(declaration_commit=(tools/'declaration_commit.txt').read_text().strip(),
        selection_file='../../../../tools/campaign/sequence/r4/selection/selection.json',
        protocol_sha256=hashlib.sha256((tools/'implementation_protocol.json').read_bytes()).hexdigest(),
        scoring_marker_sha256=hashlib.sha256((public/'test-scoring-started.json').read_bytes()).hexdigest(),
        central_reuse='Identical R3 central model identities and scores; no retraining or rescoring.',
        optional_pooled_dp='Not run; prioritized three-site DP and both matched noiseless twins.',
        pod_left_running=True)
    summary['reporting']=dict(created_at=datetime.now(timezone.utc).isoformat(),
        source='Existing R1/R3 records plus once-scored R4 records only; no data access or recomputation.')
    save(target/'summary.json',summary)
    audit=target/'r4'
    audit.mkdir(exist_ok=False)
    for name in ('protocol.json','pre_scoring_verification.json','test-scoring-started.json','runtime.json'):
        shutil.copy2(public/name,audit/name)
    shutil.copytree(public/'runs',audit/'runs')
    shutil.copytree(public/'logs',audit/'logs')
    for path in (public/'evidence').glob('*-score.json'):
        shutil.copy2(path,audit/path.name)
    shutil.copy2(public/'evidence/summary.json',audit/'scored-summary.json')
    shutil.copy2(tools/'selection/selection.json',audit/'selection.json')
    selection=read(tools/'selection/selection.json')
    lines=['## R4 once-scored results','',
        'The selected schedule and all new training were frozen before the exclusive',
        'TEST scoring marker. All nine federations passed their 135 node-round checks;',
        'six matched noiseless twins were scored once. The R3 central model identities',
        'and scores are reused on the byte-identical split. No scored model was retrained.',
        '', 'Values below are mean ± sample SD over three seeds. The AUC gap is paired',
        'federated-DP minus central. All scores use the held-out subjects’ windows.', '']
    arms=['central','nonprivate_federated','clipped_noiseless','federated_dp','trivial']
    labels=['Central (R3)','Non-private federated','Clipped noiseless','Federated-DP','Trivial']
    def fmt(obj):return f"{obj['mean']:.6f} ± {obj['sd']:.6f}"
    for metric,title in [('macro_auc','Macro one-vs-rest AUC'),('accuracy','Accuracy'),('log_loss','Log-loss')]:
        lines += [f'### {title}','','| ε | '+' | '.join(labels)+' |'+(' DP − central |' if metric=='macro_auc' else ''),
                  '|---:|'+'---:|'*len(arms)+('---:|' if metric=='macro_auc' else '')]
        for cell in incoming['cells']:
            s=cell['summary']
            lines.append('| '+str(cell['epsilon'])+' | '+' | '.join(fmt(s[a][metric]) for a in arms)+' |'+
                         (' '+fmt(s['gap_macro_auc'])+' |' if metric=='macro_auc' else ''))
        lines+=['']
    lines+=['Pooled-DP was not run. Accuracy-versus-majority and chance-AUC checks are',
        'annotations only. No alternative schedule or rerun followed these outcomes.', '',
        'The R3 diagnosis measures clipping loss −0.076769 and a further marginal-noise',
        'estimate −0.086579 on the inner split. Its ordered schedule/federation terms',
        'depend on sampling and optimizer resets; they do not reallocate the historical',
        'held-out gap. See the [complete diagnosis](../../../../tools/campaign/sequence/r4/SEQUENCE_DIAGNOSIS_R4.md).','',
        'Public evidence includes node-reported privacy settings, independent full-horizon',
        'accounting, all per-seed arm metrics, model/runner hashes and wall-clock timings.',
        f"Declaration commit: `{summary['r4']['declaration_commit']}`.",
        'The sequence pod remains running. No package code was changed.','']
    readme=target/'README.md'
    text=readme.read_text().replace('All nine scored cells are indexed in','The nine pre-R4 scored cells are indexed in')
    readme.write_text('\n'.join(lines)+'\n---\n\n'+text)
    # Manifest only the new implementation evidence; original evidence is immutable.
    paths=[p for p in target.glob('har_r4_window_*.json')]+list(audit.rglob('*.json'))
    (audit/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(target)}\n' for p in sorted(paths)))
    print(json.dumps([dict(epsilon=c['epsilon'],summary=c['summary']) for c in incoming['cells']],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    a=p.parse_args()
    main(a.repo,a.public)
