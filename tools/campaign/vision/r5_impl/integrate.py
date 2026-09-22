#!/usr/bin/env python3
"""Integrate public R5 records, preserving all earlier cell evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main(repo):
    evidence=repo/'inst/extdata/campaign/vision'
    tools=repo/'tools/campaign/vision/r5_impl'
    r5=read(evidence/'busbra_r5_summary.json')
    summary=read(evidence/'summary.json')
    assert not any(c.get('phase')=='dp_aware_selected_schedule' for c in summary['cells'])
    for cell in summary['cells']:
        cell['phase']='registry_defaults'
        cell['interpretation_status']='schedule_limited'
        cell['interpretation']='R3 schedule-limited at registry defaults; central was a finite-schedule twin (AUC 0.596183 ± 0.041527), not a converged representation comparator. Original JSON unchanged.'
    summary['cells'] += r5['cells']
    summary['r4_status']='stopped_before_test_access'
    summary['r4_interpretation']='Selection lesson: non-private pruning does not transfer under DP. Actual inner-validation AUC 0.500940 / 0.478683; stopped final matrix, no R4 held-out score.'
    summary['diagnostic_cells']=[dict(phase='r4_nonprivate_pruning',epsilon=8,
        scoring_scope='171 inner-validation training-cohort patients; no outer test',
        file='../../../../tools/campaign/vision/r4/diagnosis/dp-confirmation.json',
        status='diagnosis_only',auc=[r['inner_validation']['auc'] for r in read(repo/'tools/campaign/vision/r4/diagnosis/dp-confirmation.json')],
        interpretation=summary['r4_interpretation']),
        dict(phase='r5_dp_aware_selection',epsilon=8,file='r5/selection.json',status='diagnosis_only',
            interpretation='736 DP-emulated schedules; top three real confirmations; selected by highest actual inner-validation AUC, training patients only.')]
    summary['r5_status']='executed'
    summary['r5_summary_file']='busbra_r5_summary.json'
    summary['r5_protocol_sha256']=r5['protocol_sha256']
    summary['r5_model_params']=r5['model_params']
    summary['r5_wall_clock']=r5['wall_clock']
    summary['scored_replicates']=18
    summary['scored_replicates_by_phase']=dict(r3=9,r4=0,r5=9)
    summary['summary_scope']='Original top-level release/host/wall_clock describe R3; each R5 cell and busbra_r5_summary.json carry current runtime and timing. R4 diagnosis has no held-out scores.'
    summary['updated_at']=datetime.now(timezone.utc).isoformat()
    save(evidence/'summary.json',summary)
    readme=evidence/'README.md'
    readme.write_text(readme.read_text().replace('R5 is declared before training:', 'R5 was declared before training and is now executed:') +
        '\n'+(evidence/'busbra_r5_report.md').read_text())
    original=read(tools/'preservation-before.json')
    allowed={'inst/extdata/campaign/vision/README.md','inst/extdata/campaign/vision/summary.json'}
    changed=[name for name,h in original.items() if name not in allowed and hashlib.sha256((repo/name).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    preservation=dict(existing_files_verified=len(original)-len(allowed),all_prior_records_byte_identical=True,
        intentional_updated_indexes=sorted(allowed),preexisting_r4_working_edits_preserved=True)
    save(evidence/'busbra_r5_preservation.json',preservation)
    hashes={str(p.relative_to(evidence)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(evidence.glob('busbra_r5_*')) if p.is_file()}
    hashes.update({str(p.relative_to(evidence)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((evidence/'r5_impl').rglob('*')) if p.is_file()})
    save(evidence/'busbra_r5_sha256.json',hashes)
    print(json.dumps(preservation))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);main(p.parse_args().repo)
