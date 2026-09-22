#!/usr/bin/env python3
"""Confirm the two TRAIN-only candidates through actual dsFlower federations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from dsflower_runner import params, validation
from diagnose import controls, extract, metrics, save, sha
from stage_training import stage


def main(root, resume_startup_failure=False):
    controls()
    tools=Path(__file__).resolve().parent
    out=root/'r4/diagnosis'
    shortlist=json.loads((out/'shortlist.json').read_text())
    with np.load(out/'training-features.npz') as data:
        x,y=data['x'],data['y']
    with np.load(out/'inner-indices.npz') as data:
        val=data['validation']
    results=[]
    start=time.monotonic()
    staged=Path('/tmp/cells-vision-r4/inner-training')
    if resume_startup_failure:
        assert (out/'training-staging.json').exists() and (staged/'staging-audit.json').exists()
        assert not (out/'dp-confirmation.json').exists()
    else:
        save(out/'training-staging.json',stage(out/'collections',staged))
    for candidate in shortlist:
        name=candidate['id']; job_path=out/f'job-{name}.json'
        job=json.loads(job_path.read_text()); run=Path(job['run_dir'])
        if run.exists():
            failed=json.loads((run/'federation-status.json').read_text())
            assert resume_startup_failure and failed['status']=='failed' and failed['cleanup_ok']
            assert failed['error'].startswith('SuperLink did not become ready within 15 seconds.')
            assert not list((run/'public-capture').glob('*'))
            run=run.with_name(run.name+'-startup-retry')
            job['run_dir']=str(run)
        job['collection_root']=str(staged)
        job_path=out/f'staged-job-{run.name}.json';save(job_path,job)
        print('INNER_FEDERATION_START',name,datetime.now(timezone.utc).isoformat(),flush=True)
        subprocess.run(['Rscript',str(tools/'run_federated.R'),str(root),str(job_path)],check=True,timeout=2400)
        status=json.loads((run/'federation-status.json').read_text())
        assert status['status']=='trained_unscored' and status['cleanup_ok']
        initial=json.loads((run/'public-capture/public-initial.json').read_text())
        cfg=initial['config']
        xs,ys,_,_,hashes=extract(Path(job['collection_root']),cfg)
        expected={(hashlib.sha256(a.tobytes()).hexdigest(),hashlib.sha256(b.tobytes()).hexdigest()) for a,b in zip(xs,ys)}
        captures=[json.loads(p.read_text()) for p in (run/'public-capture').glob('accountant-*.json')]
        assert len(captures)==15
        assert not list((run/'public-capture').glob('failure-*.json'))
        for capture in captures:
            pins=capture['training_pins']
            assert all(pins[k]==candidate[k] for k in ('learning_rate','batch_size','local_epochs'))
            assert pins['optimizer']['name']==candidate['optimizer']
            assert pins['optimizer']['weight_decay']==candidate['weight_decay']
            assert capture['observed_round_steps']==160 and capture['mechanism']['total_steps']==800
        assert {(c['features_sha256'],c['targets_sha256'],c['round']) for c in captures}=={(a,b,r) for a,b in expected for r in range(1,6)}
        assert all(c['mechanism']['accounting_population']==227 and c['privacy_config']['epsilon']==8
            and c['privacy_config']['delta']==1e-6 and c['privacy_config']['clipping_norm']==1 for c in captures)
        model=params.load_user_model(cfg,512,'cross_entropy')
        model.load_state_dict(torch.load(Path(status['output_dir'])/'model.pt',map_location='cpu',weights_only=True))
        probability=np.asarray(validation.neural_predictions(model,x[val],'cross_entropy'))[:,1]
        row=dict(candidate=candidate,inner_validation=metrics(y[val],probability),n_fit_patients=681,n_validation_patients=171,
            actual_federation=True,epsilon=8,delta=1e-6,clipping_norm=1,rounds=5,sites=3,
            mechanism=captures[0]['mechanism'],feature_parity=True,elapsed_s=status['elapsed_s'],test_accessed=False,
            federation_status_sha256=sha(run/'federation-status.json'),model_sha256=status['model_sha256'])
        save(run/'inner-validation.json',row); results.append(row); save(out/'dp-confirmation.json',results)
        print('INNER_VALIDATION',json.dumps(row),flush=True)
    selected=sorted(results,key=lambda r:(-r['inner_validation']['auc'],r['candidate']['id']))[0]
    save(out/'selection.json',dict(selected=selected,all_candidates=results,test_accessed=False,
        selected_at=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-start,
        selection_rule='Maximum actual federated-DP patient inner-validation AUC at epsilon 8; tie candidate id ascending.'))
    print('SCHEDULE_SELECTED',json.dumps(selected),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume-startup-failure',action='store_true')
    args=p.parse_args();main(args.root,args.resume_startup_failure)
