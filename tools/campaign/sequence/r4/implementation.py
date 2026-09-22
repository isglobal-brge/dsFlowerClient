#!/usr/bin/env python3
"""Frozen TRAIN-only selection and once-only R4 training; no TEST reader."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from dsflower_runner import client_app, server_app
from emulate import arrays, federated
from implementation_verify import verify_run, sha

TOOLS = Path(__file__).resolve().parent

SEEDS = [20260922, 20260923, 20260924]
CANDIDATES = [dict(name='adam01_e4_b512', optimizer='adam', lr=.01, epochs=4, batch=512),
              dict(name='adam003_e8_b256', optimizer='adam', lr=.003, epochs=8, batch=256)]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def protocol(schedule):
    return dict(rounds=5, delta=1e-6, clipping_norm=1, seeds=SEEDS, epsilon_order=[1,4,8], units=['window'],
        model_params=dict(n_tokens=128, n_features=9, n_classes=6, hidden=32,
            optimizer=schedule['optimizer'], learning_rate=schedule['lr'], batch_size=schedule['batch'],
            local_epochs=schedule['epochs'], scheduler='none', weight_decay=0, l1_penalty=0))


def run_contract(root, work, prepared, schedule, seed, epsilon, inner):
    assert not (root/'r4/implementation/test-scoring-started.json').exists()
    assert not work.exists()
    request = dict(work=str(work), prepared=str(prepared), seed=seed, epsilon=epsilon,
                   protocol=protocol(schedule), inner_selection=inner, test_accessed=False)
    request_path=work.parent/(work.name+'-request.json')
    save(request_path, request)
    with (work.parent/(work.name+'.log')).open('x') as log:
        subprocess.run(['Rscript', str(TOOLS/'run_implementation.R'), str(root), str(request_path)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    save(work/'protocol.json', request['protocol'])
    save(work/'execution-status.json', dict(status='trained_unscored', protocol_sha256=sha(work/'protocol.json')))
    audit=read(prepared/'audit.json')
    data=np.load(prepared/'train.npz')
    result=verify_run(work, request['protocol'], audit, data, 'window', epsilon, seed)
    save(work/'verification.json', dict(status='passed', captured_node_rounds=15,
        test_accessed=False, accounting=result['accounting'], model_sha256=sha(result['checkpoint'])))
    return result


def selection(root):
    work=root/'r4/implementation/selection'
    work.mkdir(parents=True, exist_ok=False)
    prepared=root/'r4/prepared'
    audit=read(prepared/'audit.json')
    plan=dict(candidates=CANDIDATES, seeds=SEEDS, epsilon=8, rounds=5, delta=1e-6, clipping_norm=1,
        train_subjects=audit['train_subjects'], validation_subjects=audit['validation_subjects'],
        site_subjects=audit['site_subjects'], n_per_site=audit['n_per_site'],
        pruning='Top two of the diagnosis noiseless-clipped endpoint ranking; no new pruning fits.',
        criterion='Maximum mean final-round real federated-DP macro-AUC over the three seeds; ties choose candidate order.',
        test_accessed=False)
    save(work/'plan.json', plan)
    val=np.load(prepared/'validation.npz')
    sys.path.insert(0,str(root/'Rlib/dsFlowerClient/python'))
    from predict_helper import predict_pytorch_spec
    from score_and_assemble import metrics, mean_sd
    records=[]
    for schedule in CANDIDATES:
        reps=[]
        for seed in SEEDS:
            name=f"{schedule['name']}-seed{seed}"
            print('INNER_START', name, flush=True)
            result=run_contract(root,work/name,prepared,schedule,seed,8,True)
            cfg=result['initial']['config']
            vx=client_app._apply_feature_bounds(val['X'],cfg)
            probs=np.asarray(predict_pytorch_spec(str(result['checkpoint']),vx,'prob',cfg['model-spec-b64'],
                                                  'cross_entropy',num_classes=6))
            rep=dict(seed=seed, metrics=metrics(val['y'],probs),model_sha256=sha(result['checkpoint']),
                elapsed_s=result['status']['elapsed_s'])
            save(work/name/'inner-metrics.json',rep)
            reps.append(rep)
            print('INNER_COMPLETE',name,rep['metrics'],flush=True)
        record=dict(schedule=schedule, per_replicate=reps,
                    summary={k:mean_sd([r['metrics'][k] for r in reps]) for k in ('macro_auc','accuracy','log_loss')})
        records.append(record)
        save(work/'sweep.json',records)
    selected=max(records,key=lambda r:r['summary']['macro_auc']['mean'])
    save(work/'selection.json',dict(**plan, sweep=records, selected=selected,
        completed_at=datetime.now(timezone.utc).isoformat()))
    print('SELECTED',selected,flush=True)


def training(root):
    work=root/'r4/implementation'
    assert not (work/'test-scoring-started.json').exists()
    frozen=read(TOOLS/'implementation_protocol.json')
    selected=read(work/'selection/selection.json')['selected']['schedule']
    assert frozen['model_params']==protocol(selected)['model_params']
    assert frozen['selection_sha256']==sha(work/'selection/selection.json')
    save(work/'protocol.json',frozen)
    prepared=work/'prepared'
    prepared.mkdir(exist_ok=False)
    audit=read(root/'prepared/audit.json')
    inner=read(root/'r4/prepared/audit.json')
    audit.update(bounds=inner['bounds'], channel_bounds=inner['channel_bounds'],
                 channel_abs_bounds=inner['channel_abs_bounds'], bounds_source=inner['bounds_source'])
    audit['n_privacy_units_per_site']=audit['n_per_site']
    save(prepared/'audit.json',audit)
    for name in ('train.csv','train.npz'):
        (prepared/name).symlink_to(root/'prepared'/name)
    cfg=read(root/'r4/config.json')
    data=np.load(prepared/'train.npz')
    x=client_app._apply_feature_bounds(data['X'],cfg)
    y,subjects=data['y'],data['subjects']
    assert len(y)==7352 and len(set(subjects))==21
    # Reuse the frozen R3 central scores/models by identity on the identical split.
    for seed in SEEDS:
        torch.manual_seed(seed)
        initial=arrays(server_app._build_initial_model(cfg))
        twin=work/'twins'/f'seed{seed}'
        twin.mkdir(parents=True,exist_ok=False)
        for clipped in (False,True):
            name='clipped_noiseless' if clipped else 'nonprivate_federated'
            result=federated(cfg,initial,(x,y,subjects),(x[::8],y[::8]),audit,selected,
                clipped=clipped,out=twin/(name+'.json'),seed=seed)
            result.update(seed=seed,model_sha256=sha(twin/(name+'.pt')),test_accessed=False,
                          monitoring='Every eighth TRAIN window only; final round always retained.',
                          initial_tensor_sha256=[__import__('hashlib').sha256(a.tobytes()).hexdigest() for a in initial])
            save(twin/(name+'.json'),result)
    runs=work/'runs'
    runs.mkdir(exist_ok=False)
    for epsilon in (1,4,8):
        for seed in SEEDS:
            print('CELL_START',epsilon,seed,flush=True)
            result=run_contract(root,runs/f'window-eps{epsilon}-seed{seed}',prepared,selected,seed,epsilon,False)
            print('CELL_TRAINED',epsilon,seed,result['status']['elapsed_s'],flush=True)
    print('ALL_TRAINED_UNSCORED',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['select','train'],required=True)
    args=p.parse_args()
    root=args.root
    os.environ.update(R_LIBS_USER=str((root/'Rlib').resolve()),DSFLOWER_VENV_ROOT=str((root/'venvs').resolve()),
        DSFLOWER_CLIENT_VENV_ROOT=str((root/'client').resolve()),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET='1',DSFLOWER_NODE_SECRET_FILE='/tmp/dsflower-sequence-r4-parent/secret')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    (selection if args.mode=='select' else training)(root)

if __name__=='__main__':
    main()
