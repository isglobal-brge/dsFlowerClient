#!/usr/bin/env python3
"""Verify frozen R4 artifacts, open TEST once, score each new model once."""
import argparse
import base64
import importlib.metadata
import os
import subprocess
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from implementation import read, save, SEEDS, TOOLS
from implementation_verify import verify_run, sha
from dsflower_runner import client_app
from prepare_public_data import load_split
from score_and_assemble import metrics, mean_sd


def main(root, verify_only):
    work=root/'r4/implementation'
    frozen=read(work/'protocol.json')
    assert sha(work/'protocol.json')==sha(TOOLS/'implementation_protocol.json')
    assert frozen['selection_sha256']==sha(work/'selection/selection.json')
    audit=read(work/'prepared/audit.json')
    assert sha(work/'prepared/train.npz')==audit['train_npz_sha256']
    data=np.load(work/'prepared/train.npz')
    old_path=TOOLS.parents[3]/'inst/extdata/campaign/sequence/har_window_pytorch_lstm_eps8.json'
    old=read(old_path)
    for k in ('archive_sha256','train_npz_sha256','site_subjects','train_subjects','n_per_site','features','bounds'):
        assert audit[k]==old['dataset'][k], k
    runs=[verify_run(work/'runs'/f'window-eps{epsilon}-seed{seed}',frozen,audit,data,'window',epsilon,seed)
          for epsilon in (1,4,8) for seed in SEEDS]
    central={rep['seed']:rep for rep in old['per_replicate']}
    twin_status={}
    for seed in SEEDS:
        initial={tuple(r['initial']['tensor_sha256']) for r in runs if r['seed']==seed}
        assert initial=={tuple(central[seed]['central_training']['initial_tensor_sha256'])}
        c=central[seed]['central_training']
        assert (c['epochs'],c['batch_size'],c['learning_rate'],c['steps'],c['optimizer_reset_every_epochs'])==(20,256,.01,580,4)
        for arm in ('nonprivate_federated','clipped_noiseless'):
            path=work/'twins'/f'seed{seed}'/(arm+'.pt')
            status=read(path.with_suffix('.json'))
            assert sha(path)==status['model_sha256'] and status['final']['round']==5
            assert status['schedule']==read(work/'selection/selection.json')['selected']['schedule']
            assert initial=={tuple(status['initial_tensor_sha256'])}
            twin_status[(seed,arm)]=status
    # TRAIN-only predictor smoke test before the scoring gate.
    sys.path.insert(0,str(root/'Rlib/dsFlowerClient/python'))
    from predict_helper import predict_pytorch_spec
    torch.set_num_threads(2)
    cfg=runs[0]['initial']['config']
    for run in runs:
        assert json.loads(base64.b64decode(run['initial']['config']['model-spec-b64'])) == json.loads(base64.b64decode(central[run['seed']]['effective_config']['model-spec-b64']))
    vx=client_app._apply_feature_bounds(data['X'][:16],cfg)
    for arm in ('nonprivate_federated','clipped_noiseless'):
        p=np.asarray(predict_pytorch_spec(str(work/'twins'/f'seed{SEEDS[0]}'/(arm+'.pt')),vx,'prob',
                           cfg['model-spec-b64'],'cross_entropy',num_classes=6))
        assert p.shape==(16,6) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
    verification=dict(status='verified_unscored', federations=9, node_round_captures=135,
        new_noiseless_twins=6,central='Reuse frozen R3 central model identities and held-out metrics, identical split and seeds.',
        central_source_sha256=sha(old_path),test_accessed=False,protocol_sha256=sha(work/'protocol.json'))
    save(work/'pre_scoring_verification.json',verification)
    print(json.dumps(verification),flush=True)
    if verify_only:
        return
    runtime=read(root/'r4/provisioning-runtime.json')
    rcode = """library(dsFlower); library(dsFlowerClient); cat(jsonlite::toJSON(list(
      dsFlower=as.character(packageVersion("dsFlower")), dsFlowerClient=as.character(packageVersion("dsFlowerClient")),
      server_runner_sha256=dsFlower:::.compute_harness_hash(),
      client_runner_sha256=dsFlowerClient:::.compute_local_runner_hash()),auto_unbox=TRUE))"""
    observed=json.loads(subprocess.check_output(['Rscript','-e',rcode],text=True,
        env=dict(os.environ,R_LIBS_USER=str((root/'Rlib').resolve()))))
    for key,value in observed.items():
        assert runtime['packages'][key]==value
    assert sha(root/'Rlib/dsFlower/python/sitecustomize.py')==runtime['packages']['server_guard_sha256']
    for pin in (TOOLS/'requirements-r3.txt').read_text().splitlines():
        if pin and not pin.startswith('#'):
            name,version=pin.split('==')
            assert importlib.metadata.version(name)==version
    runtime['provisioning_recorded_at']=runtime['recorded_at']
    runtime['recorded_at']=datetime.now(timezone.utc).isoformat()
    runtime['installed_identity_reverified']=observed
    runtime['declaration_commit']=(TOOLS/'declaration_commit.txt').read_text().strip()
    runtime.update(pod_id='fad5nedghbctap',campaign_revision='R4',
        initialization_seeds=SEEDS,tooling_sha256={p.name:sha(p) for p in TOOLS.glob('*') if p.is_file()},
        privacy_randomness='Unchanged node-owned cryptographic sampling and Gaussian-noise streams.')
    save(work/'runtime.json',runtime)
    with (work/'test-scoring-started.json').open('x') as stream:
        json.dump(dict(started_at=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(work/'protocol.json'),
            federated_models=[r['status']['model_sha256'] for r in runs],
            twin_models=[s['model_sha256'] for s in twin_status.values()],
            central_models_reused=[central[s]['central_training']['model_sha256'] for s in SEEDS]),stream,indent=2)
    started=time.monotonic()
    archive=root/'data_cache/uci-har-240.zip'
    assert sha(archive)==audit['archive_sha256']
    x,y,subjects=load_split(archive,'test')
    assert len(y)==2947 and len(set(subjects))==9 and set(subjects).isdisjoint(audit['train_subjects'])
    assert sorted(map(int,set(subjects)))==old['dataset']['test_subjects']
    values=client_app._apply_feature_bounds(x.reshape(len(y),-1),cfg)
    prior=np.bincount(data['y'],minlength=6).astype(float)
    prior/=prior.sum()
    trivial=metrics(y,np.tile(prior,(len(y),1)))
    assert trivial==central[SEEDS[0]]['trivial']
    out=work/'evidence'
    out.mkdir(exist_ok=False)

    def score(path, config):
        start=time.monotonic()
        p=np.asarray(predict_pytorch_spec(str(path),values,'prob',config['model-spec-b64'],
                                       'cross_entropy',num_classes=6))
        result=metrics(y,p)
        return dict(metrics=result,diagnostics=dict(annotation_only=True,
            predicted_class_counts=np.bincount(p.argmax(1),minlength=6).tolist(),
            mean_class_probability=p.mean(0).tolist(),
            max_probability_span_across_windows=float(np.ptp(p,axis=0).max()),
            macro_auc_above_chance=result['macro_auc']>.5,
            accuracy_above_majority=result['accuracy']>trivial['accuracy']),
            elapsed_s=time.monotonic()-start)

    twins={}
    for (seed,arm),status in twin_status.items():
        result=score(work/'twins'/f'seed{seed}'/(arm+'.pt'),cfg)
        twins[(seed,arm)]=result
        save(out/f'{arm}-seed{seed}-score.json',result)
    grouped={e:[] for e in (1,4,8)}
    for run in runs:
        seed,epsilon=run['seed'],run['epsilon']
        result=score(run['checkpoint'],run['initial']['config'])
        rep=dict(seed=seed,federated_dp=result['metrics'],central=central[seed]['central'],trivial=trivial,
            nonprivate_federated=twins[(seed,'nonprivate_federated')]['metrics'],
            clipped_noiseless=twins[(seed,'clipped_noiseless')]['metrics'],
            gap_macro_auc=result['metrics']['macro_auc']-central[seed]['central']['macro_auc'],
            federated_model_sha256=run['status']['model_sha256'],central_training=central[seed]['central_training'],
            central_reuse=dict(source_file=old_path.name,source_sha256=sha(old_path),same_split_verified=True,
                method='Frozen R3 model identity and metrics reused without retraining or rescoring.'),
            twin_training={arm:twin_status[(seed,arm)] for arm in ('nonprivate_federated','clipped_noiseless')},
            initial_tensor_sha256=run['initial']['tensor_sha256'],effective_config=run['initial']['config'],
            node_contract=read(run['run']/'node-contract.json'),node_round_captures=run['captures'],
            independent_accounting=run['accounting'],
            elapsed_s=dict(federated_training=run['status']['elapsed_s'],scoring=result['elapsed_s'],
                nonprivate_federated=twin_status[(seed,'nonprivate_federated')]['elapsed_s'],
                clipped_noiseless=twin_status[(seed,'clipped_noiseless')]['elapsed_s']),
            utility_diagnostics=dict(federated_dp=result['diagnostics'],
                **{a:twins[(seed,a)]['diagnostics'] for a in ('nonprivate_federated','clipped_noiseless')}))
        grouped[epsilon].append(rep)
        save(out/f'federated-eps{epsilon}-seed{seed}-score.json',result)
        print('SCORED',epsilon,seed,result['metrics'],flush=True)
    cells=[]
    interpretation=('Window-level mechanism measurement on subject-disjoint sites; no subject-level protection. '
        'Schedule selected on TRAIN subjects by real epsilon-8 inner-validation mean macro-AUC. '
        'The central gap includes finite-schedule, federation, clipping and noise effects; '
        'the matched clipped-noiseless contrast estimates marginal noise cost with independent sampling streams.')
    for epsilon,reps in grouped.items():
        summary={arm:{k:mean_sd([r[arm][k] for r in reps]) for k in ('macro_auc','accuracy','log_loss')}
                 for arm in ('central','nonprivate_federated','clipped_noiseless','federated_dp','trivial')}
        summary['gap_macro_auc']=mean_sd([r['gap_macro_auc'] for r in reps])
        summary['noise_contrast_macro_auc']=mean_sd([r['federated_dp']['macro_auc']-r['clipped_noiseless']['macro_auc'] for r in reps])
        cell=dict(schema='dsflower-sequence-campaign-r3-v1',revision='R4',status='executed',contract='pytorch_lstm',
            unit='window',epsilon=epsilon,protocol=frozen,protocol_sha256=sha(work/'protocol.json'),runtime=runtime,
            interpretation=interpretation,dataset=dict(audit,test_accessed=True,preparation_test_accessed=False,
                n_test_windows=len(y),n_test_subjects=9,test_subjects=sorted(map(int,set(subjects)))),
            per_replicate=reps,summary=summary,selection=read(work/'selection/selection.json'),
            diagnostics=dict(annotation_only=True,central_learned=True,
                federated_auc_above_chance=summary['federated_dp']['macro_auc']['mean']>.5,
                federated_accuracy_above_majority=summary['federated_dp']['accuracy']['mean']>trivial['accuracy']),
            limitations=[interpretation,'Fixed split; sample SD describes three training replicates only.',
                'No end-to-end private selection or campaign-wide composition guarantee.',
                'No pooled-DP arm; three-site DP and both noiseless controls prioritized.',
                'Central metrics reused from R3; twins shared across epsilons.'],
            scoring_marker_sha256=sha(work/'test-scoring-started.json'),
            wall_clock=dict(scoring_and_assembly_elapsed_s=time.monotonic()-started),
            scored_at=datetime.now(timezone.utc).isoformat())
        filename=f'har_r4_window_pytorch_lstm_eps{epsilon}.json'
        with (out/filename).open('x') as stream:
            json.dump(cell,stream,indent=2,allow_nan=False)
            stream.write('\n')
        cells.append(dict(file=filename,sha256=sha(out/filename),revision='R4',status='executed',
            contract='pytorch_lstm',dataset=audit['name'],unit='window',epsilon=epsilon,
            summary=summary,interpretation=interpretation,diagnostics=cell['diagnostics'],
            diagnostics_annotation_only=True,limitations=cell['limitations']))
    save(out/'summary.json',dict(cells=cells,scored_at=datetime.now(timezone.utc).isoformat(),
        scoring_elapsed_s=time.monotonic()-started))
    print('SCORING_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    main(a.root,a.verify_only)
