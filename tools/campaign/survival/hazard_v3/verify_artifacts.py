#!/usr/bin/env python3
"""Read-only model, configuration, split and score verification after confirmation."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from central_and_score import build,sha
from dsflower_runner import client_app,survival
from metrics import score
from validate_completion import require,same
import numpy as np
import pandas as pd
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace',type=Path);args=ap.parse_args()
    root=args.workspace;archive=root/'dsFlowerClient/inst/extdata/campaign/survival/hazard-v3'
    selected=json.loads((archive/'selection.json').read_text())
    verified=[];verified_development=[]
    development=[]
    for name,digest in selected['development_sha256'].items():
        path=archive/'development'/name
        require(sha(path)==digest,'locked development record')
        record=json.loads(path.read_text())
        require(record['status']=='executed','development execution')
        identity=path.stem.removeprefix('cell-')
        run=root/'runtime/hazard_v3/runs'/identity
        models=list((run/'federation/artifact').glob('*/model.pt'))
        require(len(models)==1 and sha(models[0])==record['artifact_checksum'],'development model hash')
        dest=archive/'artifacts'/identity
        dest.mkdir(exist_ok=True)
        for item in ('model.pt','metadata.json','history.json'):
            shutil.copyfile(models[0].parent/item,dest/item)
        for item in ('config.json','scores.json'):
            shutil.copyfile(run/item,dest/item)
        development.append(path)
    torch.set_num_threads(1)
    for path in sorted(archive.glob('cell-support2-*.json'))+sorted(development):
        record=json.loads(path.read_text())
        if record['status']!='executed':continue
        identity=path.stem.removeprefix('cell-');meta=record['dataset'];cfg=dict(record['public_config'])
        files=archive/'artifacts'/identity
        metadata=json.loads((files/'metadata.json').read_text())
        require(metadata['strategy'].lower()==record['hazard_v3_config']['strategy'],'released strategy')
        require(metadata['num_rounds']==cfg['num-server-rounds'] and metadata['n_clients']==record['site_count'],'released topology/horizon')
        conf=survival.config_from_run(cfg,cfg['loss-name'])
        same(metadata['survival_config'],conf,'released grid')
        require(metadata['features']==meta['features'],'released features')
        require(metadata['privacy']=='server-enforced-dp','released privacy enforcement')
        model_params=metadata['model_params']
        for key,expected in [('learning_rate',cfg['learning-rate']),('batch_size',cfg['batch-size']),
            ('local_epochs',cfg['local-epochs']),('optimizer',cfg['optimizer-name']),
            ('weight_decay',0),('scheduler','none')]:
            same(model_params[key],expected,'released model parameters/'+key)
        same(metadata['feature_lower'],meta['feature_bounds']['lower'],'released lower bounds')
        same(metadata['feature_upper'],meta['feature_bounds']['upper'],'released upper bounds')
        is_development=record['hazard_v3_phase']=='development'
        split=root/'runtime/hazard_v3'/('inner' if is_development else 'confirmation')/(
            str(meta['seed']) if is_development else f'{meta["subset"]}-{meta["seed"]}')
        for part in ('train','test'):
            require(sha(split/f'{part}.csv')==meta[f'{part}_sha256'],'split hash')
        training=pd.read_csv(split/'train.csv',usecols=['subject_id'],dtype=str)['subject_id']
        training_ids=set(training)
        require(len(training)==len(training_ids)==meta['n_train'],'unique training patients')
        covered=set()
        for site in meta['sites']:
            site_path=split/f'site{site["site"]}.csv'
            require(sha(site_path)==site['split_sha256'],'site file hash')
            patients=pd.read_csv(site_path,usecols=['subject_id'],dtype=str)['subject_id']
            patient_ids=set(patients)
            require(len(patients)==len(patient_ids)==site['n_subjects'],'unique site patients')
            require(covered.isdisjoint(patient_ids),'patients overlap across sites')
            covered.update(patient_ids)
        require(covered==training_ids,'site partition differs from pooled training')
        test=pd.read_csv(split/'test.csv',float_precision='round_trip')
        require(training_ids.isdisjoint(set(test['subject_id'].astype(str))),'within-seed training/evaluation overlap')
        cfg['feature-bounds']=meta['feature_bounds']
        x=client_app._apply_feature_bounds(test[meta['features']].to_numpy(dtype=np.float32),cfg)
        branches=[('federated_dp','model.pt')]
        if not is_development:
            branches+=[('central','central.pt'),('central_dp','central_dp.pt'),('null','null.pt')]
        for branch,name in branches:
            digest=record['artifact_checksum'] if branch=='federated_dp' else record['results'][branch+'_model_sha256']
            require(sha(files/name)==digest,'released model file hash')
            model=build(cfg);model.load_state_dict(torch.load(files/name,map_location='cpu',weights_only=True));model.eval()
            with torch.no_grad():output=model(torch.from_numpy(np.zeros_like(x) if branch=='null' else x)).numpy()
            same(score(output,test,conf),record['results'][branch],f'{identity}/{branch}')
        (verified_development if is_development else verified).append(identity)
    report=dict(status='verified',verified_confirmation_cells=len(verified),confirmation_models_per_cell=4,
                development_models_per_cell=1,identities=verified,
                verified_development_cells=len(verified_development),development_identities=verified_development,
                patient_partition_checks='Unique subjects, disjoint sites, exact union with pooled training, and disjoint within-seed evaluation subjects; every site file hash verified',
                method='Read-only reload of archived released weights, independent NumPy metrics, frozen split/model/configuration hashes')
    (archive/'artifact_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ARTIFACTS_VERIFIED',len(verified),'confirmation;',len(verified_development),'development')


if __name__=='__main__':main()
