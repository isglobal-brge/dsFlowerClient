#!/usr/bin/env python3
"""Read-only model, configuration, split and score verification after confirmation."""
import argparse
import base64
import json
from pathlib import Path
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
    verified=[]
    torch.set_num_threads(1)
    for path in sorted(archive.glob('cell-support2-*.json')):
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
        same(metadata['feature_lower'],meta['feature_bounds']['lower'],'released lower bounds')
        same(metadata['feature_upper'],meta['feature_bounds']['upper'],'released upper bounds')
        split=root/'runtime/hazard_v3/confirmation'/f'{meta["subset"]}-{meta["seed"]}'
        for part in ('train','test'):
            require(sha(split/f'{part}.csv')==meta[f'{part}_sha256'],'split hash')
        test=pd.read_csv(split/'test.csv',float_precision='round_trip')
        cfg['feature-bounds']=meta['feature_bounds']
        x=client_app._apply_feature_bounds(test[meta['features']].to_numpy(dtype=np.float32),cfg)
        for branch,name in [('federated_dp','model.pt'),('central','central.pt'),('central_dp','central_dp.pt'),('null','null.pt')]:
            digest=record['artifact_checksum'] if branch=='federated_dp' else record['results'][branch+'_model_sha256']
            require(sha(files/name)==digest,'released model file hash')
            model=build(cfg);model.load_state_dict(torch.load(files/name,map_location='cpu',weights_only=True));model.eval()
            with torch.no_grad():output=model(torch.from_numpy(np.zeros_like(x) if branch=='null' else x)).numpy()
            same(score(output,test,conf),record['results'][branch],f'{identity}/{branch}')
        verified.append(identity)
    report=dict(status='verified',verified_confirmation_cells=len(verified),models_per_cell=4,identities=verified,
                method='Read-only reload of archived released weights, independent NumPy metrics, frozen split/model/configuration hashes')
    (archive/'artifact_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ARTIFACTS_VERIFIED',len(verified))


if __name__=='__main__':main()
