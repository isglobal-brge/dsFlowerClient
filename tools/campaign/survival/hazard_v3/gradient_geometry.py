#!/usr/bin/env python3
"""Initial patient gradient geometry on public inner training data; no fitting."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from central_and_score import build,mechanism,sha
from dsflower_runner import survival,client_app
from metrics import public_targets
import numpy as np
import pandas as pd
import torch

root=Path(sys.argv[1]);base=root/'runtime/hazard_v3'
evidence=root/'dsFlowerClient/inst/extdata/campaign/survival'
rows=[]
torch.set_num_threads(1)
for seed in (1101,1102):
    folder=base/'inner'/str(seed)
    meta=json.loads((folder/'split.json').read_text())
    assert meta['evaluation_role']=='inner validation'
    frame=pd.read_csv(folder/'train.csv',float_precision='round_trip')
    for variant in ('hazard','lognormal','weibull'):
        if variant=='hazard':
            cfg=json.loads((base/'runs'/f'dev-g01-seed{seed}'/'config.json').read_text())
        else:
            cfg=json.loads((evidence/f'cell-support2-full-{variant}-eps8-seed{seed}.json').read_text())['public_config']
        cfg['feature-bounds']=meta['feature_bounds']
        x=client_app._apply_feature_bounds(frame[meta['features']].to_numpy(dtype=np.float32),cfg)
        conf=survival.config_from_run(cfg,cfg['loss-name'])
        t,e,v=public_targets(frame,conf)
        y=survival.period_targets(t,e,v,conf) if variant=='hazard' else np.column_stack([t,e,v]).astype(np.float32)
        model=build(cfg)
        output=model(torch.from_numpy(x)).detach().requires_grad_(True)
        survival.loss_factory(cfg['loss-name'],cfg)(output,torch.from_numpy(y)).backward()
        residual=output.grad.numpy()*len(x)
        augmented=np.column_stack([x,np.ones(len(x),dtype=np.float32)])
        gradient=(residual[:,:,None]*augmented[:,None,:]).reshape(len(x),-1)
        norms=np.linalg.norm(gradient,axis=1)
        gradient*=np.minimum(1,1/np.maximum(norms,1e-12))[:,None]
        mech=mechanism(meta['sites'][0]['n_subjects'],cfg,8)
        noise_sd=mech['noise_multiplier']/mech['expected_batch_size']
        rows.append(dict(seed=seed,variant=variant,n_train=len(x),n_parameters=gradient.shape[1],
            train_sha256=sha(folder/'train.csv'),unclipped_mean_norm=float(norms.mean()),
            unclipped_maximum_norm=float(norms.max()),fraction_clipped_at_initialization=float((norms>1).mean()),
            norm_of_mean_clipped_gradient=float(np.linalg.norm(gradient.mean(0))),
            site_noise_sd_per_coordinate=noise_sd,
            site_noise_rms_l2=noise_sd*np.sqrt(gradient.shape[1]),mechanism=mech))
result=dict(scope='Initial geometry only; public inner training data; no fit and no validation or test score; not a final-model SNR decomposition',rows=rows)
with (evidence/'hazard-v3/gradient_geometry.json').open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False)
for r in rows:print(r['variant'],r['seed'],'clipped',round(r['fraction_clipped_at_initialization'],4),'mean gradient',round(r['norm_of_mean_clipped_gradient'],4),'noise RMS',round(r['site_noise_rms_l2'],4))
