#!/usr/bin/env python3
"""Unnoised public mechanistic controls; never selection or admission evidence.

These array-only SGD controls isolate sequential step budget, local averaging,
and unit clipping on the inner split. They do not simulate a DP federation and
cannot replace the actual DSLite development cells. No outer test is opened.
"""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from central_and_score import build,sha
from metrics import public_targets,score
from dsflower_runner import params,survival,client_app
import numpy as np
import pandas as pd
from scipy.special import expit
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace',type=Path);args=ap.parse_args()
    root=args.workspace; base=root/'runtime/hazard_v3'
    archive=root/'dsFlowerClient/inst/extdata/campaign/survival/hazard-v3'
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),seeds=[1101,1102],configurations=['g01','g02'],
        controls=['pooled_full_steps','pooled_site_step_count','three_site_unclipped','three_site_clipped_noiseless'],
        purpose='Public unnoised mechanism diagnostics only; no scores enter selection; no DP claim',
        script_sha256=sha(__file__))
    with (archive/'diagnostic_plan.json').open('x') as f:json.dump(plan,f,indent=2)
    results=[]
    torch.set_num_threads(1)
    for cid in plan['configurations']:
        for seed in plan['seeds']:
            cfg=json.loads((base/'runs'/f'dev-{cid}-seed{seed}'/'config.json').read_text())
            folder=base/'inner'/str(seed);meta=json.loads((folder/'split.json').read_text())
            assert meta['evaluation_role']=='inner validation'
            conf=survival.config_from_run(cfg,cfg['loss-name']);K=len(conf['edges'])-1
            cfg['feature-bounds']=meta['feature_bounds']
            def arrays(frame):
                x=client_app._apply_feature_bounds(frame[meta['features']].to_numpy(dtype=np.float32),cfg)
                t,e,v=public_targets(frame,conf)
                return x,survival.period_targets(t,e,v,conf)
            train=pd.read_csv(folder/'train.csv',float_precision='round_trip')
            validation=pd.read_csv(folder/'test.csv',float_precision='round_trip')
            x,y=arrays(train); xv,_=arrays(validation)
            sites=[arrays(pd.read_csv(folder/f'site{i}.csv',float_precision='round_trip')) for i in range(1,4)]
            initial=[a.copy() for a in params.get_torch_params(build(cfg))]
            assert [a.shape for a in initial]==[(K,x.shape[1]),(K,)]
            # Verify the direct linear per-patient gradient against frozen autograd.
            model=build(cfg);model.zero_grad()
            survival.loss_factory(cfg['loss-name'],cfg)(model(torch.from_numpy(x[:1])),torch.from_numpy(y[:1])).backward()
            residual=(expit(x[:1]@initial[0].T+initial[1])-y[:1,:K])*y[:1,K:2*K]/K
            np.testing.assert_allclose(model[0].weight.grad.numpy(),residual.T@x[:1],atol=2e-7,rtol=2e-6)
            nsteps=math.ceil(len(x)/cfg['batch-size']);site_steps=math.ceil(len(sites[0][0])/cfg['batch-size'])
            logits=x@initial[0].T+initial[1]
            residual=(expit(logits)-y[:,:K])*y[:,K:2*K]/K
            norm=np.sqrt((residual**2).sum(1)*(1+(x*x).sum(1)))
            row=dict(config=cid,seed=seed,K=K,edges=conf['edges'],train_sha256=sha(folder/'train.csv'),
                initial_gradient=dict(mean=float(norm.mean()),maximum=float(norm.max()),fraction_over_one=float((norm>1).mean())),
                exposure_per_interval=y[:,K:2*K].sum(0).tolist(),events_per_interval=y[:,:K].sum(0).tolist(),
                site_steps_per_epoch=site_steps,pooled_steps_per_epoch=nsteps,controls={})
            def local(weights,data,rng,steps,clip):
                xx,yy=data;w,b=[a.copy() for a in weights]
                geometry=math.ceil(len(xx)/cfg['batch-size']);q=1/geometry;divisor=max(1,len(xx)//geometry)
                clipped=sampled=0
                for _ in range(steps):
                    index=np.flatnonzero(rng.random(len(xx))<q)
                    xb,yb=xx[index],yy[index]
                    r=(expit(xb@w.T+b)-yb[:,:K])*yb[:,K:2*K]/K
                    norms=np.sqrt((r*r).sum(1)*(1+(xb*xb).sum(1)))
                    sampled+=len(index);clipped+=int((norms>1).sum())
                    if clip:r*=np.minimum(1,1/np.maximum(norms,1e-12))[:,None]
                    w-=cfg['learning-rate']*(r.T@xb)/divisor
                    b-=cfg['learning-rate']*r.sum(0)/divisor
                return [w,b],clipped,sampled
            for control in plan['controls']:
                weights=[a.copy() for a in initial]
                pooled=control.startswith('pooled');clipping=control.endswith('noiseless')
                rngs=[np.random.default_rng(seed+i) for i in range(3)]
                clipped=sampled=0
                for _ in range(cfg['num-server-rounds']):
                    if pooled:
                        steps=(nsteps if control=='pooled_full_steps' else site_steps)*cfg['local-epochs']
                        weights,c,n=local(weights,(x,y),rngs[0],steps,False)
                        clipped+=c;sampled+=n
                    else:
                        updates=[]
                        for data,rng in zip(sites,rngs):
                            fitted,c,n=local(weights,data,rng,site_steps*cfg['local-epochs'],clipping)
                            updates.append(fitted);clipped+=c;sampled+=n
                        weights=[np.stack([a[i] for a in updates]).mean(0) for i in range(2)]
                row['controls'][control]=dict(**score(xv@weights[0].T+weights[1],validation,conf),
                    sampled_patient_visits=sampled,gradient_fraction_over_one=clipped/max(1,sampled))
            results.append(row)
            print(cid,seed,{k:round(v['c_index'],6) for k,v in row['controls'].items()},flush=True)
    with (archive/'development_diagnostics.json').open('x') as f:json.dump(dict(plan=plan,results=results),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
