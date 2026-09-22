#!/usr/bin/env python3
"""Public-data twins using the frozen node loop and Flower aggregation classes."""
import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import platform
import resource
import secrets
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import central_and_score as original
from metrics import public_targets, score
import numpy as np
import pandas as pd
import torch
from dsflower_runner import client_app, params, seeding, survival
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg, FedAvgM, FedAdam, FedYogi

sha, build, mechanism = original.sha, original.build, original.mechanism


def optimizer_pins(cfg):
    optimizer = dict(name=cfg['optimizer-name'], weight_decay=0., l1_penalty=0.)
    if optimizer['name'] == 'sgd':
        optimizer.update(momentum=cfg['optimizer-momentum'], nesterov=cfg['optimizer-nesterov'])
    else:
        optimizer.update(beta1=cfg['optimizer-beta1'], beta2=cfg['optimizer-beta2'],
                         eps=cfg['optimizer-eps'], amsgrad=cfg['optimizer-amsgrad'])
    return dict(loss_name=cfg['loss-name'], batch_size=int(cfg['batch-size']),
                local_epochs=int(cfg['local-epochs']), num_rounds=int(cfg['num-server-rounds']),
                n_classes=2, learning_rate=cfg['learning-rate'], optimizer=optimizer,
                scheduler=dict(name='none'))


def aggregation(cfg):
    common = dict(fraction_train=1., fraction_evaluate=0., min_train_nodes=1,
                  min_evaluate_nodes=0, min_available_nodes=1, weighted_by_key='num-examples')
    name = cfg['strategy']
    if name == 'fedavgm':
        return FedAvgM(**common, server_learning_rate=1., server_momentum=.9)
    if name in ('fedadam', 'fedyogi'):
        eta, eta_l = (.1, .1) if name == 'fedadam' else (.01, .0316)
        return (FedAdam if name == 'fedadam' else FedYogi)(
            **common, eta=eta, eta_l=eta_l, beta_1=.9, beta_2=.99, tau=.001)
    return FedAvg(**common)


def aggregate(strategy, current, local_arrays, rnd):
    # This is pooled post-processing with one endpoint, not a substitute federation.
    record = ArrayRecord(numpy_ndarrays=params.get_torch_params(current))
    if isinstance(strategy, FedAvgM):
        if strategy.current_arrays is None:
            strategy.current_arrays = record
    elif isinstance(strategy, (FedAdam, FedYogi)):
        strategy.current_arrays = {k: a.numpy() for k, a in record.items()}
    reply = Message(RecordDict({'arrays': ArrayRecord(numpy_ndarrays=local_arrays),
                               'metrics': MetricRecord({'num-examples': 1})}),
                    dst_node_id=1, message_type='train')
    arrays, _ = strategy.aggregate_train(rnd, [reply])
    assert arrays is not None
    return arrays.to_numpy_ndarrays()


def train(cfg, x, y, epsilon, seed, private):
    model = build(cfg)
    pins = optimizer_pins(cfg)
    strategy = aggregation(cfg)
    rounds, epochs, batch = (int(cfg[k]) for k in ('num-server-rounds','local-epochs','batch-size'))
    effective = mechanism(len(x), cfg, epsilon) if private else None
    secret = secrets.token_bytes(32) if private else None
    rng = np.random.default_rng(seed)
    count = math.ceil(len(x)/batch)
    expected = max(1, int(len(x)/count))
    criterion = original.dp_harness.loss_from_allowlist(cfg['loss-name'], cfg)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for rnd in range(1, rounds+1):
        current = build(cfg)
        params.set_torch_params(current, params.get_torch_params(model))
        pins['round_index'] = rnd
        if private:
            master = hmac.new(secret, f'public-hazard-v3-pooled:{seed}:{rnd}'.encode(), hashlib.sha256).digest()
            arrays, _ = client_app._dp_fit(model, x, y,
                dict(epsilon=epsilon,delta=1e-5,clipping_norm=1.,n_samples=len(x)),
                pins,len(x),cfg,master,effective['noise_multiplier'])
            effective['execution_device'] = str(next(model.parameters()).device)
        else:
            optimizer = client_app._build_optimizer(model, pins)
            for _ in range(epochs):
                for _ in range(count):
                    index = np.flatnonzero(rng.random(len(x)) < 1/count)
                    optimizer.zero_grad()
                    loss = criterion(model(xt[index]),yt[index])*(len(index)/expected)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        for p in model.parameters():
                            p.nan_to_num_(nan=0.,posinf=1e6,neginf=-1e6).clamp_(-1e6,1e6)
            arrays = params.get_torch_params(model)
        arrays = aggregate(strategy, current, arrays, rnd)
        model = build(cfg)
        params.set_torch_params(model, arrays)
    return model, effective


def main():
    ap = argparse.ArgumentParser()
    for arg in ('config','split','out','federated-model'):
        ap.add_argument('--'+arg,type=Path,required=True)
    ap.add_argument('--epsilon',type=float,required=True)
    ap.add_argument('--development',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(1)
    seeding.seed_torch(bytes(32))
    started=time.monotonic()
    cfg=json.loads(args.config.read_text())
    meta=json.loads((args.split/'split.json').read_text())
    assert not args.development or meta['evaluation_role']=='inner validation'
    conf=survival.config_from_run(cfg,cfg['loss-name'])
    cfg['feature-bounds']=meta['feature_bounds']
    for part in ('train','test'):
        assert sha(args.split/f'{part}.csv')==meta[f'{part}_sha256']
    train_frame,test_frame=(pd.read_csv(args.split/f'{part}.csv',float_precision='round_trip') for part in ('train','test'))
    features=meta['features']
    x,xt=(client_app._apply_feature_bounds(f[features].to_numpy(dtype=np.float32),cfg) for f in (train_frame,test_frame))
    times,events,valid=public_targets(train_frame,conf)
    x=client_app._totalize_private_features(x)
    x[~valid]=0.
    y=survival.period_targets(times,events,valid,conf)
    result=dict(seed=meta['seed'],n_train=meta['n_train'],minimum_site_n=min(s['n_subjects'] for s in meta['sites']))
    def evaluate(model, values=xt):
        model.cpu().eval()
        with torch.no_grad():
            return score(model(torch.from_numpy(values)).numpy(),test_frame,conf)
    model=build(cfg)
    model.load_state_dict(torch.load(args.federated_model,map_location='cpu',weights_only=True))
    result['federated_dp']=evaluate(model)
    result['model_sha256']=sha(args.federated_model)
    result['site_mechanisms']=[dict(site=s['site'],**mechanism(s['n_subjects'],cfg,args.epsilon)) for s in meta['sites']]
    if not args.development:
        for label, private in (('central',False),('central_dp',True),('null',False)):
            values=np.zeros_like(x) if label=='null' else x
            fitted,effective=train(cfg,values,y,args.epsilon,meta['seed'],private)
            result[label]=evaluate(fitted,np.zeros_like(xt) if label=='null' else xt)
            path=args.out.parent/f'{label}.pt'
            torch.save(fitted.state_dict(),path)
            result[label+'_model_sha256']=sha(path)
            if private:
                result['pooled_mechanism']=effective
    import flwr,opacus,scipy
    result['versions']=dict(python=platform.python_version(),torch=torch.__version__,opacus=opacus.__version__,
        flwr=flwr.__version__,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
        cuda_available=torch.cuda.is_available(),platform=platform.platform(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        nonprivate_twin_device='cpu',dp_twin_device='cpu',heldout_evaluation_device='cpu',
        federation_device_rule='Frozen _dp_fit; verified CPU-only pod')
    result['twin_matching']=dict(architecture_loss_preprocessing_initialization_optimizer_schedule='exact',
        initialization_seed=0,pooled_epochs=cfg['num-server-rounds']*cfg['local-epochs'],
        aggregation=cfg['strategy'],aggregation_implementation='Flower 1.31.0 same strategy, one pooled endpoint',
        differences='Pooled population changes q and sequential step count; nonprivate removes clipping/noise; fixed unit site weights; independent unrecorded DP secrets; development omits twins.')
    result['elapsed_s']=time.monotonic()-started
    result['max_rss_native_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['memory_measurement']=dict(scope='central/scoring process peak resident memory',native_unit='KiB',federation_peak_memory_measured=False)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
