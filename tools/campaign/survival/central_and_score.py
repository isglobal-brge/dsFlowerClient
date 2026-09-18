#!/usr/bin/env python3
"""Executed public-cohort exact twins and released-model scoring.

Uses the trusted model builder/DP loop; the independent metrics module scores
only analyst-local public held-out data. No node reporting endpoint is added.
"""
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

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'inst/flower_app'))
from dsflower_runner import client_app, dp_harness, params, seeding, server_app, survival
from metrics import public_targets, score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mechanism(n, cfg, epsilon):
    from opacus.accountants import PRVAccountant
    value = dp_harness.effective_dpsgd_mechanism(
        epsilon, 1e-5, 1., int(n), int(cfg['batch-size']),
        int(cfg['local-epochs']), int(cfg['num-server-rounds']))
    accountant = PRVAccountant()
    accountant.history = [(value['noise_multiplier'], value['sample_rate'], value['total_steps'])]
    delta0 = 1e-5/(1+math.exp(epsilon/2))
    eps0 = accountant.get_epsilon(delta0)
    value.update(accountant='PRV', add_remove_delta=delta0,
                 independently_recomputed_replace_one_epsilon=2*eps0,
                 independently_recomputed_replace_one_delta=delta0*(1+math.exp(eps0)))
    assert 2*eps0 <= epsilon + 1e-8, 'independent full-horizon epsilon gate'
    assert delta0*(1+math.exp(eps0)) <= 1e-5 + 1e-12, 'independent delta gate'
    return value


def build(cfg):
    initial = server_app._build_initial_model(cfg)
    model = params.load_user_model(cfg, int(cfg['num-features']), cfg['loss-name'])
    params.set_torch_params(model, params.get_torch_params(initial))
    return model


def train(cfg, x, y, epsilon, seed, private):
    model = build(cfg)
    rounds, epochs, batch = (int(cfg[k]) for k in ('num-server-rounds', 'local-epochs', 'batch-size'))
    pins = dict(loss_name=cfg['loss-name'], batch_size=batch, local_epochs=epochs,
                num_rounds=rounds, n_classes=2, learning_rate=cfg['learning-rate'],
                optimizer=dict(name='sgd', weight_decay=0., l1_penalty=0., momentum=0., nesterov=False),
                scheduler=dict(name='none'))
    if private:
        effective = mechanism(len(x), cfg, epsilon)
        # Unrecorded random secret, never a released public noise seed. Separate
        # domains per round and experiment, as in the node's trusted mechanism.
        secret = secrets.token_bytes(32)
        pcfg = dict(epsilon=epsilon,delta=1e-5,clipping_norm=1.,n_samples=len(x))
        for rnd in range(1, rounds+1):
            pins['round_index'] = rnd
            master = hmac.new(secret, f'public-central-diagnostic:{seed}:{rnd}'.encode(), hashlib.sha256).digest()
            arrays, _ = client_app._dp_fit(model, x, y, pcfg, pins, len(x), cfg,
                                           master, effective['noise_multiplier'])
            # New model/optimizer per federation round, matching node lifecycle.
            model = build(cfg)
            params.set_torch_params(model, arrays)
        return model, effective
    criterion = dp_harness.loss_from_allowlist(cfg['loss-name'], cfg)
    count = int(math.ceil(len(x)/batch))
    expected = max(1, int(len(x)/count))
    rng = np.random.default_rng(seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(rounds):
        optimizer = client_app._build_optimizer(model, pins)
        for _ in range(epochs):
            for _ in range(count):
                index = np.flatnonzero(rng.random(len(x)) < 1/count)
                optimizer.zero_grad()
                loss = criterion(model(xt[index]), yt[index]) * (len(index)/expected)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for p in model.parameters():
                        p.nan_to_num_(nan=0.,posinf=1e6,neginf=-1e6).clamp_(-1e6,1e6)
    return model, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epsilon', type=float, required=True)
    ap.add_argument('--federated-model', type=Path)
    ap.add_argument('--score-only', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    cfg = json.loads(args.config.read_text())
    meta = json.loads((args.split/'split.json').read_text())
    conf = survival.config_from_run(cfg, cfg['loss-name'])
    train_frame, test_frame = (pd.read_csv(args.split/f'{part}.csv') for part in ('train','test'))
    features = meta['features']
    cfg['feature-bounds'] = meta['feature_bounds']
    x, xt = (client_app._apply_feature_bounds(frame[features].to_numpy(dtype=np.float32), cfg)
             for frame in (train_frame,test_frame))
    times, events, valid = public_targets(train_frame, conf)
    x = client_app._totalize_private_features(x)
    x[~valid] = 0.
    y = (survival.period_targets(times,events,valid,conf) if 'edges' in conf
         else np.column_stack([times,events,valid]).astype(np.float32))
    result = {'seed':meta['seed'],'n_train':meta['n_train'],
              'minimum_site_n':min(s['n_subjects'] for s in meta['sites'])}
    def evaluate(model):
        model.eval()
        with torch.no_grad():
            output = model(torch.from_numpy(xt)).cpu().numpy()
        return score(output,test_frame,conf)
    if args.federated_model:
        model=build(cfg)
        model.load_state_dict(torch.load(args.federated_model, map_location='cpu',weights_only=True))
        result['federated_dp']=evaluate(model)
        result['model_sha256']=sha(args.federated_model)
    if not args.score_only:
        central,_=train(cfg,x,y,args.epsilon,meta['seed'],False)
        result['central']=evaluate(central)
        private, effective=train(cfg,x,y,args.epsilon,meta['seed'],True)
        result['central_dp']=evaluate(private)
        result['pooled_mechanism']=effective
        # The null is covariate free; a scalar intercept for AFT or one intercept
        # per public period. Retain held-out NLL as well as constant-risk C-index.
        null,_=train(cfg,np.zeros_like(x),y,args.epsilon,meta['seed'],False)
        with torch.no_grad():
            result['null']=score(null(torch.zeros_like(torch.from_numpy(xt))).numpy(),test_frame,conf)
        result['site_mechanisms']=[dict(site=s['site'],**mechanism(s['n_subjects'],cfg,args.epsilon)) for s in meta['sites']]
    import flwr, opacus, scipy
    result['versions']={'python':platform.python_version(),'torch':torch.__version__,
        'opacus':opacus.__version__,'flwr':flwr.__version__,'numpy':np.__version__,
        'scipy':scipy.__version__,'device':'cuda' if torch.cuda.is_available() else 'cpu',
        'platform':platform.platform(),'deterministic_algorithms':True}
    result['twin_matching']={
        'architecture_loss_preprocessing_initialization_optimizer_schedule':'exact',
        'initialization_seed':0,'pooled_epochs':cfg['num-server-rounds']*cfg['local-epochs'],
        'differences':'Pooled population changes q and steps; nonprivate twin has no clipping/noise; FedAvg averages sites equally; central DP secret is unrecorded and independent.',
    }
    result['elapsed_s']=time.monotonic()-started
    result['max_rss_native_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
