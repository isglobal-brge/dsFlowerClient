#!/usr/bin/env python3
"""Training-only analytical emulator of the released two-logit DP head."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from dsflower_runner import dp_harness, seeding


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def controls():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def metrics(y, probability):
    p = np.asarray(probability, dtype=float)
    differences = p[y == 1, None] - p[None, y == 0]
    safe = np.clip(p, 1e-15, 1-1e-15)
    return dict(auc=float(((differences > 0).sum()+.5*(differences == 0).sum())/differences.size),
        accuracy=float(((p >= .5) == y).mean()), brier=float(np.mean((p-y)**2)),
        log_loss=float(-np.mean(y*np.log(safe)+(1-y)*np.log1p(-safe))))


def load_data(root):
    old = root/'r4/diagnosis'
    with np.load(old/'training-features.npz') as data:
        x,y,ids = data['x'],data['y'],data['ids']
        outer_xs = [data[f'x{i}'] for i in range(3)]
        outer_ys = [data[f'y{i}'] for i in range(3)]
    with np.load(old/'inner-indices.npz') as data:
        train,val = data['train'],data['validation']
    index = {str(v):i for i,v in enumerate(ids)}
    xs,ys = [],[]
    for site in range(1,4):
        with (old/f'collections/site{site}/samples.csv').open() as stream:
            patients = list(dict.fromkeys(row['subject_id'] for row in csv.DictReader(stream)))
        indices = [index[v] for v in patients]
        assert set(indices).issubset(set(train)) and len(indices)==227
        xs.append(x[indices]); ys.append(y[indices])
    # Inner extraction batches differ from outer-site batches. Use verified
    # inner tensors when available, keeping R4's validation tensor unchanged.
    inner_cache = root/'r5/inner-features.npz'
    if inner_cache.exists():
        with np.load(inner_cache) as data:
            xs = [data[f'x{i}'] for i in range(3)]
            ys = [data[f'y{i}'] for i in range(3)]
    prior = old/'federations/adam_lr0.003_e20_b32-startup-retry/public-capture'
    with np.load(prior/'public-initial-arrays.npz') as data:
        initial = [data[str(i)] for i in range(len(data.files))]
    cfg = json.loads((prior/'public-initial.json').read_text())['config']
    return dict(xs=xs,ys=ys,vx=x[val],vy=y[val],initial=initial,cfg=cfg,
        outer_xs=outer_xs,outer_ys=outer_ys,x=x,y=y)


def master_key(seed, round_index, site):
    return hashlib.sha256(f'vision-r5-diagnostic|{seed}|{round_index}|{site}'.encode()).digest()


def fit_round(initial, x, y, candidate, mechanism, master, device='cpu'):
    """Same Gaussian/Poisson streams as _dp_fit given the same master key."""
    weight,bias = [torch.nn.Parameter(torch.as_tensor(a,device=device).clone()) for a in initial]
    tx = torch.as_tensor(x,device=device)
    ty = torch.as_tensor(y,device=device).long()
    noise_rng = seeding.np_rng(seeding.sub_seed(master,'noise'))
    sample_rng = seeding.np_rng(seeding.sub_seed(master,'sample'))
    kw = dict(lr=candidate['learning_rate'],weight_decay=candidate['weight_decay'])
    if candidate['optimizer']=='sgd':
        optimizer = torch.optim.SGD([weight,bias], momentum=candidate.get('momentum',0), **kw)
    else:
        optimizer = torch.optim.Adam([weight,bias], **kw)
    with torch.no_grad():
        for _ in range(candidate['local_epochs']*mechanism['steps_per_epoch']):
            idx = np.flatnonzero(sample_rng.bernoulli_mask_one_in(mechanism['steps_per_epoch'],len(x)))
            xb,yb = tx[idx],ty[idx]
            logits = torch.nn.functional.linear(xb,weight,bias)
            residual = torch.softmax(logits.clamp(-30,30),1)
            residual[torch.arange(len(idx),device=device),yb] -= 1
            residual *= ((logits >= -30) & (logits <= 30))
            gw = (residual[:,:,None]*xb[:,None,:]).clamp(-1,1)
            gb = residual.clamp(-1,1)
            # Opacus computes parameter norms, then the norm across parameters.
            norms = torch.stack([gw.flatten(1).norm(2,dim=1),gb.norm(2,dim=1)],dim=1).norm(2,dim=1)
            factor = (1/(norms+1e-6)).clamp(max=1)
            sums = [torch.einsum('i,ijk->jk',factor,gw),torch.einsum('i,ij->j',factor,gb)]
            for parameter,total in zip([weight,bias],sums):
                noise = torch.as_tensor(noise_rng.normal(0,mechanism['noise_multiplier'],size=tuple(parameter.shape)),dtype=parameter.dtype,device=device)
                parameter.grad = (total+noise)/mechanism['expected_batch_size']
            optimizer.step()
            for parameter in [weight,bias]:
                parameter.nan_to_num_(nan=0,posinf=1e6,neginf=-1e6).clamp_(-1e6,1e6)
    return [p.detach().cpu().numpy().copy() for p in [weight,bias]]


def fit(initial, xs, ys, candidate, mechanism, seed, device='cpu'):
    arrays = [a.copy() for a in initial]
    for round_index in range(1,6):
        nodes = [fit_round(arrays,x,y,candidate,mechanism,master_key(seed,round_index,i),device)
                 for i,(x,y) in enumerate(zip(xs,ys))]
        arrays = [np.add.reduce([node[j] for node in nodes])/3 for j in range(2)]
    return arrays


def probability(arrays,x):
    with torch.no_grad():
        logits = torch.nn.functional.linear(torch.from_numpy(x),*[torch.from_numpy(a) for a in arrays])
        logits = torch.nan_to_num(logits,nan=0,posinf=30,neginf=-30).clamp(-30,30)
        return logits.softmax(1)[:,1].numpy()


def candidate(optimizer,lr,epochs,batch,momentum=0,decay=0):
    return dict(id=f'{optimizer}_m{momentum:g}_lr{lr:g}_e{epochs}_b{batch}_wd{decay:g}',
        optimizer=optimizer,learning_rate=lr,local_epochs=epochs,batch_size=batch,momentum=momentum,weight_decay=decay,
        batch_mode='full_site' if batch == 227 else 'fixed')


def main(root):
    controls(); out=root/'r5'; out.mkdir(exist_ok=True)
    assert json.loads((out/'validation.json').read_text())['passed'], 'Validate before ranking'
    assert json.loads((out/'finiteclamp-validation.json').read_text())['passed']
    assert json.loads((out/'inner-feature-parity.json').read_text())['passed']
    data=load_data(root)
    candidates=[]
    for optimizer,momentum,rates in [('sgd',0,[.001,.01,.03,.1,.3,1,3,10]),
                                     ('sgd',.9,[.001,.01,.03,.1,.3,1,3,10]),
                                     ('adam',0,[.0001,.0003,.001,.003,.01,.03,.1])]:
        for lr,epochs,batch,decay in itertools.product(rates,[1,2,4,8],[32,64,128,227],[0,.0001]):
            candidates.append(candidate(optimizer,lr,epochs,batch,momentum,decay))
    seeds=[20260923,20260924,20260925]
    save(out/'search-declaration.json',dict(candidates=candidates,noise_seeds=seeds,epsilon=8,delta=1e-6,
        rounds=5,site_sizes=[227]*3,validation_patients=171,initialization_seed=20260922,
        full_site_semantics='Batch 227 denotes full site: use 284 for each final site and 852 for a pooled full-site twin; fixed batches stay literal.',
        ranking='Mean private inner-validation AUC across three independent noise/sampling seeds; ties candidate id.',
        confirmation='Top three at epsilon 8, one actual federation each; if best AUC < .60 run its epsilon 4 and 1.',
        test_accessed=False))
    mechanisms={}
    for epochs,batch in itertools.product([1,2,4,8],[32,64,128,227]):
        mechanisms[epochs,batch]=dp_harness.effective_dpsgd_mechanism(8,1e-6,1,227,batch,epochs,5)
    save(out/'mechanisms.json',[dict(local_epochs=e,batch_size=b,**m) for (e,b),m in mechanisms.items()])
    results=[]; start=time.monotonic()
    for i,c in enumerate(candidates):
        m=mechanisms[c['local_epochs'],c['batch_size']]
        scores=[metrics(data['vy'],probability(fit(data['initial'],data['xs'],data['ys'],c,m,s),data['vx'])) for s in seeds]
        row=dict(candidate=c,mechanism=m,seeds=seeds,metrics=scores,
            mean_auc=float(np.mean([s['auc'] for s in scores])),sd_auc=float(np.std([s['auc'] for s in scores],ddof=1)))
        results.append(row)
        if (i+1)%16==0:
            save(out/'emulation-progress.json',results)
            print('SEARCH',i+1,len(candidates),'elapsed',round(time.monotonic()-start,1),'best',max(r['mean_auc'] for r in results),flush=True)
    ranked=sorted(results,key=lambda r:(-r['mean_auc'],r['candidate']['id']))
    save(out/'ranked-candidates.json',ranked)
    save(out/'shortlist.json',[r['candidate'] for r in ranked[:3]])
    print('SHORTLIST',json.dumps(ranked[:3]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
