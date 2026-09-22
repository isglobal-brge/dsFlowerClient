"""Frozen candidate order and deterministic training-only split construction."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_data import SOURCES, encode, sha

SEEDS=(1101,1102,1103)
DEV_SEEDS=SEEDS[:2]


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write('\n')


def grid():
    configs=[]
    def add(k=5,lr=.05,optimizer='sgd',strategy='fedavg',rounds=10,epochs=4,batch=64,kind='quantile'):
        configs.append(dict(id=f'g{len(configs)+1:02}',protocol_version=3,K=k,learning_rate=lr,
            optimizer=optimizer,strategy=strategy,rounds=rounds,local_epochs=epochs,
            batch_size=batch,grid=kind,preprocessing='declared public bounds to [-1,1]'))
    add(k=10,kind='equal_width')
    for k in (5,8,12,10): add(k=k)
    for k in (5,8): add(k=k,lr=.1)
    for k in (5,8): add(k=k,optimizer='adam',lr=.02,epochs=1)
    for strategy in ('fedavgm','fedadam','fedyogi'):
        for k in (5,8): add(k=k,strategy=strategy,rounds=20,epochs=2)
    for k in (5,8): add(k=k,batch=32)
    for k in (5,8): add(k=k,lr=.1,rounds=20,epochs=1)
    for k in (5,8): add(k=k,rounds=20,epochs=4)
    for lr in (.05,.1):
        for k in (5,8): add(k=k,optimizer='adam',lr=lr,epochs=1)
    for k in (5,8): add(k=k,lr=.02,epochs=2)
    for strategy in ('fedavgm','fedadam','fedyogi'):
        for k in (5,8): add(k=k,strategy=strategy,optimizer='adam',lr=.02,epochs=2)
    for k in (10,12): add(k=k,optimizer='adam',lr=.02,epochs=2)
    assert len(configs)==35
    return configs


def save_split(dest,train,test,meta,parts,protocol):
    dest.mkdir(parents=True)
    train.to_csv(dest/'train.csv',index=False)
    test.to_csv(dest/'test.csv',index=False)
    sites=[]
    for i,part in enumerate(parts,1):
        part.to_csv(dest/f'site{i}.csv',index=False)
        sites.append(dict(site=i,source_rows=len(part),n_subjects=len(part),split_sha256=sha(dest/f'site{i}.csv')))
    assert set(train.subject_id).isdisjoint(test.subject_id)
    assert sum(map(len,parts))==len(train)
    meta=dict(meta,protocol_version=3,protocol_sha256=sha(protocol),n_train=len(train),n_test=len(test),
        train_sha256=sha(dest/'train.csv'),test_sha256=sha(dest/'test.csv'),sites=sites)
    write(dest/'split.json',meta)
    return dest


def prepare_outer(raw_path,dest,protocol):
    # Data ingestion freezes canonical outer splits before modeling. Development
    # below can operate with outer test.csv absent or poisoned.
    assert sha(raw_path)==SOURCES['support2']['sha256']
    frame,bounds=encode(pd.read_csv(raw_path),'support2')
    for seed in SEEDS:
        keys=frame.subject_id.map(lambda subject: hashlib.sha256(f'{seed}:{subject}'.encode()).hexdigest())
        ordered=frame.loc[keys.sort_values().index].reset_index(drop=True)
        train,test=ordered.iloc[:7284].copy(),ordered.iloc[7284:].copy()
        meta=dict(schema_version=1,dataset='support2',source=SOURCES['support2'],seed=seed,subset='full',
            public_fixture=True,n_original_rows=len(frame),time_origin='study_entry',
            subject_provenance='One public release row per stable subject ID',
            split_rule='SHA256(seed:id), floor(0.8*N) train, no outcome stratification',
            features=list(bounds),feature_bounds=dict(lower=[b[0] for b in bounds.values()],upper=[b[1] for b in bounds.values()]),
            evaluation_role='sealed outer holdout')
        save_split(dest/f'support2-full-{seed}',train,test,meta,[train.iloc[i::3] for i in range(3)],protocol)


def prepare_inner(source,dest,protocol):
    meta=json.loads((source/'split.json').read_text())
    parts,valid=[],[]
    for i in range(1,4):
        path=source/f'site{i}.csv'
        assert sha(path)==meta['sites'][i-1]['split_sha256']
        frame=pd.read_csv(path,dtype={'subject_id':str},float_precision='round_trip')
        keys=frame.subject_id.map(lambda subject: hashlib.sha256(f'hazard-v3-inner:{meta["seed"]}:{subject}'.encode()).hexdigest())
        frame=frame.loc[keys.sort_values().index]
        cut=int(.8*len(frame))
        parts.append(frame.iloc[:cut]); valid.append(frame.iloc[cut:])
    train,test=pd.concat(parts,ignore_index=True),pd.concat(valid,ignore_index=True)
    meta.update(evaluation_role='inner validation',outer_manifest_sha256=sha(source/'split.json'),
        outer_train_sha256=meta['train_sha256'],split_rule='Within each outer site SHA256(hazard-v3-inner:seed:id), floor(0.8*N) train')
    return save_split(dest,train,test,meta,parts,protocol)


def config_for_seed(cfg,inner):
    result=dict(cfg)
    if cfg['grid']=='equal_width':
        edges=np.linspace(0,1825,cfg['K']+1).tolist()
    else:
        path=inner/'train.csv'
        frame=pd.read_csv(path,float_precision='round_trip')
        eligible=frame.loc[(frame.event==1)&(frame.time>=1)&(frame.time<=1825),'time']
        edges=[0.]+np.quantile(eligible,np.arange(1,cfg['K'])/cfg['K'],method='linear').tolist()+[1825.]
    assert len(edges)==cfg['K']+1 and all(a<b for a,b in zip(edges,edges[1:]))
    result.update(edges=edges,grid_source='inner-training events only; fixed before development fits; reused unchanged at confirmation',
                  grid_source_sha256=sha(inner/'train.csv'))
    return result


def prepare_confirmation(source,dest,protocol,subset):
    meta=json.loads((source/'split.json').read_text())
    for part in ('train','test'):
        assert sha(source/f'{part}.csv')==meta[f'{part}_sha256']
    train,test=(pd.read_csv(source/f'{part}.csv',dtype={'subject_id':str},float_precision='round_trip') for part in ('train','test'))
    if subset=='small600': train=train.iloc[:600].copy()
    if subset=='heterogeneous':
        train=train.sort_values(['age','subject_id']).reset_index(drop=True)
        parts=[train.iloc[i*2428:(i+1)*2428] for i in range(3)]
    else:
        nsites=2 if subset=='two-sites' else 3
        parts=[train.iloc[i::nsites] for i in range(nsites)]
    meta.update(subset=subset,evaluation_role='third protocol confirmation on reused outer holdout',
        outer_manifest_sha256=sha(source/'split.json'))
    return save_split(dest,train,test,meta,parts,protocol)


def select(rows):
    assert rows
    for row in rows:
        assert set(row['scores'])==set(map(str,DEV_SEEDS))
        assert all(np.isfinite(v) and 0<=v<=1 for v in row['scores'].values())
        row['mean']=sum(row['scores'].values())/2
    return sorted(rows,key=lambda r:(-r['mean'],r['config']['rounds']*r['config']['local_epochs'],r['config']['K'],r['config']['id']))
