#!/usr/bin/env python3
"""Generate the explicitly public CC0 synthetic three-node correctness fixture."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(1101)
x=rng.uniform(-1,1,90)
t=np.exp(5-x+rng.normal(0,.5,90))
frame=pd.DataFrame({'subject_id':['public-synthetic-'+str(i) for i in range(90)],
                   'x':x,'time':t,'event':rng.integers(0,2,90)})
train,test=frame[:72],frame[72:]
train.to_csv(p/'train.csv',index=False);test.to_csv(p/'test.csv',index=False)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
sites=[]
for site in range(3):
    a=train.iloc[np.arange(72)%3==site]
    a.to_csv(p/f'site{site+1}.csv',index=False)
    sites.append(dict(site=site+1,source_rows=len(a),n_subjects=len(a),split_sha256=sha(p/f'site{site+1}.csv')))
meta=dict(schema_version=1,dataset='synthetic-public',
          source=dict(release='prepare_synthetic.py, numpy default_rng1101',licence='CC0-1.0',sha256=sha(p/'train.csv')),
          public_fixture=True,seed=1101,subset='synthetic',n_original_rows=90,n_train=72,n_test=18,
          sites=sites,features=['x'],feature_bounds=dict(lower=[-1],upper=[1]),
          train_sha256=sha(p/'train.csv'),test_sha256=sha(p/'test.csv'),protocol_sha256=sha(Path(sys.argv[2])))
(p/'split.json').write_text(json.dumps(meta,indent=2)+'\n')
