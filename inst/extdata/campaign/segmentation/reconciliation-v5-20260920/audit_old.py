import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
B=Path('/workspace/segmentation');O=B/'reconciliation-v5-20260920'
sys.path[:0]=[str(B/'campaign-v5/tools/campaign/segmentation'),str(B/'runtime')]
from dsflower_runner import params,segmentation
from segmentation_metrics import metrics
J=lambda p:json.loads(p.read_text())
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(2);out=[]
for dataset in ('busbra','breast'):
 data=np.load(B/f'features/{dataset}/public-subject-tensors.npz'); lookup={str(s):i for i,s in enumerate(data['subjects'])}
 evidence=J(B/f'campaign-v5/inst/extdata/campaign/segmentation/batch16/{dataset}-evidence.json')
 for seed in (20260919,20260920,20260921):
  w=B/f'v3/runs-batch16/{dataset}-full-eps8-seed{seed}'
  split=J(w/'effective-split.json'); initial=J(w/'public-capture/public-initial.json'); ch=J(w/'channel-b.json')
  modelpath=next((w/'artifact').glob('*/model.pt')); model=params.load_user_model(initial['config'],segmentation.FEATURE_DIM,'segmentation_bce_dice');model.load_state_dict(torch.load(modelpath,map_location='cpu',weights_only=True));model.eval()
  ix=[lookup[s] for s in sorted(split['test'])];X=data['X'][ix];y=data['y'][ix,:1]
  with torch.no_grad(): p=np.concatenate([model(torch.from_numpy(X[i:i+16])).sigmoid().numpy() for i in range(0,len(X),16)])
  rep=next(r for r in evidence['replicates'] if r['seed']==seed and r['epsilon']==8)
  assert H(modelpath)==ch['artifact_sha256']==rep['artifact_sha256']
  assert metrics(p,y)==ch['metrics']==rep['federated_dp']
  row=dict(dataset=dataset,seed=seed,model_path=str(modelpath),model_sha256=H(modelpath),foreground_dice=metrics(p,y)['foreground_positive']['dice'],foreground_pixel_rate=float((p>=.5).mean()),subjects_with_foreground=int((p>=.5).reshape(len(p),-1).any(1).sum()),max_probability=float(p.max()),public_pretraining_present='public_pretraining' in initial)
  out.append(row);print(row,flush=True)
(O/'old-export-audit.json').write_text(json.dumps(out,indent=2)+'\n')
