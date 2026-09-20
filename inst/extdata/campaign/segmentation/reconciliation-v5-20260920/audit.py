import os, sys, json, hashlib, shutil
from pathlib import Path
import numpy as np
import torch
B=Path('/workspace/segmentation'); O=B/'reconciliation-v5-20260920'; O.mkdir(exist_ok=True)
T=B/'campaign-v5/tools/campaign/segmentation'
sys.path[:0]=[str(T),str(B/'runtime')]
os.environ['F_SEG_V4_CONFIG']=str(B/'v5/active-candidate.json')
os.environ['F_SEG_V5_BINDINGS']=str(B/'v5/public-pretraining/epochs60')
from benchmark_hooks.public_initialization import verify_capture
from assemble_evidence import load_replicate
from segmentation_metrics import metrics
from dsflower_runner import params,segmentation
J=lambda p:json.loads(p.read_text())
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
A=lambda arr:hashlib.sha256(b''.join(a.tobytes() for a in arr)).hexdigest()
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
result={'evidence_identity':[], 'cells':[], 'development':[]}
protected={}
for batch in (16,64):
 for dataset in ('busbra','breast'):
  p=B/f'campaign-v5/inst/extdata/campaign/segmentation/batch{batch}/{dataset}-evidence.json'
  q=B/f'v5/evidence/batch{batch}/{dataset}-evidence.json'
  d,e=J(p),J(q)
  matches=[]
  for version in ('v3','v4'):
   for old in (B/version).glob(f'**/{dataset}-evidence.json'):
    if H(old)==H(p): matches.append(str(old))
  result['evidence_identity'].append(dict(packaged_path=str(p),packaged_sha256=H(p),packaged_protocol=d['protocol_sha256'],actual_path=str(q),actual_sha256=H(q),actual_protocol=e['protocol_sha256'],exact_matches=matches,packaged_epsilon8={arm:d['summaries']['8'][arm]['foreground_positive']['dice'] for arm in ('federated_dp','pooled_dp')},actual_epsilon8={arm:e['summaries']['8'][arm]['foreground_positive']['dice'] for arm in ('federated_dp','pooled_dp')}))
  target=O/f'verified-existing-v5-evidence/batch{batch}';target.mkdir(parents=True,exist_ok=True);shutil.copy2(q,target/q.name)
  protected[str(p)]=H(p);protected[str(q)]=H(q)
for dataset in ('busbra','breast'):
 data=np.load(B/f'features/{dataset}/public-subject-tensors.npz',allow_pickle=False)
 lookup={str(s):i for i,s in enumerate(data['subjects'])}
 for seed in (20260919,20260920,20260921):
  w=B/f'v5/runs-batch16/{dataset}-full-eps8-seed{seed}'
  binding=J(w/'public-pretraining-binding.json'); verify_capture(w,binding)
  z=np.load(w/'public-capture/public-initial-arrays.npz');arr=[z[str(i)] for i in range(6)]
  checkpoint=np.load(binding['checkpoint']); pretrained=[checkpoint[str(i)] for i in range(6)]
  initial=J(w/'public-capture/public-initial.json');split=J(w/'effective-split.json');source=J(w/'source-split.json')
  assert split['test']==source['test'] and set(split['train']).isdisjoint(split['test'])
  rep=load_replicate(w,dataset,'full',8,seed,provenance=B/'campaign-v5/inst/extdata/campaign/segmentation/provenance',batch_size=16)
  modelpath=next((w/'artifact').glob('*/model.pt'))
  model=params.load_user_model(initial['config'],segmentation.FEATURE_DIM,'segmentation_bce_dice')
  model.load_state_dict(torch.load(modelpath,map_location='cpu',weights_only=True),strict=True);model.eval()
  indices=[lookup[s] for s in sorted(split['test'])]; X=data['X'][indices];y=data['y'][indices,:1]
  with torch.no_grad(): probability=np.concatenate([model(torch.from_numpy(X[i:i+16])).sigmoid().numpy() for i in range(0,len(X),16)])
  stored=np.loadtxt(w/'public-probabilities.csv',delimiter=',').reshape(probability.shape)
  fresh=metrics(probability,y);channel=J(w/'channel-b.json')
  assert H(modelpath)==channel['artifact_sha256']==rep['artifact_sha256']
  assert fresh==channel['metrics'], (fresh,channel['metrics'])
  assert np.max(np.abs(stored-probability))<1e-5
  assert A(arr)==A(pretrained) and A(arr)!=binding['initial_random_tensor_sha256']
  row=dict(dataset=dataset,seed=seed,checkpoint_file_sha256=H(Path(binding['checkpoint'])),binding_checkpoint_sha256=binding['checkpoint_sha256'],initial_capture_file_sha256=H(w/'public-capture/public-initial-arrays.npz'),initial_tensor_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in arr],initial_combined_tensor_sha256=A(arr),checkpoint_combined_tensor_sha256=A(pretrained),random_combined_tensor_sha256=binding['initial_random_tensor_sha256'],checkpoint_applied=True,n_train=len(split['train']),n_test=len(split['test']),outer_test_verified=True,model_path=str(modelpath),model_sha256=H(modelpath),fresh_metrics=fresh,foreground_pixel_rate=float((probability>=.5).mean()),subjects_with_foreground=int((probability>=.5).reshape(len(y),-1).any(1).sum()),probability_max_absolute_difference=float(np.max(np.abs(stored-probability))),replicate=rep)
  result['cells'].append(row)
  for p in [modelpath,w/'channel-b.json',w/'public-pretraining-binding.json',w/'public-capture/public-initial-arrays.npz',w/'public-probabilities.csv',w/'twins/twins.json']:
   protected[str(p)]=H(p)
  (O/'audit-results.json').write_text(json.dumps(result,indent=2)+'\n')
  print(dataset,seed,fresh['foreground_positive']['dice'],row['foreground_pixel_rate'],flush=True)
for seed in (20260919,20260920,20260921):
 w=B/f'v5/development/pre60-batch16-rounds20-seed{seed}'
 binding=J(w/'public-pretraining-binding.json');verify_capture(w,binding)
 split=J(w/'effective-split.json'); c=J(w/'channel-b.json')
 result['development'].append(dict(seed=seed,checkpoint_applied=True,n_train=len(split['train']),n_test=len(split['test']),foreground_dice=c['metrics']['foreground_positive']['dice'],split=split))
summary=J(B/'v5/summary-v5.json');result['original_floors']=summary['floors'];result['selection']=summary['selection']
for dataset in ('busbra','breast'):
 rows=[r for r in result['cells'] if r['dataset']==dataset]
 result[dataset+'_epsilon8_means']={arm:float(np.mean([r['replicate'][arm]['foreground_positive']['dice'] for r in rows])) for arm in ('federated_dp','pooled_dp','federated_nonprivate','pooled_nonprivate')}
assert result['busbra_epsilon8_means']['federated_dp']==summary['floors']['16']['across_seed_means']['foreground_dice']
assert all(H(Path(p))==h for p,h in protected.items())
result['protected_files_unchanged']=True
(O/'protected-sha256.json').write_text(json.dumps(protected,indent=2)+'\n')
(O/'audit-results.json').write_text(json.dumps(result,indent=2)+'\n')
print('COMPLETE',result['busbra_epsilon8_means'],result['breast_epsilon8_means'],flush=True)
