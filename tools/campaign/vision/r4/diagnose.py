#!/usr/bin/env python3
"""R4 TRAIN-only feature extraction, logistic CV and finite schedule pruning."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from dsflower_runner import client_app, params, server_app, vision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score import metrics
from prepare import write_csv


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def controls():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def extract(collection, cfg):
    encoder, size, is3d, device = vision.prepare_backbone(cfg['backbone'],
        cfg['vision-extractor-profile'], 512, 224)
    xs, ys, ids, image_labels, hashes = [], [], [], [], []
    for i in range(1, 4):
        site = collection / f'site{i}'
        with (site / 'samples.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        paths = [str(site / 'images' / row['relative_path']) for row in rows]
        y = np.asarray([int(row['pathology'] == 'malignant') for row in rows], dtype=np.float32)
        groups = [row['subject_id'] for row in rows]
        x = client_app._totalize_private_features(vision.extract_features_from_paths(encoder, paths, size, is3d, device))
        x, py = client_app._pool_by_patient(x, y, groups, 'cross_entropy')
        x = client_app._totalize_private_features(x)
        xs.append(x); ys.append(py); ids.extend(dict.fromkeys(groups)); image_labels.extend(y)
        hashes.append(dict(site=i, features_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            targets_sha256=hashlib.sha256(py.tobytes()).hexdigest(), manifest_sha256=sha(site / 'samples.csv')))
        print('EXTRACTED_TRAIN_SITE', i, len(x), flush=True)
    assert len(set(ids)) == len(ids)
    return xs, ys, np.asarray(ids), np.asarray(image_labels), hashes


def logistic(x, y):
    """Fixed C=1 L2 logistic regression, unpenalized intercept; same affine class."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    def objective(v):
        z = x @ v[:-1] + v[-1]
        residual = expit(z) - y
        loss = np.mean(np.logaddexp(0, z) - y*z) + .5*np.dot(v[:-1], v[:-1])/len(y)
        grad = np.r_[x.T @ residual / len(y) + v[:-1]/len(y), residual.mean()]
        return loss, grad
    fit = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method='L-BFGS-B',
        options=dict(maxiter=10000, ftol=1e-12, gtol=1e-9, maxls=50))
    assert fit.success, fit.message
    return fit.x, dict(converged=bool(fit.success), iterations=int(fit.nit), objective=float(fit.fun),
        gradient_inf_norm=float(np.max(np.abs(fit.jac))), message=str(fit.message), C=1.0)


def predict_logistic(v, x):
    return expit(x.astype(np.float64) @ v[:-1] + v[-1])


def finite(cfg, x, y, vx, vy, candidate, seed):
    torch.manual_seed(seed)
    model = server_app._build_initial_model(cfg).cuda()
    tx, ty = torch.from_numpy(x).cuda(), torch.from_numpy(y).long().cuda()
    vt = torch.from_numpy(vx).cuda()
    steps = math.ceil(227/candidate['batch_size'])
    divisor = max(1, len(x)//steps)
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    for _ in range(5):
        cls = torch.optim.Adam if candidate['optimizer'] == 'adam' else torch.optim.SGD
        opt = cls(model.parameters(), lr=candidate['learning_rate'], weight_decay=candidate['weight_decay'])
        for _ in range(candidate['local_epochs']*steps):
            idx = np.flatnonzero(rng.random(len(x)) < 1/steps)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(tx[idx]), ty[idx], reduction='sum')/divisor if len(idx) else model(tx[:1]).sum()*0
            loss.backward(); opt.step()
    with torch.no_grad():
        p = model(vt).softmax(1)[:, 1].cpu().numpy()
    return dict(candidate=candidate, metrics=metrics(vy,p), steps=steps*5*candidate['local_epochs'],
        sample_rate=1/steps, expected_batch_divisor=divisor, elapsed_s=time.monotonic()-start,
        interpretation='Pooled nonprivate head, site-matched step count and Poisson rate; no clipping or noise.')


def inner_collections(root, ids):
    source = root / 'prepared/vision/20260919'
    destination = root / 'r4/diagnosis/collections'
    lookup = {}
    for i in range(1,4):
        site = source / f'site{i}'
        with (site/'samples.csv').open() as f:
            for row in csv.DictReader(f):
                lookup.setdefault(row['subject_id'], []).append((site, row))
    # Inner site assignment depends only on identifiers and the public seed.
    order = sorted(map(str, ids), key=lambda v: hashlib.sha256(('r4-sites|'+v).encode()).digest())
    site_ids = [order[i::3] for i in range(3)]
    assert [len(v) for v in site_ids] == [227]*3
    for i, group in enumerate(site_ids,1):
        target = destination/f'site{i}'
        (target/'images').mkdir(parents=True, exist_ok=False)
        metadata, manifests, index = [], [], []
        for subject in group:
            for oldsite, row in lookup[subject]:
                image = target/'images'/row['relative_path']
                shutil.copyfile(oldsite/'images'/row['relative_path'], image)
                metadata.append(row)
                manifests.append(dict(sample_id=row['image_id'], source_kind='single_file',
                    primary_uri=image.name, files_json=json.dumps([dict(path=image.name,role='primary')]), content_hash=sha(image), n_files=1))
                index.append(dict(sample_id=row['image_id'], uri=str(image), content_hash=sha(image),size=image.stat().st_size,source_kind='single_file'))
        write_csv(target/'samples.csv',metadata); write_csv(target/'sample_manifests.csv',manifests); write_csv(target/'content_hash_index.csv',index)
        dataset=f'busbra.site{i}'
        save(target/'manifest.yaml',dict(schema_version=1,dataset_id=dataset,
            metadata=dict(uri=str(target/'samples.csv'),format='csv',id_col='image_id',privacy_unit='patient',privacy_unit_col='subject_id',
                privacy_unit_canonicalization='trim-utf8-v2',label_col='pathology',label_levels=['benign','malignant']),
            assets=dict(images=dict(type='image_root',uri=str(target/'images'),path_col='relative_path')),
            sample_manifests=dict(uri=str(target/'sample_manifests.csv'),format='csv'),
            content_hash_index=dict(uri=str(target/'content_hash_index.csv'),format='csv')))
        save(target/'registry.yaml',{'schema_version':1,dataset:dict(enabled=True,backend='file',manifest_uri=str(target/'manifest.yaml'))})
    return destination, [ids_hash(v) for v in site_ids]


def main(root):
    controls()
    out=root/'r4/diagnosis'
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(Path(__file__).with_name('diagnosis-protocol.json').read_text())
    seed=protocol['diagnosis']['seed']
    old=root/'runs/pytorch_resnet18-eps1-seed20260919/public-capture/public-initial.json'
    cfg=json.loads(old.read_text())['config']
    xs,ys,ids,image_labels,hashes=extract(root/'prepared/vision/20260919',cfg)
    x,y=np.concatenate(xs),np.concatenate(ys)
    assert x.shape==(852,512)
    # Never read a split JSON, test metadata, held-out image or old predictions.
    np.savez(out/'training-features.npz',x=x,y=y,ids=ids,image_labels=image_labels,
        **{f'x{i}':a for i,a in enumerate(xs)},**{f'y{i}':a for i,a in enumerate(ys)})
    rng=np.random.default_rng(seed)
    classes=[rng.permutation(np.flatnonzero(y==c)) for c in (0,1)]
    folds=[np.concatenate([np.array_split(c,5)[i] for c in classes]) for i in range(5)]
    cv=[]; oof=np.empty(len(y))
    for i,val in enumerate(folds):
        train=np.setdiff1d(np.arange(len(y)),val)
        v,fit=logistic(x[train],y[train]); p=predict_logistic(v,x[val]); oof[val]=p
        cv.append(dict(fold=i,metrics=metrics(y[val],p),fit=fit,n_train=len(train),n_validation=len(val),
            train_ids_sha256=ids_hash(ids[train]),validation_ids_sha256=ids_hash(ids[val])))
        print('LOGISTIC_CV',json.dumps(cv[-1]),flush=True)
    n0=round(171*len(classes[0])/852)
    val=np.concatenate([classes[0][:n0],classes[1][:171-n0]])
    train=np.setdiff1d(np.arange(852),val)
    assert len(train)==681 and len(val)==171
    np.savez(out/'inner-indices.npz',train=train,validation=val)
    collection,site_hashes=inner_collections(root,ids[train])
    audit=dict(started_at=datetime.now(timezone.utc).isoformat(),test_accessed=False,split_seed=20260919,
        training_only_source=str(root/'prepared/vision/20260919'),n_training_patients=852,n_training_images=len(image_labels),
        training_ids_sha256=ids_hash(ids),inner_train_ids_sha256=ids_hash(ids[train]),inner_validation_ids_sha256=ids_hash(ids[val]),
        hashed_inner_train_ids=[hashlib.sha256(s.encode()).hexdigest() for s in ids[train]],
        hashed_inner_validation_ids=[hashlib.sha256(s.encode()).hexdigest() for s in ids[val]],
        inner_site_ids_sha256=site_hashes,feature_hashes=hashes,diagnosis_cache_sha256=sha(out/'training-features.npz'),
        logistic_cv=cv,logistic_cv_mean_auc=float(np.mean([v['metrics']['auc'] for v in cv])),
        logistic_cv_sd_auc=float(np.std([v['metrics']['auc'] for v in cv],ddof=1)),logistic_oof=metrics(y,oof),
        selection_rule=protocol['diagnosis']['selection_rule'])
    save(out/'audit.json',audit)
    candidates=[dict(id='registry_default',learning_rate=.001,local_epochs=1,batch_size=32,optimizer='sgd',weight_decay=0)]
    for optimizer,rates in [('sgd',[.01,.03,.1]),('adam',[.001,.003,.01])]:
        for lr in rates:
            for epochs in [5,20]:
                for batch in [32,128]:
                    candidates.append(dict(id=f'{optimizer}_lr{lr}_e{epochs}_b{batch}',learning_rate=lr,local_epochs=epochs,batch_size=batch,optimizer=optimizer,weight_decay=0))
    save(out/'candidates-before-sweep.json',candidates)
    results=[]
    for c in candidates:
        row=finite(cfg,x[train],y[train],x[val],y[val],c,seed)
        results.append(row); save(out/'nonprivate-sweep.json',results)
        print('FINITE',json.dumps(row),flush=True)
    ranked=sorted(results,key=lambda r:(-r['metrics']['auc'],r['candidate']['id']))[:2]
    for row in ranked:
        candidate=row['candidate']; name=candidate['id']
        model_params=dict(protocol['model_params'],**{k:v for k,v in candidate.items() if k!='id'})
        job=dict(run_dir=str(out/'federations'/name),collection_root=str(collection),epsilon=8,seed=seed,
            model_params=model_params,n_patients_per_site=[227]*3,split_sha256=audit['inner_train_ids_sha256'],candidate_id=name)
        save(out/f'job-{name}.json',job)
    save(out/'shortlist.json',[r['candidate'] for r in ranked])
    print('DIAGNOSIS_PRUNING_COMPLETE',audit['logistic_cv_mean_auc'],[r['candidate']['id'] for r in ranked],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
