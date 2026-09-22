#!/usr/bin/env python3
"""Foreground development, immutable selection, and a single confirmation pass."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading
import urllib.request

from protocol import (SEEDS,DEV_SEEDS,SOURCES,grid,sha,write,prepare_outer,
                      prepare_inner,config_for_seed,prepare_confirmation,select)


def now(): return datetime.now(timezone.utc).isoformat()

# Avoid contention on the unchanged server-global 10-second spawn lock.
launch_lock=threading.Lock()
last_launch=0.
launch_number=0


def launch_command(command):
    global last_launch,launch_number
    with launch_lock:
        remaining=30-(time.monotonic()-last_launch)
        if remaining>0:
            threading.Event().wait(remaining)  # one fixed startup spacing, no polling
        cpus=sorted(os.sched_getaffinity(0))
        cpu=cpus[launch_number%len(cpus)]
        launch_number+=1
        last_launch=time.monotonic()
    return ['taskset','-c',str(cpu)]+command,cpu


def run_cell(root,base,archive,phase,identity,split,cfg,epsilon):
    tools=root/'dsFlowerClient/tools/campaign/survival/hazard_v3'
    out=base/'runs'/identity
    config_path=base/'configs'/f'{identity}.json'
    write(config_path,cfg)
    env=os.environ.copy()
    env.update(R_LIBS_USER=str(root/'runtime/rlib'),TMPDIR=str(root/'runtime/tmp'),
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=now()
    command,cpu=launch_command(['Rscript',str(tools/'run_cell.R'),str(root),str(split),
        str(epsilon),str(out),str(config_path),phase])
    with (base/'logs'/f'{identity}.log').open('x') as log:
        result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    path=out/'evidence.json'
    if path.exists():
        record=json.loads(path.read_text())
    else:
        record=dict(schema_version=1,record_type='cell',task='survival',status='failed',
            protocol_version=3,variant='hazard',contract='pytorch_discrete_hazard',epsilon=epsilon,
            delta=1e-5,clip=1,privacy_unit='patient',adjacency='replace_one',cleanup_ok=False,
            dataset=json.loads((split/'split.json').read_text()),
            started_utc=started,finished_utc=now(),error=f'R exit {result.returncode} before evidence; retained runtime log')
    record.update(hazard_v3_phase=phase,hazard_v3_config=cfg,driver_returncode=result.returncode,
        resource_policy=dict(cpu_affinity=[cpu],startup_spacing_seconds=30,reason='OS resource scheduling; package thread environment is intentionally sanitized'))
    dest=archive/('development' if phase=='development' else 'pilot' if phase=='pilot' else '')
    dest.mkdir(parents=True,exist_ok=True)
    write(dest/f'cell-{identity}.json',record)
    if phase!='development' and record['status']=='executed':
        exported=archive/'artifacts'/identity
        exported.mkdir(parents=True)
        # Whitelist only released public artifacts; no recursive runtime copying.
        models=list((out/'federation/artifact').glob('*/model.pt'))
        assert len(models)==1 and sha(models[0])==record['artifact_checksum']
        for name in ('model.pt','metadata.json','history.json'):
            shutil.copyfile(models[0].parent/name,exported/name)
        for name in ('config.json','scores.json','central.pt','central_dp.pt','null.pt'):
            shutil.copyfile(out/name,exported/name)
    print(now(),identity,record['status'],
          record.get('results',{}).get('federated_dp',{}).get('c_index'),flush=True)
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('workspace',type=Path)
    ap.add_argument('phase',choices=['prepare','pilot','run'])
    ap.add_argument('--jobs',type=int,default=4)
    ap.add_argument('--resume-after-startup-failure',action='store_true')
    ap.add_argument('--sweep-cutoff',default='2026-09-22T02:40:00+00:00')
    args=ap.parse_args()
    root=args.workspace.resolve()
    tools=root/'dsFlowerClient/tools/campaign/survival/hazard_v3'
    protocol=tools/'PROTOCOL.md'
    base=root/'runtime/hazard_v3'
    archive=root/'dsFlowerClient/inst/extdata/campaign/survival/hazard-v3'
    if args.phase=='prepare':
        base.mkdir()
        for folder in ('logs','configs','runs'): (base/folder).mkdir(mode=0o700)
        raw=root/'data/survival/support2.csv'
        if not raw.exists(): urllib.request.urlretrieve(SOURCES['support2']['url'],raw)
        prepare_outer(raw,base/'outer',protocol)
        for seed in SEEDS:
            prepare_inner(base/'outer'/f'support2-full-{seed}',base/'inner'/str(seed),protocol)
        write(archive/'preregistration.json',dict(protocol_version=3,created_utc=now(),
            grid=grid(),development_seeds=DEV_SEEDS,confirmation_seeds=SEEDS,
            protocol_sha256=sha(protocol),diagnosis_sha256=sha(archive/'HAZARD_V3_SUMMARY.md'),
            public_source_sha256=sha(raw),pod_id='x0w6ewmpinpsuk',
            tool_sha256={p.name:sha(p) for p in sorted(tools.glob('*')) if p.is_file()},
            inner_manifest_sha256={str(s):sha(base/'inner'/str(s)/'split.json') for s in SEEDS}))
        print('PREPARED; no fit or outer holdout score',flush=True)
        return
    if args.phase=='pilot':
        subprocess.run([sys.executable,str(tools.parent/'prepare_synthetic.py'),
            str(base/'synthetic'),str(protocol)],check=True)
        meta_path=base/'synthetic/split.json'
        meta=json.loads(meta_path.read_text());meta['evaluation_role']='synthetic integration'
        meta_path.write_text(json.dumps(meta,indent=2)+'\n')
        cfg=dict(grid()[7],id='pilot',rounds=2,local_epochs=1,strategy='fedavgm',
            edges=[0,7,30,180,730,1825])
        record=run_cell(root,base,archive,'pilot','synthetic-hazard-v3',base/'synthetic',cfg,8)
        assert record['status']=='executed'
        return
    if args.resume_after_startup_failure:
        investigation=json.loads((archive/'infrastructure_investigation.json').read_text())
        assert investigation['recovery_identity']=='dev-g01-seed1102'
        assert not (archive/'selection.json').exists() and not (archive/'confirmation_started.json').exists()
    write(base/('infrastructure_resume.json' if args.resume_after_startup_failure else 'run_started.json'),dict(started_utc=now(),sweep_cutoff=args.sweep_cutoff,jobs=args.jobs,
        tool_sha256={p.name:sha(p) for p in sorted(tools.glob('*')) if p.is_file()}))
    pilot=json.loads((archive/'pilot/cell-synthetic-hazard-v3.json').read_text())
    assert pilot['status']=='executed'
    allcfg=grid()
    configs={cfg['id']:{s:config_for_seed(cfg,base/'inner'/str(s)) for s in SEEDS} for cfg in allcfg}
    if args.resume_after_startup_failure:
        assert json.loads((archive/'frozen_configurations.json').read_text())==json.loads(json.dumps(configs))
    else:
        write(archive/'frozen_configurations.json',configs)
    rows=[]
    cutoff=datetime.fromisoformat(args.sweep_cutoff)
    for offset in range(0,len(allcfg),4):
        if datetime.now(timezone.utc)>=cutoff:
            break
        wave=allcfg[offset:offset+4]
        tasks=[(cfg,seed) for cfg in wave for seed in DEV_SEEDS]
        def dev(item):
            cfg,seed=item
            identity=f'dev-{cfg["id"]}-seed{seed}'
            previous=archive/'development'/f'cell-{identity}.json'
            if args.resume_after_startup_failure and previous.exists():
                record=json.loads(previous.read_text())
                assert record['status']=='executed' and record['hazard_v3_config']==configs[cfg['id']][seed]
                print('REUSE_EXECUTED',identity,flush=True)
                return record
            if args.resume_after_startup_failure and identity==investigation['recovery_identity']:
                identity+='-startup-recovery'
            return run_cell(root,base,archive,'development',identity,
                base/'inner'/str(seed),configs[cfg['id']][seed],8)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            records=list(pool.map(dev,tasks))
        if any(r['status']!='executed' or r['driver_returncode'] for r in records):
            raise RuntimeError('Development failure retained; selection and confirmation prohibited')
        for cfg in wave:
            row=dict(config=cfg,scores={str(r['dataset']['seed']):r['results']['federated_dp']['c_index']
                for r in records if r['hazard_v3_config']['id']==cfg['id']})
            rows.append(row)
        print('COMPLETED_CONFIGURATIONS',len(rows),flush=True)
    ranked=select(rows)
    selected=ranked[0]['config']
    # Inclusion is decided using elapsed time only, before any confirmation CSV opens.
    envelopes=datetime.now(timezone.utc)<datetime.fromisoformat('2026-09-22T02:50:00+00:00')
    arms=['full','two-sites']+(['heterogeneous','small600'] if envelopes else [])
    selection=dict(protocol_version=3,selected=selected,ranked=ranked,selected_utc=now(),
        rule='maximum mean inner-validation federated-DP C-index; total epochs; K; ID',
        per_seed_configs=configs[selected['id']],confirmation_arms=arms,
        omitted_configurations=allcfg[len(rows):],omission_rule='unstarted suffix at predeclared wall-clock cutoff',
        development_sha256={p.name:sha(p) for p in sorted((archive/'development').glob('*.json'))},
        protocol_sha256=sha(protocol),preregistration_sha256=sha(archive/'preregistration.json'))
    write(archive/'selection.json',selection)
    with (archive/'sweep.csv').open('x',newline='') as handle:
        fields=list(selected)+['seed1101','seed1102','mean','rank']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for rank,row in enumerate(ranked,1):
            writer.writerow(dict(row['config'],seed1101=row['scores']['1101'],seed1102=row['scores']['1102'],mean=row['mean'],rank=rank))
    write(archive/'confirmation_started.json',dict(started_utc=now(),selection_sha256=sha(archive/'selection.json'),arms=arms))
    splits={(arm,seed):prepare_confirmation(base/'outer'/f'support2-full-{seed}',
        base/'confirmation'/f'{arm}-{seed}',protocol,arm) for arm in arms for seed in SEEDS}
    tasks=[(arm,eps,seed) for arm in arms for eps in ([1,4,8] if arm=='full' else [8]) for seed in SEEDS]
    def confirm(item):
        arm,eps,seed=item
        return run_cell(root,base,archive,'confirmation',f'support2-{arm}-hazard-v3-eps{eps}-seed{seed}',
            splits[arm,seed],configs[selected['id']][seed],eps)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records=list(pool.map(confirm,tasks))
    write(archive/'confirmation_complete.json',dict(finished_utc=now(),
        status='executed' if all(r['status']=='executed' for r in records) else 'incomplete',
        selection_sha256=sha(archive/'selection.json'),attempted_cells=len(records)))
    subprocess.run([sys.executable,str(tools/'report.py'),str(archive)],check=True)


if __name__=='__main__': main()
