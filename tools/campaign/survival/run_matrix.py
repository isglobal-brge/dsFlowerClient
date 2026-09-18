#!/usr/bin/env python3
"""Run the preregistered public matrix after all three synthetic gates pass."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import datetime
import os
from pathlib import Path
import shutil
import subprocess


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('workspace',type=Path)
    ap.add_argument('--jobs',type=int,default=2)
    args=ap.parse_args()
    root=args.workspace.resolve()
    archive=root/'dsFlowerClient/inst/extdata/campaign/survival'
    for variant in ('weibull','lognormal','hazard'):
        pilot=archive/f'cell-synthetic-{variant}.json'
        if not pilot.exists() or json.loads(pilot.read_text()).get('status')!='executed':
            raise SystemExit('All three final synthetic gates must execute before cohort scoring')
    script=root/'dsFlowerClient/tools/campaign/survival/run_cell.R'
    matrix=[]
    for dataset,subsets in [('support2',['full','small600','heterogeneous']),('lung1',['full'])]:
        for subset in subsets:
            for variant in ['weibull','lognormal','hazard']:
                for epsilon in ([8] if subset=='heterogeneous' else [1,4,8]):
                    for seed in [1101,1102,1103]:
                        identity=f'{dataset}-{subset}-{variant}-eps{epsilon}-seed{seed}'
                        matrix.append((identity,dataset,subset,variant,epsilon,seed))
    def run(item):
        identity,dataset,subset,variant,epsilon,seed=item
        dest=archive/f'cell-{identity}.json'
        if dest.exists() and json.loads(dest.read_text()).get('status')=='executed':
            return identity,'already executed'
        out=root/'runtime/runs'/identity
        if (out/'evidence.json').exists():
            raise RuntimeError('Prior failed attempt requires explicit investigation and a new attempt directory: '+identity)
        split=root/'data/survival/splits'/f'{dataset}-{subset}-{seed}'
        env=os.environ.copy()
        env.update(R_LIBS_USER=str(root/'runtime/rlib'),TMPDIR=str(root/'runtime/tmp'))
        (root/'runtime/tmp').mkdir(exist_ok=True)
        with (root/'logs'/f'{identity}.log').open('w') as log:
            result=subprocess.run(['Rscript',str(script),str(root),str(split),variant,str(epsilon),str(out),'10','2'],
                                  cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        path=out/'evidence.json'
        if not path.exists():
            meta=json.loads((split/'split.json').read_text())
            build=json.loads((root/'runtime/build.json').read_text())
            failure=dict(schema_version=1,record_type='cell',task='survival',status='failed',
                phase='preflight',dataset=meta,variant=variant,epsilon=epsilon,delta=1e-5,clip=1,
                privacy_unit='patient',adjacency='replace_one',site_count=3,
                cleanup_ok=False,
                package_commits=build['commits'],installed_build=build,
                runner_sha256=build['runner_sha256'],
                error=f'Rscript exited {result.returncode} before evidence output; see task-owned log {identity}.log',
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            dest.write_text(json.dumps(failure,indent=2)+'\n')
            return identity,'failed during infrastructure preflight'
        evidence=json.loads(path.read_text())
        shutil.copyfile(path,dest)
        return identity,evidence['status']
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(run,item) for item in matrix]):
            identity,status=future.result()
            print(identity,status,flush=True)
    subprocess.run(['python3',str(script.with_name('summarize.py')),str(archive)],check=True)


if __name__=='__main__':
    main()
