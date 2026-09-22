#!/usr/bin/env python3
"""Export an explicit allowlist of public R4 records; never copy node state."""
import argparse
from pathlib import Path
import shutil


def main(root, destination):
    work=root/'r4/implementation'
    destination.mkdir(parents=True,exist_ok=False)
    def copy(source, relative):
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for name in ('plan.json','sweep.json','selection.json'):
        copy(work/'selection'/name,Path('selection')/name)
    for run in (work/'selection').iterdir():
        if not run.is_dir() or not (run/'federation-status.json').is_file():
            continue
        for name in ('request.json','protocol.json','execution-status.json','federation-status.json',
                     'node-contract.json','verification.json','inner-metrics.json'):
            copy(run/name,Path('selection')/run.name/name)
        for path in (run/'public-capture').glob('accountant-*.json'):
            copy(path,Path('selection')/run.name/'public-capture'/path.name)
        copy(run/'public-capture/public-initial.json',Path('selection')/run.name/'public-capture/public-initial.json')
        copy(work/'selection'/(run.name+'.log'),Path('selection/logs')/(run.name+'.log'))
    for name in ('protocol.json','pre_scoring_verification.json','test-scoring-started.json','runtime.json'):
        path=work/name
        if path.exists():
            copy(path,Path(name))
    if (work/'evidence').exists():
        for path in (work/'evidence').glob('*.json'):
            copy(path,Path('evidence')/path.name)
    if (work/'runs').exists():
        for path in (work/'runs').glob('*.log'):
            copy(path,Path('logs')/path.name)
        for run in (work/'runs').iterdir():
            if not run.is_dir():
                continue
            for name in ('federation-status.json','execution-status.json','verification.json','protocol.json'):
                copy(run/name,Path('runs')/run.name/name)
    print(destination)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    main(a.root,a.out)
