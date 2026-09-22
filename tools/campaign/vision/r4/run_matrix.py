#!/usr/bin/env python3
"""Run the declared R4 matrix in the foreground without opening held-out data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from diagnose import controls, extract, save, sha
import train_twins
from stage_training import stage


def main(root):
    controls()
    tools=Path(__file__).resolve().parent
    protocol=json.loads((tools/'protocol.json').read_text())
    assert protocol['status']=='declared_before_corrected_training'
    assert protocol['split_seed']==20260919
    assert not (root/'r4/scoring-lock.json').exists()
    lock=root/'r4/matrix-start.json'
    with lock.open('x') as f:
        json.dump(dict(started_at=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(tools/'protocol.json'),readme_sha256=sha(tools.parent/'README.md'),test_accessed=False),f,indent=2)
    old=json.loads((root/'runs/pytorch_resnet18-eps1-seed20260919/public-capture/public-initial.json').read_text())
    # Fresh final-phase extraction; diagnosis cache is never used for the cell.
    collection=Path('/tmp/cells-vision-r4/final-training')
    save(root/'r4/training-staging.json',stage(root/'prepared/vision/20260919',collection))
    features=extract(collection,old['config'])
    jobs=root/'r4/jobs';jobs.mkdir(exist_ok=False)
    start=time.monotonic()
    for epsilon in protocol['epsilon_order']:
        for seed in protocol['seeds']:
            run=root/'r4/runs'/f'pytorch_resnet18-eps{epsilon}-seed{seed}'
            job=dict(run_dir=str(run),collection_root=str(collection),epsilon=epsilon,seed=seed,
                model_params=protocol['model_params'],n_patients_per_site=[284]*3,
                split_sha256=protocol['split']['sha256_by_seed']['20260919'])
            job_path=jobs/f'eps{epsilon}-seed{seed}.json';save(job_path,job)
            began=time.monotonic()
            print('R4_TRAIN',epsilon,seed,datetime.now(timezone.utc).isoformat(),flush=True)
            subprocess.run(['Rscript',str(tools/'run_federated.R'),str(root),str(job_path)],check=True,timeout=3000)
            train_twins.main(root,run,features=features)
            save(run/'execution-timing.json',dict(elapsed_s=time.monotonic()-began,test_accessed=False))
            print('R4_TRAINED_UNSCORED',epsilon,seed,flush=True)
    save(root/'r4/matrix-complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-start,test_accessed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
