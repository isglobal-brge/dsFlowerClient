#!/usr/bin/env python3
"""Export public diagnosis records, excluding patient tensors and identifiers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main(root):
    source=root/'r5'
    destination=source/'export'
    destination.mkdir(exist_ok=True)
    names=['runtime-preflight.json','pod-initial.json','pod-final.json','validation.json',
        'finiteclamp-validation.json','shortlist-validation.json','shortlist-validation.log',
        'aggregation-validation.json','aggregation-validation.log',
        'inner-feature-parity.json','search-declaration.json',
        'mechanisms.json','ranked-candidates.json','shortlist.json','dp-confirmation.json',
        'selection.json','training-staging.json','validation.log','validation-outer-cache-mismatch.log',
        'finiteclamp-validation.log','inner-feature-parity.log','emulation.log',
        'runner-final.json','runner-final-import-timeout.json','tooling-sha256.json','preservation.json','VISION_DIAGNOSIS_R5.md']
    for name in names:
        if (source/name).exists():
            shutil.copyfile(source/name,destination/name)
    for run in (source/'federations').iterdir():
        target=destination/'federations'/run.name
        target.mkdir(parents=True,exist_ok=True)
        for name in ['job.json','federation-status.json','inner-validation.json','startup-wait.json','node-contract.json']:
            if (run/name).exists():
                shutil.copyfile(run/name,target/name)
        public=target/'public-capture';public.mkdir(exist_ok=True)
        for path in (run/'public-capture').glob('*.json'):
            shutil.copyfile(path,public/path.name)
        shutil.copyfile(run/'public-capture/public-initial-arrays.npz',public/'public-initial-arrays.npz')
        status=json.loads((run/'federation-status.json').read_text())
        artifact=Path(status['output_dir'])
        assert hashlib.sha256((artifact/'model.pt').read_bytes()).hexdigest() == status['model_sha256']
        for name in ['model.pt','metadata.json','history.json']:
            if (artifact/name).exists():
                shutil.copyfile(artifact/name,target/name)
    hashes={str(path.relative_to(destination)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob('*')) if path.is_file() and path.name!='SHA256SUMS'}
    (destination/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in hashes.items()))
    print('Exported',len(hashes),'public records; raw arrays remain on the vision pod.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
