#!/usr/bin/env python3
"""Finish a timed-out import check without changing packages or repeating fits."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verify_runtime import probe

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
path=a.root/'r5/runner-final.json'
record=json.loads(path.read_text())
assert record['installed_versions']==record['expected_versions']
assert all(v==record['expected_runner_sha256'] for v in record['installed_runner_sha256'].values())
assert record['release_probe']['returncode']==0 and record['torch_probe']['timed_out']
failed=path.with_name('runner-final-import-timeout.json')
assert not failed.exists()
path.rename(failed)
record['torch_probe']=probe(record['torch_probe']['argv'],120)
record['import_timeout_recovery']=dict(previous_record=failed.name,original_timeout_s=45,
    retry_timeout_s=120,training_repeated=False,packages_changed=False)
passed=record['torch_probe']['returncode']==0
record.update(status='verified' if passed else 'blocked',blocker_kind=None if passed else 'import_check_failed',
    error=None if passed else 'Bounded final import check did not complete successfully.')
path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(record['status'],record['torch_probe'],flush=True)
assert passed
