#!/usr/bin/env python3
"""Retain the investigated pre-training failure before its sole recovery attempt."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

root=Path(sys.argv[1]).resolve()
base=root/'runtime/hazard_v3'
archive=root/'dsFlowerClient/inst/extdata/campaign/survival/hazard-v3'
identity='dev-g01-seed1102'
failed=archive/'development'/f'cell-{identity}.json'
record=json.loads(failed.read_text())
assert record['status']=='failed' and record['cleanup_ok'] is True
assert record['error']=='SuperNode startup failed or returned no ACK on: site1.'
assert 'results' not in record and 'artifact_checksum' not in record
assert not list((base/'runs'/identity).rglob('model.pt'))
assert not (archive/'selection.json').exists() and not (archive/'confirmation_started.json').exists()
records=[json.loads(p.read_text()) for p in (archive/'development').glob('*.json')]
assert len(records)==8 and sum(r['status']=='executed' for r in records)==7
assert all(r['cleanup_ok'] for r in records)
# Refuse overlapping campaign drivers; never print process command lines.
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:args=(proc/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert not any(a.endswith(b'/hazard_v3/run.py') for a in args),'original driver still active'
dest=archive/'failures';dest.mkdir(exist_ok=True)
digest=hashlib.sha256(failed.read_bytes()).hexdigest()
shutil.copyfile(base/'runs'/identity/'federation/failure-diagnostics.log',dest/'startup-diagnostics.log')
failed.rename(dest/failed.name)
value=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),recovery_identity=identity,
    failure_file='failures/'+failed.name,failure_sha256=digest,
    phase='SuperNode startup before server training submission',scored_model_returned=False,
    exact_low_level_cause='Unavailable: package masks the underlying startup exception; no site1 SuperNode log was captured.',
    observed='Three other federations launched simultaneously. No OOM events. Shared spawn lock timeout10 seconds; 2-second spawn checks under lock. Trusted environment strips OMP/BLAS variables.',
    interpretation='Concurrent startup lock contention is plausible, not proven. Separately verified CPU oversubscription explains slow first-wave runtime.',
    resource_repair='30-second launch spacing and one-CPU OS affinity per federation; no package or privacy change.',
    thread_probe='Pinned CPU0, clean default PyTorch reports get_num_threads()==1; original affinity has32 CPUs.',
    scope='Reuse seven successful first-wave cells; recover only this pre-training failure once in a new directory; no candidate omission or score-driven retry.',
    original_configuration_and_split_unchanged=True,
    original_failure_unchanged=True)
with (archive/'infrastructure_investigation.json').open('x') as handle:
    json.dump(value,handle,indent=2);handle.write('\n')
print('PRETRAINING_FAILURE_RETAINED; seven successful cells preserved')
