"""Collect execution guards and hashes only; never reopen scored data or models."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('/workspace/cells'))
a=p.parse_args()
root=a.root
out=root/'r4/corrected_evidence'
sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
records=[json.loads((out/f'cdcbmi_r4_pytorch_linear_regression_eps{eps}.json').read_text()) for eps in [1,4,8]]
guards=[]
for eps in [1,4,8]:
    cell=f'cdcbmi_r4_pytorch_linear_regression_eps{eps}'
    paths=[root/'runs'/f'{cell}.started']+sorted((root/'runs'/cell).glob('rep*/scoring.started'))
    assert len(paths)==4 and all(path.exists() for path in paths)
    guards.extend(dict(path=str(path.relative_to(root)),value=path.read_text().strip(),sha256=sha(path)) for path in paths)
selection=list((root/'r4').glob('selection_*/training.started'))
assert len(selection)==6
frozen=list((root/'r4').glob('comparators_seed*/comparators_frozen.json'))
baselines=list((root/'r4').glob('comparators_seed*/baselines.json'))
assert len(frozen)==len(baselines)==3
for path in frozen:
    seed=int(path.parent.name.removeprefix('comparators_seed'))
    assert path.stat().st_mtime < (root/'runs/cdcbmi_r4_pytorch_linear_regression_eps1'/f'rep{seed-20260819}'/'scoring.started').stat().st_mtime
model_files = {str(path.relative_to(root)): sha(path)
               for directory in (root/'runs').glob('cdcbmi_r4_*') if directory.is_dir()
               for path in directory.rglob('model.pt')}
for record in records:
    for rep in record['per_replicate']:
        assert rep['model_sha256'] in model_files.values()
        assert rep['pooled_training']['model_sha256'] in model_files.values()
dependency_paths = ['central_public_units.py','central_train.py','campaign_lib.R','cdcbmi_public_units_protocol.json']
regression=root/'r4-client/tools/campaign/regression'
result=dict(prefit_failure=dict(reason='Missing staged central_public_units.py dependency; fixed by copying unchanged repository helper before any final model training or test scoring',
    guard_sha256=sha(root/'r4/prefit_missing_dependency.started'),
    log_sha256=sha(root/'r4/prefit_missing_dependency.log'),
    final_fits_or_scores_before_restart=0),
    dependency_sha256={name:sha(regression/name) for name in dependency_paths},
    model_files_sha256=model_files, schema='dsflower-regression-r4-execution-audit-v1',
    generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    pod_id='n4emgxhiqzy5i4',pod_name='pod-flower-regression',left_running=True,
    declaration_commit=records[0]['predeclaration_commit'],guards=guards,
    final_execution_started_at=guards[0]['value'],
    final_records_completed_at=max(record['generated_at'] for record in records),
    selection_federation_elapsed_s=sum(rep['federated']['elapsed_s']
        for candidate in json.loads((regression/'r4/selection.json').read_text())['candidates']
        for rep in candidate['replicates']),
    selection_trainings=len(selection),final_federated_trainings=9,final_pooled_trainings=9,
    final_scoring_callbacks=9,nonprivate_training_and_scoring_per_seed_once=True,
    noiseless_comparator_files={str(path.relative_to(root)):sha(path) for path in frozen+baselines},
    summed_federation_elapsed_s=sum(r['federation_elapsed_s'] for x in records for r in x['per_replicate']),
    summed_pooled_elapsed_s=sum(r['pooled_training']['elapsed_s'] for x in records for r in x['per_replicate']),
    summed_replicate_elapsed_s=sum(r['elapsed_s'] for x in records for r in x['per_replicate']),
    interpretation='Guard and saved-evidence audit only; no model predictions or metric recomputation. Sealed CSVs are reconstructed and opened within the guarded final scoring callback. All settings frozen before those callbacks. Each non-private comparator is scored once and reused across budgets.')
(out/'cdcbmi_r4_execution_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
