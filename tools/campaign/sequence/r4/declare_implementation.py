#!/usr/bin/env python3
"""Freeze the winning TRAIN-only schedule before any full-data R4 training."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

TOOLS=Path(__file__).resolve().parent
REPO=TOOLS.parents[3]
selection_path=TOOLS/'selection/selection.json'
selection=json.loads(selection_path.read_text())
s=selection['selected']['schedule']
r3=json.loads((TOOLS.parent/'r3/protocol.json').read_text())
old=json.loads((REPO/'inst/extdata/campaign/sequence/har_window_pytorch_lstm_eps8.json').read_text())
model=dict(r3['model_params'],learning_rate=s['lr'],local_epochs=s['epochs'],batch_size=s['batch'])
protocol=dict(revision='R4',declared_at=datetime.now(timezone.utc).isoformat(),
    contract='pytorch_lstm',units=['window'],rounds=5,seeds=selection['seeds'],epsilon_order=[1,4,8],
    delta=1e-6,clipping_norm=1,model_params=model,
    selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    selection_criterion=selection['criterion'],inner_selection=selection,
    site_subjects=old['dataset']['site_subjects'],train_subjects=old['dataset']['train_subjects'],
    n_per_site=old['dataset']['n_per_site'],
    steps_per_site=[math.ceil(n/s['batch'])*s['epochs']*5 for n in old['dataset']['n_per_site']],
    layout='Raw C-order token-major flat rows, recurrent input [N,128,9], DPLSTM hidden32, last hidden state, six logits.',
    channel_abs_bounds=[1,1,1,1,1,1,2,2,2],
    preprocessing='Contract applies clip(x,-b,b)/b once; fixed public design constants, not empirical extrema.',
    central='Reuse frozen R3 model identities and once-scored metrics after exact split/bounds/init verification: Adam .01, batch256, 20 epochs, reset each four epochs, 580 steps.',
    nonprivate_federated='Selected five-round schedule, equal site weights, paired public Poisson streams, expected-batch divisor, no clipping or noise, reset optimizer each round.',
    clipped_noiseless='Same as nonprivate twin, installed _dp_fit with sigma0: coordinate clamp +/-1 then per-window global norm1 clipping.',
    pooled_dp='Not scheduled; prioritize the three-site DP matrix and both noiseless controls.',
    trivial='TRAIN class frequencies; TRAIN majority class.',
    scoring='Exclusive scoring marker after all training and verification; official TEST read once; each new model predicted once. No alternative or retraining after scoring.',
    metrics=['macro_one_vs_rest_auc','accuracy','log_loss'],gap='Federated-DP macro-AUC minus paired central; mean and sample SD over three seeds.',
    privacy_scope='Window-level mechanism measurement on subject-disjoint sites; no subject-level protection. No end-to-end private selection or campaign-wide composition guarantee.',
    randomness='Public initialization seeds; unchanged node-owned cryptographic sampling/noise.',
    package_changes='None',dataset_citation_keys=['anguita_har_2013','uci_har'])
with (TOOLS/'implementation_protocol.json').open('x') as f:
    json.dump(protocol,f,indent=2,allow_nan=False);f.write('\n')
lines=['# Corrected R4 window-level cell: pre-run declaration','',
    f"Declared {protocol['declared_at']} before full-data R4 training or TEST scoring.",
    f"Selected **Adam {s['lr']}, batch {s['batch']}, {s['epochs']} local epochs × five rounds**,",
    'hidden32, no scheduler or penalties. Selection uses TRAIN subjects only,',
    'with the diagnosis’s unchanged subject-disjoint inner split: fit subjects',
    ', '.join(map(str,selection['train_subjects']))+'; validation subjects '+', '.join(map(str,selection['validation_subjects']))+'.',
    'Three subject-disjoint sites retain their original membership after removing',
    'inner-validation subjects. The two highest noiseless-clipped diagnosis',
    'candidates were compared using the REAL window-DP contract at ε=8 across',
    'seeds 20260922–20260924; final-round mean inner-validation macro-AUC selects',
    'the winner. No early checkpoint selection or held-out access was used.','',
    '| Adam LR / local epochs / batch | Inner AUC mean ± SD | Accuracy mean ± SD | Log-loss mean ± SD |',
    '|---|---:|---:|---:|']
for record in selection['sweep']:
    c=record['schedule']; m=record['summary']
    lines.append(f"| {c['lr']} / {c['epochs']} / {c['batch']} | "+' | '.join(f"{m[k]['mean']:.6f} ± {m[k]['sd']:.6f}" for k in ('macro_auc','accuracy','log_loss'))+' |')
lines+=['',
    'The corrected cell uses ε∈{1,4,8}, δ=1e-6, window unit, unit clipping, the',
    'same 21 TRAIN and nine held-out subjects, original three sites (2,553 /',
    '2,397 / 2,402 windows), split and three seeds as `har_window_*`.',
    f"The selected schedule has {protocol['steps_per_site']} optimizer steps per site.",
    'Raw 128×9 token-major windows receive the public channel-bound transform',
    '`clip(x,-b,b)/b` once, with `b=(1,1,1,1,1,1,2,2,2)`.',
    '**This is a window-level mechanism measurement on subject-disjoint sites',
    'and provides no subject-level protection.**','',
    'Comparators on the identical split: the same network centrally trained',
    'without privacy under the R3 nominal schedule (Adam .01, batch256, 20 epochs,',
    'Adam reset every four epochs, 580 steps); selected-schedule non-private',
    'federated twin without clipping or noise; selected-schedule clipped noiseless',
    'federated twin; and TRAIN-frequency/majority trivial predictor. Reuse the',
    'frozen R3 central model identities and metrics after exact split, bounds and',
    'initialization verification. The two noiseless twins use matched public',
    'Poisson streams and equal site weights; real DP retains node-owned randomness.',
    'The optional pooled-DP twin is not scheduled.','',
    'Report macro one-vs-rest AUC, accuracy and log-loss, mean ± sample SD over',
    'three seeds, and paired federated-DP minus central AUC gaps. Diagnostics',
    'are annotations. Complete all training before the exclusive scoring marker;',
    'read held-out windows once, score each new model once, then make no further',
    'alternative or rerun. Selection supplies no end-to-end private guarantee.',
    'No campaign-wide composition guarantee is claimed.','',
    'HAR provenance: Anguita et al. (2013), *A Public Domain Dataset for Human',
    'Activity Recognition Using Smartphones*, ESANN; UCI dataset 240, DOI',
    '10.24432/C54S4K. Thesis citation keys: `anguita_har_2013` and `uci_har`.',
    'The existing official archive SHA-256 and all R3 evidence are preserved.',
    'No package code changes; reuse `pod-flower-sequence-2` and leave it running.','']
text='\n'.join(lines)
for path in (TOOLS.parent/'README.md',REPO/'inst/extdata/campaign/sequence/README.md'):
    path.write_text(text+'\n---\n\n'+path.read_text())
print(json.dumps(dict(schedule=s,summary=selection['selected']['summary']),indent=2))
