#!/usr/bin/env python3
"""Quantify historical replicate overlap without opening any outer test file."""
import csv
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
base=root/'runtime/hazard_v3'
archive=root/'dsFlowerClient/inst/extdata/campaign/survival/hazard-v3'
seeds=(1101,1102,1103)
train={}
for seed in seeds:
    with (base/'outer'/f'support2-full-{seed}'/'train.csv').open() as handle:
        train[seed]={row['subject_id'] for row in csv.DictReader(handle)}
    assert len(train[seed])==7284
rows=[]
for development_seed in (1101,1102):
    for confirmation_seed in seeds:
        rows.append(dict(development_seed=development_seed,confirmation_seed=confirmation_seed,
            development_outer_training_subjects_outside_confirmation_training=
                len(train[development_seed]-train[confirmation_seed])))
result=dict(method='Set differences of outer training subject IDs only; no test file, event, time, prediction or score read.',
    interpretation='The same 9105 public subjects underlie all historical splits. IDs outside another replicate training set belong to its test set. Development training/validation is disjoint from its own replicate test, but not globally disjoint from other replicate tests. These are descriptive reused-holdout benchmark results, not independent nested validation.',
    rows=rows)
with (archive/'split_overlap.json').open('x') as handle:
    json.dump(result,handle,indent=2)
print(json.dumps(result,indent=2))
