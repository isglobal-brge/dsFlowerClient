#!/usr/bin/env python3
"""Freeze public subject splits and fixed baseline encodings before scoring."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    'support2': {
        'url': 'https://archive.ics.uci.edu/static/public/880/data.csv',
        'sha256': '9da794bbd5c3a6a816e677cc17535e58c122d9ef4cbefd404489330a9f9cd2de',
        'licence': 'CC BY 4.0',
        'licence_url': 'https://archive-beta.ics.uci.edu/dataset/880/support2',
        'release': 'UCI 880 metadata last_updated 2024-09-09',
        'attribution': 'Frank Harrell and SUPPORT investigators (1995)',
    },
    'lung1': {
        'url': 'https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv',
        'sha256': '132f72b58b9660bf5e6b24b9817b335f1896360bc253e1f2034a3ffee593e6fd',
        'licence': 'CC BY-NC 3.0',
        'licence_url': 'https://www.cancerimagingarchive.net/collection/nsclc-radiomics/',
        'release': 'Clinical version 3 October 2019; collection version 4 2020-10-22',
        'attribution': 'Aerts et al. (2014), DOI 10.7937/K9/TCIA.2015.PF0M9REI',
    },
}
EDGES = [0, 7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 540, 730, 1095, 1460, 1825]
SEEDS = [1101, 1102, 1103]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(raw, name):
    frame = pd.DataFrame(index=raw.index)
    bounds = {}

    def numeric(source, target, lower, upper):
        values = pd.to_numeric(raw[source], errors='coerce')
        frame[target] = values.replace([np.inf, -np.inf], np.nan).fillna((lower + upper) / 2).clip(lower, upper)
        bounds[target] = [lower, upper]

    def indicator(source, target, value):
        frame[target] = (raw[source].fillna('').astype(str) == value).astype(int)
        bounds[target] = [0, 1]

    numeric('age', 'age', 0, 100)
    if name == 'support2':
        indicator('sex', 'male', 'male')
        numeric('num.co', 'comorbidities', 0, 10)
        numeric('diabetes', 'diabetes', 0, 1)
        numeric('dementia', 'dementia', 0, 1)
        for i, value in enumerate(['yes', 'metastatic']):
            indicator('ca', f'cancer_{i}', value)
        for i, value in enumerate(['ARF/MOSF w/Sepsis', 'CHF', 'COPD', 'Cirrhosis', 'Colon Cancer', 'Coma', 'Lung Cancer', 'MOSF w/Malig']):
            indicator('dzgroup', f'disease_{i}', value)
        id_col, time_col, event_col = 'id', 'd.time', 'death'
    else:
        indicator('gender', 'male', 'male')
        for source, target, upper in [('clinical.T.Stage', 't_stage', 4), ('Clinical.N.Stage', 'n_stage', 3), ('Clinical.M.Stage', 'm_stage', 1)]:
            numeric(source, target, 0, upper)
        for i, value in enumerate(['large cell', 'squamous cell carcinoma', 'adenocarcinoma', 'nos']):
            indicator('Histology', f'histology_{i}', value)
        id_col, time_col, event_col = 'PatientID', 'Survival.time', 'deadstatus.event'
    frame.insert(0, 'subject_id', raw[id_col].astype(str))
    frame['time'] = pd.to_numeric(raw[time_col], errors='coerce')
    frame['event'] = pd.to_numeric(raw[event_col], errors='coerce')
    assert len(frame) == len(raw) and frame.subject_id.is_unique
    return frame, bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        path = args.data / f'{name}.csv'
        assert sha(path) == source['sha256'], 'public release checksum differs'
        frame, bounds = encode(pd.read_csv(path), name)
        for seed in SEEDS:
            keys = frame.subject_id.map(lambda subject: hashlib.sha256(f'{seed}:{subject}'.encode()).hexdigest())
            ordered = frame.loc[keys.sort_values().index].reset_index(drop=True)
            cut = int(np.floor(0.8 * len(frame)))
            train, test = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
            for subset in (['full', 'small600', 'heterogeneous'] if name == 'support2' else ['full']):
                dest = args.out / f'{name}-{subset}-{seed}'
                dest.mkdir(exist_ok=True)
                selected = train.iloc[:600].copy() if subset == 'small600' else train.copy()
                if subset == 'heterogeneous':
                    selected = selected.sort_values(['age', 'subject_id']).reset_index(drop=True)
                    groups = np.minimum(2, np.arange(len(selected)) * 3 // len(selected))
                else:
                    groups = np.arange(len(selected)) % 3
                assert not set(selected.subject_id) & set(test.subject_id)
                selected.to_csv(dest / 'train.csv', index=False)
                test.to_csv(dest / 'test.csv', index=False)
                sites = []
                for site in range(3):
                    part = selected.iloc[np.flatnonzero(groups == site)]
                    part.to_csv(dest / f'site{site + 1}.csv', index=False)
                    sites.append({'site': site + 1, 'source_rows': len(part), 'n_subjects': part.subject_id.nunique(), 'split_sha256': sha(dest / f'site{site + 1}.csv')})
                manifest = {
                    'schema_version': 1, 'dataset': name, 'source': source,
                    'protocol_sha256': sha(args.protocol), 'seed': seed, 'subset': subset,
                    'public_fixture': True, 'subject_provenance': 'One release row per stable public subject ID; no duplicate IDs',
                    'n_original_rows': len(frame), 'n_train': len(selected), 'n_test': len(test),
                    'split_rule': 'SHA256(seed:id), floor(0.8*N) train, no outcome stratification',
                    'train_sha256': sha(dest / 'train.csv'), 'test_sha256': sha(dest / 'test.csv'),
                    'sites': sites, 'features': list(bounds),
                    'feature_bounds': {'lower': [v[0] for v in bounds.values()], 'upper': [v[1] for v in bounds.values()]},
                    'time_origin': 'study_entry' if name == 'support2' else 'treatment_start',
                }
                (dest / 'split.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('FROZEN 12 dataset/split/subset manifests; no scores computed')


if __name__ == '__main__':
    main()
