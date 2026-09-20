"""Preregistered public-development choices; never imported by the runner."""
import hashlib
import json
import math
import os
from pathlib import Path

SEEDS = (20260919, 20260920, 20260921)
PARAMETERS = {'current': 41537, 'narrow': 9521, 'pointwise': 129}


def candidates():
    return [dict(decoder=d, batch_size=b, rounds=r, parameters=n)
            for d, n in PARAMETERS.items() for b in (16, 64) for r in (10, 20)]


def active_config():
    path = os.environ.get('F_SEG_V4_CONFIG')
    if not path:
        return dict(decoder='current', rounds=10)
    value = json.loads(Path(path).read_text())
    if value not in candidates():
        raise ValueError('configuration is not a preregistered v4 candidate')
    return value


def inner_split(source):
    sites, validation = [], []
    for site in source['sites']:
        ordered = sorted(site, key=lambda s: (hashlib.sha256(
            f"inner-v4|{source['seed']}|{s}".encode()).hexdigest(), s))
        size = len(ordered) // 5
        validation.extend(ordered[:size])
        sites.append(ordered[size:])
    train = sum(sites, [])
    assert len(set(train + validation)) == len(source['train'])
    assert set(train + validation) == set(source['train'])
    assert not set(train + validation) & set(source['test'])
    return dict(seed=source['seed'], train=train, test=validation, sites=sites, variant='full')


def select(results, splits):
    ranked = []
    for candidate in candidates():
        rows = [r for r in results if r['candidate'] == candidate]
        if len(rows) != 3 or {r['seed'] for r in rows} != set(SEEDS):
            continue
        if any(r['status'] != 'executed' for r in rows):
            continue
        score = sum(r['dice'] for r in rows) / 3
        trivial = sum(r['strongest_trivial'] for r in rows) / 3
        updates = sum(math.ceil(len(site) / candidate['batch_size']) * 3 * candidate['rounds']
                      for split in splits for site in split['sites'])
        ranked.append(dict(candidate=candidate, mean_dice=score,
                           mean_foreground_dice=sum(r['foreground_dice'] for r in rows) / 3,
                           strongest_trivial=trivial, logical_site_updates=updates,
                           floor_passed=score >= .50 and score >= trivial + .10))
    if not ranked:
        raise ValueError('no candidate has three validated development releases')
    ranked.sort(key=lambda r: (-r['mean_dice'], r['candidate']['parameters'],
                              r['logical_site_updates'], r['candidate']['decoder'],
                              r['candidate']['batch_size']))
    return dict(selected=ranked[0], ranking=ranked,
                confirmation='66 cells: selected batch primary; other batch sensitivity',
                promotion='vetted remains FALSE; reviewer decides')
