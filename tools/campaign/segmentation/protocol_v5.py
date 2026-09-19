"""Frozen Remedy 2 grid; v4 mechanism configurations are reused verbatim."""
import math
from protocol_v4 import SEEDS, inner_split


def candidates():
    return [dict(decoder='narrow', parameters=9521, batch_size=b, rounds=r,
                 pretraining_epochs=e)
            for e in (20, 60) for b in (16, 64) for r in (10, 20)]


def mechanism_candidate(candidate):
    if candidate not in candidates():
        raise ValueError('not a preregistered v5 candidate')
    return {k: v for k, v in candidate.items() if k != 'pretraining_epochs'}


def select(results, splits):
    ranked = []
    for candidate in candidates():
        rows = [r for r in results if r['candidate'] == candidate]
        if (len(rows) != 3 or {r['seed'] for r in rows} != set(SEEDS)
                or any(r['status'] != 'executed' for r in rows)):
            continue
        mean = lambda key: sum(r[key] for r in rows) / 3
        values = {k: mean(k) for k in ('dice', 'foreground_dice', 'strongest_trivial',
                                      'foreground_strongest_trivial')}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('nonfinite development result')
        updates = sum(math.ceil(len(site) / candidate['batch_size']) * 3 * candidate['rounds']
                      for split in splits for site in split['sites'])
        ranked.append(dict(candidate=candidate, **values, logical_site_updates=updates,
                           floor_passed=floor(values)))
    if not ranked:
        raise ValueError('no candidate has all three validated development releases')
    ranked.sort(key=lambda r: (-r['foreground_dice'], r['candidate']['parameters'],
                              r['logical_site_updates'], r['candidate']['pretraining_epochs'],
                              r['candidate']['decoder'], r['candidate']['batch_size']))
    return dict(selected=ranked[0], ranking=ranked,
                selection_metric='mean federated-DP inner-validation foreground-positive Dice',
                confirmation='66 cells; selected batch primary, other batch sensitivity; no pooling',
                promotion='vetted remains FALSE; reviewer decides')


def floor(values):
    # Preserve v4's all-subject floor and also enforce the requested foreground floor.
    return (values['dice'] >= .50 and values['dice'] >= values['strongest_trivial'] + .10
            and values['foreground_dice'] >= .50
            and values['foreground_dice'] >= values['foreground_strongest_trivial'] + .10)
