"""Channel-B scores for public held-out survival fixtures only."""
import math

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import t as student_t


def concordance(time, event, risk):
    """Strict earlier-event pairs; all time ties excluded, exact risk ties half."""
    time, event, risk = (np.asarray(value) for value in (time, event, risk))
    pairs = (time[:, None] < time[None, :]) & (event[:, None] == 1)
    denominator = int(pairs.sum())
    if not denominator:
        return None
    wins = (risk[:, None] > risk[None, :]) + 0.5 * (risk[:, None] == risk[None, :])
    return float(wins[pairs].sum() / denominator)


def public_targets(frame, config):
    time = np.asarray(frame['time'], dtype=float)
    event = np.asarray(frame['event'], dtype=float)
    valid = np.isfinite(time) & (time >= config['t_min']) & np.isin(event, [0, 1])
    time = np.where(valid, time, config['t_min'])
    event = np.where(valid & (time <= config['horizon']), event, 0)
    return np.minimum(time, config['horizon']), event, valid


def score(output, frame, config):
    """Independent NumPy/SciPy likelihood reference; no private metric release."""
    time, event, valid = public_targets(frame, config)
    output = np.asarray(output, dtype=float)
    if 'edges' not in config:
        mu = output.reshape(-1).clip(-10, 10)
        risk = -mu
        log_u = np.log(time / config['time_scale'])
        a = log_u - mu
        d = config['dispersion']
        if config['distribution'] == 'weibull':
            ch = np.exp(d * a)
            logf = math.log(d) - mu + (d-1)*a - ch
            logs = -ch
        else:
            z = a/d
            logf = -log_u - math.log(d) - 0.5*math.log(2*math.pi) - 0.5*z*z
            logs = log_ndtr(-z)
        nll = -event*(logf-math.log(config['time_scale'])) - (1-event)*logs
    else:
        edges = np.asarray(config['edges'])
        k = len(edges)-1
        assert output.shape == (len(time), k)
        index = np.searchsorted(edges[1:], time, side='left').clip(0, k-1)
        labels = np.zeros_like(output)
        labels[np.arange(len(time)), index] = event
        mask = np.where(event[:, None] == 1, np.arange(k)[None, :] <= index[:, None], edges[1:][None, :] <= time[:, None])
        nll = ((np.logaddexp(0, output)-labels*output)*mask).sum(1)/k
        logsurv = -np.logaddexp(0, output).cumsum(1)
        at_left = np.column_stack([np.ones(len(time)), np.exp(logsurv[:, :-1])])
        risk = -(at_left*np.diff(edges)).sum(1)
    return {
        'c_index': concordance(time[valid], event[valid], risk[valid]),
        'heldout_nll': float(nll[valid].mean()) if valid.any() else None,
        'n_evaluated_public_subjects': int(valid.sum()),
        'n_invalid_public_subjects': int((~valid).sum()),
    }


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('confidence intervals require >=2 finite executed replicates')
    sd = values.std(ddof=1)
    half = student_t.ppf(0.975, len(values)-1)*sd/math.sqrt(len(values))
    return {'n': len(values), 'mean': float(values.mean()), 'sd': float(sd),
            'ci95': [float(values.mean()-half), float(values.mean()+half)],
            'method': 'Student-t over matched public split replicates; overlapping splits'}


def envelopes(by_epsilon, primary=False, small_n=False):
    epsilons = sorted(by_epsilon)
    summaries = {}
    near = []
    for epsilon in epsilons:
        cells = by_epsilon[epsilon]
        gaps = [c['central']['c_index']-c['federated_dp']['c_index'] for c in cells]
        summaries[str(epsilon)] = {
            'federated': mean_ci([c['federated_dp']['c_index'] for c in cells]),
            'central': mean_ci([c['central']['c_index'] for c in cells]),
            'gap': mean_ci(gaps),
        }
        for cell, gap in zip(cells, gaps):
            condition = cell['central']['c_index'] < .95 and abs(gap) < .005
            near.append({'epsilon': epsilon, 'seed': cell['seed'],
                         'historic_flag': bool(cell['n_train']*epsilon < 2000 and condition),
                         'minimum_site_companion_flag': bool(cell['minimum_site_n']*epsilon < 2000 and condition)})
    adjacent = []
    for prev, curr in zip(epsilons, epsilons[1:]):
        p, c = summaries[str(prev)]['gap'], summaries[str(curr)]['gap']
        adjacent.append({'from': prev, 'to': curr, 'flag': c['mean']-p['mean'] > max(c['sd'], p['sd'])})
    floor = None
    if primary:
        cells = by_epsilon[8]
        u = summaries['8']['federated']['mean']
        null = np.mean([c['null']['c_index'] for c in cells])
        floor = {'pass': bool(u >= .60 and u >= null+.05), 'observed': u,
                 'absolute_floor': .60, 'null_margin': .05, 'null': float(null)}
    return {'gap_sign': 'G = central - federated (reversed from historic delta)',
            'summaries': summaries, 'epsilon_envelope': adjacent,
            'utility_floor': floor,
            'small_n_trend': None if not small_n else {'flag': summaries['8']['gap']['sd'] > summaries['1']['gap']['sd']},
            'near_central': near}
