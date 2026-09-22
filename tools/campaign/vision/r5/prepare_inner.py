#!/usr/bin/env python3
"""Re-extract only the already staged R4 inner-training patients for exact parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import emulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r4.diagnose import controls, extract


def main(root):
    # R4 controls and extractor preserve its image batch composition and patient
    # summation order; reordering an outer-cohort feature cache is insufficient.
    controls()
    data = emulate.load_data(root)
    staged = Path('/tmp/cells-vision-r4/inner-training')
    assert staged.is_dir()
    xs, ys, ids, _, hashes = extract(staged, data['cfg'])
    assert [x.shape for x in xs] == [(227, 512)] * 3
    assert len(ids) == len(set(ids)) == 681
    expected = {(row['features_sha256'], row['targets_sha256']) for row in hashes}
    previous = root/'r4/diagnosis/federations/adam_lr0.003_e20_b32-startup-retry/public-capture'
    captures = [json.loads(path.read_text()) for path in previous.glob('accountant-*.json')]
    assert len(captures) == 15
    observed = {(row['features_sha256'], row['targets_sha256']) for row in captures}
    assert expected == observed, 'Fresh R4 inner extraction differs from the actual private runner tensors'
    differences = []
    for index, (x, y, outer_x, outer_y) in enumerate(zip(xs, ys, data['xs'], data['ys']), 1):
        assert np.array_equal(y, outer_y)
        differences.append(dict(site=index,
            feature_max_abs_difference=float(np.max(np.abs(x-outer_x))),
            changed_coordinates=int(np.count_nonzero(x != outer_x)),
            total_coordinates=int(x.size),
            targets_identical=True))
    out = root/'r5'
    out.mkdir(exist_ok=True)
    destination = out/'inner-features.npz'
    assert not destination.exists(), 'Preserve the first validated inner cache'
    np.savez(destination, **{f'x{i}': x for i, x in enumerate(xs)},
        **{f'y{i}': y for i, y in enumerate(ys)})
    result = dict(passed=True, test_accessed=False, source=str(staged),
        n_inner_training_patients=681, site_sizes=[227]*3,
        feature_parity_with_all_r4_captures=True, feature_hashes=hashes,
        cache_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        outer_cache_comparison=differences,
        validation_features='Unchanged R4 outer training cache on its 171 inner-validation indices.',
        interpretation='CUDA frozen-backbone features depend numerically on image batch composition. Re-extraction uses the exact R4 staged inner manifests; no representation or preprocessing change.')
    emulate.save(out/'inner-feature-parity.json', result)
    print('INNER_FEATURE_PARITY', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
