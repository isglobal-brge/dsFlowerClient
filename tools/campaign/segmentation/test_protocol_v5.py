import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from dsflower_runner import params, segmentation, server_app, client_app, dp_harness
from feature_smoke import config
from protocol_v5 import candidates, mechanism_candidate, select, floor, SEEDS
from benchmark_hooks.public_initialization import apply, load_arrays, verify_capture

TOOLS = Path(__file__).resolve().parent


class ProtocolV5Tests(unittest.TestCase):
    def fixture(self, root):
        cfg = config()
        cfg['model-spec-b64'] = base64.b64encode(json.dumps(segmentation.decoder_spec('narrow')).encode()).decode()
        model = params.load_user_model(cfg, segmentation.FEATURE_DIM, 'segmentation_bce_dice')
        arrays = [np.full(tuple(p.shape), (i+1)/100, np.float32) for i, p in enumerate(model.parameters())]
        checkpoint = root / 'public.npz'
        np.savez(checkpoint, **{str(i): a for i, a in enumerate(arrays)})
        binding = dict(seed=SEEDS[0], decoder='narrow', dataset='BUSI', privacy='public_nonprivate',
                       encoder_sha256=segmentation.CHECKPOINT_SHA256, checkpoint=str(checkpoint),
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       tensor_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays])
        return cfg, model, arrays, binding

    def test_grid_is_eight_v4_mechanisms_with_two_public_prefixes(self):
        from protocol_v4 import candidates as v4_candidates
        self.assertEqual(len(candidates()), 8)
        for candidate in candidates():
            self.assertIn(mechanism_candidate(candidate), v4_candidates())
        with self.assertRaises(ValueError):
            mechanism_candidate(dict(candidates()[0], rounds=1))

    def test_selection_uses_foreground_and_complete_seeds_not_all_subject_score(self):
        rows = [dict(candidate=c, seed=s, status='executed', dice=.8, foreground_dice=.1,
                     strongest_trivial=.3, foreground_strongest_trivial=.2)
                for c in candidates() for s in SEEDS]
        splits = [dict(sites=[list(range(228))]*3)]*3
        for row in rows:
            if row['candidate'] == candidates()[-1]:
                row.update(dice=.2, foreground_dice=.4)
        selected = select(rows, splits)['selected']
        self.assertEqual(selected['candidate'], candidates()[-1])
        self.assertFalse(selected['floor_passed'])
        rows[-1]['status'] = 'failed'
        self.assertNotEqual(select(rows, splits)['selected']['candidate'], candidates()[-1])
        with self.assertRaises(ValueError):
            select(rows[:2], splits)

    def test_both_floors_and_margins_are_required(self):
        good = dict(dice=.6, foreground_dice=.6, strongest_trivial=.49, foreground_strongest_trivial=.49)
        self.assertTrue(floor(good))
        for key, value in [('dice', .49), ('foreground_dice', .49),
                           ('strongest_trivial', .51), ('foreground_strongest_trivial', .51)]:
            self.assertFalse(floor(dict(good, **{key: value})))

    def test_binding_rejects_seed_hash_shape_and_nonfinite_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, model, arrays, binding = self.fixture(root)
            apply(model, cfg, binding, SEEDS[0])
            for actual, expected in zip(params.get_torch_params(model), arrays):
                np.testing.assert_array_equal(actual, expected)
            with self.assertRaises(ValueError):
                apply(model, cfg, binding, SEEDS[1])
            with self.assertRaises(ValueError):
                load_arrays(dict(binding, checkpoint_sha256='0'*64))
            for broken in (np.zeros((1,), np.float32), np.full(arrays[0].shape, np.nan, np.float32)):
                values = [broken] + arrays[1:]
                path = Path(binding['checkpoint'])
                np.savez(path, **{str(i): a for i, a in enumerate(values)})
                changed = dict(binding, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               tensor_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in values])
                with self.assertRaises(ValueError):
                    apply(model, cfg, changed, SEEDS[0])

    def test_flower_record_and_capture_contain_pretrained_arrays_not_random_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _, arrays, binding = self.fixture(root)
            capture = root / 'public-capture'
            (root / 'public-pretraining-binding.json').write_text(json.dumps(binding))
            module_path = TOOLS / 'benchmark_hooks/sitecustomize.py'
            spec = importlib.util.spec_from_file_location('public_hook_v5_test', module_path)
            hook = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hook)
            fake_server = SimpleNamespace(_initial_arrays=server_app._initial_arrays)
            with patch.dict(os.environ, F_SEG_CAPTURE_DIR=str(capture), F_SEG_INIT_SEED=str(SEEDS[0]),
                            F_SEG_V5_BINDINGS=str(root)), \
                 patch.object(client_app, '_dp_fit', client_app._dp_fit), \
                 patch.object(dp_harness, 'make_private_dpsgd', dp_harness.make_private_dpsgd), \
                 patch.dict(__import__('sys').modules, {'public_initialization': __import__(
                     'benchmark_hooks.public_initialization', fromlist=['apply'])}):
                hook._attach(fake_server)
                model, record = fake_server._initial_arrays(cfg, 'neural')
                for expected, sent, live in zip(arrays, record.to_numpy_ndarrays(), params.get_torch_params(model)):
                    np.testing.assert_array_equal(sent, expected)
                    np.testing.assert_array_equal(live, expected)
                verify_capture(root, binding)
                with self.assertRaises(ValueError):
                    verify_capture(root, dict(binding, seed=SEEDS[1]))

    def test_confirmation_floor_refuses_incomplete_or_invalid_twins(self):
        import run_v5
        with patch.object(run_v5, 'load_replicate', side_effect=ValueError('twin mismatch')):
            actual = run_v5.confirmation_floor(Path('/unused'), 16)
        self.assertEqual(actual['verdict'], 'FAIL')
        self.assertIsNone(actual['numerical_floor_pass'])
        self.assertEqual(len(actual['errors']), 3)

    def test_driver_retains_failure_finishes_grid_and_confirms_even_when_inner_floor_fails(self):
        import run_v5
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / 'v5'
            (base / 'runtime-v4.json').write_text(json.dumps({
                'protocol_sha256': hashlib.sha256((run_v5.EVIDENCE / 'protocol-v4.md').read_bytes()).hexdigest(),
                'runner_files': {}}))
            gates = base / 'gates.json'
            gates.write_text(json.dumps({f'segmentation_6_1_{i}': True for i in range(1, 8)}))
            prepared = base / 'prepared/busbra'
            prepared.mkdir(parents=True)
            sources = {}
            for seed in SEEDS:
                sites = [[f'{i}-{j}' for j in range(20)] for i in range(3)]
                sources[seed] = dict(seed=seed, train=sum(sites, []), sites=sites, test=['outer'])
                (prepared / f'split-{seed}.json').write_text(json.dumps(sources[seed]))
            calls = []
            def fake_run(command, env, log):
                calls.append(command)
                if any(str(c).endswith('pretrain_public_v5.py') for c in command):
                    for epochs in (20, 60):
                        out = root / f'public-pretraining/epochs{epochs}'
                        out.mkdir(parents=True)
                        for seed in SEEDS:
                            (out / f'seed{seed}.json').write_text(json.dumps(dict(seed=seed)))
                    (root / 'public-pretraining/audit.json').write_text('{}')
                elif command[0].endswith('run_federated.sh') and 'development' in command[-1]:
                    if sum(c[0].endswith('run_federated.sh') and 'development' in c[-1] for c in calls) == 1:
                        raise subprocess.CalledProcessError(1, command)
                elif any(str(c).endswith('score_public.py') for c in command):
                    Path(command[-1]).write_text(json.dumps(dict(trivial=dict(scores={
                        'empty': dict(foreground_positive=dict(dice=0.)),
                        'full': dict(foreground_positive=dict(dice=.2))}))))
                elif any(str(c).endswith('run_matrix.py') for c in command):
                    self.assertTrue((root / 'selection.json').exists())
                    self.assertEqual(len(json.loads((root / 'development-results.json').read_text())), 24)
            with patch.dict(os.environ, F_SEG_GATES_JSON=str(gates)), \
                 patch.object(run_v5, 'check_registration', return_value={'files': {},
                     'retained_runtime_sha256': hashlib.sha256((base / 'runtime-v4.json').read_bytes()).hexdigest()}), \
                 patch.object(run_v5, 'sha256', side_effect=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              if Path(p).exists() else '0'*64), \
                 patch.object(run_v5, 'pinned_source_split', side_effect=lambda p, d, s, provenance: sources[s]), \
                 patch.object(run_v5, 'run', side_effect=fake_run), \
                 patch.object(run_v5, 'verify_capture'), \
                 patch.object(run_v5, 'released_artifact', return_value=base / 'model'), \
                 patch.object(run_v5, 'verify_development', return_value=dict(dice=.1, foreground_dice=.1, strongest_trivial=.3)), \
                 patch.object(run_v5, 'assemble', return_value={}), \
                 patch.object(run_v5, 'confirmation_floor', return_value=dict(verdict='FAIL', numerical_floor_pass=False)):
                run_v5.study(root)
            self.assertEqual(sum(any(str(c).endswith('run_matrix.py') for c in cmd) for cmd in calls), 2)
            rows = json.loads((root / 'development-results.json').read_text())
            self.assertEqual(sum(r['status'] == 'failed' for r in rows), 1)
            self.assertIn('**FAIL**', (root / 'STATUS_R2.md').read_text())


if __name__ == '__main__':
    unittest.main()
