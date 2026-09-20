import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from protocol_v4 import candidates, inner_split, select, SEEDS, active_config
from test_central_twins import fixture_config
from central_twins import load_twin_pins, validate_twin_pins
from dsflower_runner import segmentation
import base64


class ProtocolV4Tests(unittest.TestCase):
    def source(self, seed):
        sites = [[f'{i}-{j}' for j in range(20)] for i in range(3)]
        return dict(seed=seed, sites=sites, train=sum(sites, []), test=['outer'])

    def test_inner_split_retains_subjects_and_site_membership_excludes_outer(self):
        source = self.source(SEEDS[0])
        actual = inner_split(source)
        self.assertEqual(actual, inner_split(source))
        self.assertEqual(set(actual['train'] + actual['test']), set(source['train']))
        self.assertFalse(set(actual['train']) & set(actual['test']))
        self.assertNotIn('outer', actual['train'] + actual['test'])
        for old, new in zip(source['sites'], actual['sites']):
            self.assertTrue(set(new) < set(old))
            self.assertEqual(len(new), 16)

    def test_selection_all_seeds_ties_parameters_then_updates_and_failed_floor_continues(self):
        rows = [dict(candidate=c, seed=s, status='executed', dice=.1, foreground_dice=.1,
                     strongest_trivial=.3) for c in candidates() for s in SEEDS]
        splits = [inner_split(self.source(s)) for s in SEEDS]
        selected = select(rows, splits)['selected']
        self.assertEqual(selected['candidate'], dict(decoder='pointwise', parameters=129, batch_size=16, rounds=10))
        self.assertFalse(selected['floor_passed'])
        # Here sites are small enough that both batch arms have one update/epoch.
        for row in rows:
            if row['candidate']['decoder'] == 'current':
                row['dice'] = .9
        rows[0]['status'] = 'failed'
        result = select(rows, splits)
        self.assertNotIn(candidates()[0], [r['candidate'] for r in result['ranking']])
        self.assertEqual(result['selected']['mean_dice'], .9)
        with self.assertRaises(ValueError):
            select(rows[:2], splits)

    def test_config_exact_candidates_and_twins_reject_model_or_horizon_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.json'
            for candidate in candidates():
                path.write_text(json.dumps(candidate))
                with patch.dict(os.environ, F_SEG_V4_CONFIG=str(path)):
                    self.assertEqual(active_config(), candidate)
                    cfg = fixture_config()
                    cfg.update({'batch-size': candidate['batch_size'], 'num-server-rounds': candidate['rounds'],
                        'model-spec-b64': base64.b64encode(json.dumps(segmentation.decoder_spec(candidate['decoder'])).encode()).decode()})
                    validate_twin_pins(cfg, fixture_config(), load_twin_pins(cfg), candidate['batch_size'])
                    bad = copy.deepcopy(cfg)
                    bad['num-server-rounds'] = 7
                    with self.assertRaises(ValueError):
                        validate_twin_pins(bad, fixture_config(), load_twin_pins(bad), candidate['batch_size'])
            path.write_text(json.dumps(dict(decoder='pointwise', rounds=10, batch_size=16, parameters=130)))
            with patch.dict(os.environ, F_SEG_V4_CONFIG=str(path)), self.assertRaises(ValueError):
                active_config()

    def test_driver_completes_development_then_confirmation_despite_floor_and_one_cell_failure(self):
        import run_v4
        from assemble_evidence import planned_cells, sha256
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            protocol = run_v4.TOOLS.parents[2] / 'inst/extdata/campaign/segmentation/protocol-v4.md'
            (base / 'runtime-v4.json').write_text(json.dumps({'protocol_sha256': sha256(protocol)}))
            gates = base / 'gates.json'
            gates.write_text(json.dumps({f'segmentation_6_1_{i}': True for i in range(1, 8)}))
            prepared = base / 'prepared/busbra'
            prepared.mkdir(parents=True)
            for seed in SEEDS:
                (prepared / f'split-{seed}.json').write_text(json.dumps(self.source(seed)))
            calls = []
            def fake_run(command, env, log):
                calls.append(command)
                if len(calls) == 1:
                    raise subprocess.CalledProcessError(1, command)
                if 'run_matrix.py' in command[1]:
                    self.assertTrue((base / 'v4/selection.json').exists())
                    self.assertEqual(len(json.loads((base / 'v4/development-results.json').read_text())), 36)
            with patch.dict(os.environ, F_SEG_GATES_JSON=str(gates)), \
                 patch.object(run_v4, 'run', side_effect=fake_run), \
                 patch.object(run_v4, 'pinned_source_split', side_effect=lambda p, d, s, provenance: self.source(s)), \
                 patch.object(run_v4, 'released_artifact', return_value=base / 'model'), \
                 patch.object(run_v4, 'verify_development', return_value=dict(dice=.1, foreground_dice=.1, strongest_trivial=.3)), \
                 patch.object(run_v4, 'assemble', return_value={}):
                run_v4.study(base / 'v4')
            selection = json.loads((base / 'v4/selection.json').read_text())
            self.assertFalse(selection['selected']['floor_passed'])
            self.assertEqual(sum('run_matrix.py' in c[1] for c in calls), 2)
            self.assertEqual(len(planned_cells()) * 2, 66)
            self.assertTrue((base / 'v4/summary-v4.json').exists())

    def test_twenty_round_capture_horizon_is_exact(self):
        import assemble_evidence
        from test_assemble_evidence import captures
        records = captures()
        extra = copy.deepcopy(records)
        for row in extra:
            row['round'] += 10
        records += extra
        for row in records:
            row['mechanism']['total_epochs'] = 60
            row['mechanism']['total_steps'] *= 2
        with patch.object(assemble_evidence, 'active_config', return_value=dict(rounds=20)), \
             patch.object(assemble_evidence, 'independent_accounting', return_value={}):
            self.assertEqual(len(assemble_evidence.validate_captures(records, [69, 68, 68], 8)), 3)
            with self.assertRaises(ValueError):
                assemble_evidence.validate_captures(records[:30], [69, 68, 68], 8)
            records[0]['mechanism']['total_epochs'] = 30
            with self.assertRaises(ValueError):
                assemble_evidence.validate_captures(records, [69, 68, 68], 8)
