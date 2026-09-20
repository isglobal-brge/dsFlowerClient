import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from run_hazard_v2 import GRID, SETTINGS, SEEDS, prepare_split, select, sha


class HazardV2Tests(unittest.TestCase):
    def test_selection_ties_and_missing_scores(self):
        scores = {c['id']: {f'{d}-{s}-{seed}': .6 for d, s in SETTINGS for seed in SEEDS} for c in GRID}
        self.assertEqual(select(scores)[0]['config']['id'], 'h01')
        scores['h06'] = {k: .7 for k in scores['h06']}
        self.assertEqual(select(scores)[0]['config']['id'], 'h06')
        scores['h06'].pop(next(iter(scores['h06'])))
        with self.assertRaises(ValueError):
            select(scores)
        scores['h06'] = dict(scores['h01'])
        scores['h06'][next(iter(scores['h06']))] = float('nan')
        with self.assertRaises(ValueError):
            select(scores)

    def test_development_never_reads_test_and_preserves_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'outer'
            source.mkdir()
            protocol = root/'protocol'
            protocol.write_text('frozen')
            frames, sites = [], []
            for site in range(1, 4):
                frame = pd.DataFrame({'subject_id': [f'{site}-{i}' for i in range(10)],
                                      'time': range(1, 11), 'event': [1]*10})
                frame.to_csv(source/f'site{site}.csv', index=False)
                frames.append(frame)
                sites.append(dict(site=site, split_sha256=sha(source/f'site{site}.csv')))
            pd.concat(frames).to_csv(source/'train.csv', index=False)
            (source/'split.json').write_text(json.dumps(dict(seed=1101, sites=sites,
                train_sha256=sha(source/'train.csv'), test_sha256='unopened', split_rule='outer')))
            # No outer test.csv exists: any attempted read must fail.
            prepare_split(source, root/'inner', protocol, True)
            a = {p.name: p.read_bytes() for p in (root/'inner').iterdir()}
            (source/'test.csv').write_text('poisoned outer outcomes and features')
            prepare_split(source, root/'inner2', protocol, True)
            self.assertEqual(a, {p.name: p.read_bytes() for p in (root/'inner2').iterdir()})
            for site in range(1, 4):
                part = pd.read_csv(root/'inner'/f'site{site}.csv')
                self.assertEqual(len(part), 8)
                self.assertTrue(all(x.startswith(f'{site}-') for x in part.subject_id))
            self.assertEqual(len(pd.read_csv(root/'inner/test.csv')), 6)

    def test_driver_runs_selection_before_thirty_confirmations(self):
        import run_hazard_v2 as driver
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'runtime').mkdir()
            secure_runs = root/'secure-runs'
            secure_runs.mkdir()
            (root/'runtime/runs').symlink_to(secure_runs, target_is_directory=True)
            tools = root/'dsFlowerClient/tools/campaign/survival'
            tools.mkdir(parents=True)
            (tools/'PROTOCOL_F_SURVIVAL.md').write_text('frozen')
            archive = root/'dsFlowerClient/inst/extdata/campaign/survival'
            archive.mkdir(parents=True)
            for variant in ('weibull', 'lognormal', 'hazard'):
                (archive/f'cell-synthetic-{variant}.json').write_text('{"status":"executed"}')
            calls = []

            def prepare(source, dest, protocol, development):
                if not development:
                    self.assertTrue((root/'runtime/hazard_v2/hazard_v2_selection.json').exists())
                dest.mkdir(parents=True)
                (dest/'split.json').write_text('{}')
                return dest

            def run(command, **kwargs):
                calls.append(command)
                if command[0] == 'Rscript':
                    out = Path(command[6])
                    out.mkdir(parents=True)
                    (out/'evidence.json').write_text(json.dumps(dict(status='executed',
                        results=dict(federated_dp=dict(c_index=.6)))))
                return SimpleNamespace(returncode=0)

            with patch('sys.argv', ['driver', tmp]), patch.object(driver, 'prepare_split', prepare), \
                 patch.object(driver.subprocess, 'run', run), \
                 patch.object(driver.subprocess, 'check_output', return_value='testcommit'), \
                 patch('validate_completion.validate_cell'):
                driver.main()
            training = [c for c in calls if c[0] == 'Rscript']
            self.assertEqual(len(training), 102)
            self.assertTrue((root/'runtime/hazard_v2/runs').is_symlink())
            self.assertTrue(all(Path(c[6]).parent == secure_runs.resolve()/'hazard_v2'
                                for c in training))
            self.assertEqual(sum('/development/' in c[3] for c in training), 72)
            self.assertEqual(sum('/confirmation/' in c[3] for c in training), 30)
            self.assertEqual(calls[-1][-1], '--hazard-v2')
            self.assertEqual(json.loads((root/'runtime/hazard_v2/hazard_v2_selection.json').read_text())['selected']['id'], 'h01')

    def test_hazard_summary_has_thirty_cell_scope(self):
        import summarize
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.argv', ['summarize', tmp, '--hazard-v2']):
                summarize.main()
            result = json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual(result['expected_matrix_cells'], 30)
            self.assertEqual(result['protocol_version'], 2)
            self.assertEqual(result['status'], 'incomplete')

    def test_release_summary_uses_selected_hazard_and_preserves_aft(self):
        import copy
        import summarize
        from test_validate_completion import complete_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'v1').mkdir()
            hazard = root/'hazard-v2'
            hazard.mkdir()
            selected = GRID[-1]
            selection = dict(selected=selected, rule='unit fixture', protocol_sha256='a'*64,
                             selected_utc='2026-01-01T00:00:00Z')
            (hazard/'hazard_v2_selection.json').write_text(json.dumps(selection))
            (hazard/'driver_complete.json').write_text(json.dumps(dict(status='executed',
                selection_sha256=sha(hazard/'hazard_v2_selection.json'))))
            for name, record in complete_fixture():
                (root/name).write_text(json.dumps(record))
                if name == 'summary.json':
                    original = copy.deepcopy(record)
                    (root/'v1/summary.json').write_text(json.dumps(record))
                elif record['variant'] == 'hazard' and record['dataset']['dataset'] != 'synthetic-public':
                    updated = copy.deepcopy(record)
                    updated.update(protocol_version=2, hazard_v2_config=selected)
                    updated['results']['federated_dp']['c_index'] = .7
                    target = hazard/('confirmation-'+name)/'evidence.json'
                    target.parent.mkdir()
                    target.write_text(json.dumps(updated))
            with patch('sys.argv', ['summarize', tmp, '--selected-hazard-v2']):
                summarize.main()
            current = json.loads((root/'summary.json').read_text())
            self.assertEqual(current['status'], 'executed')
            self.assertEqual(current['expected_matrix_cells'], 90)
            for group, previous in zip(current['groups'], original['groups']):
                if group['variant'] == 'hazard':
                    self.assertAlmostEqual(group['envelopes']['summaries']['8']['federated']['mean'], .7)
                else:
                    self.assertEqual(group, previous)
            self.assertEqual(json.loads((root/'v1/summary.json').read_text()), original)
            self.assertIn('v2 / h06', (root/'SURVIVAL_EVIDENCE_SUMMARY.md').read_text())
            target = next(hazard.glob('confirmation-*/evidence.json'))
            preserved = target.read_text()
            wrong = json.loads(preserved)
            wrong['hazard_v2_config'] = GRID[0]
            target.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, 'invalid or duplicate'):
                summarize.selected_hazard_records(root)
            target.unlink()
            with self.assertRaisesRegex(ValueError, 'all 30'):
                summarize.selected_hazard_records(root)
            target.write_text(preserved)
            (hazard/'hazard_v2_selection.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'completion does not match'):
                summarize.selected_hazard_records(root)


if __name__ == '__main__':
    unittest.main()
