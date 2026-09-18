import json
from pathlib import Path
import shutil
import tempfile
import unittest
import jsonschema
from assemble_evidence import planned_cells, sha256
from combine_evidence import combine

ARCHIVE = Path(__file__).resolve().parents[3] / 'inst/extdata/campaign/segmentation'


class CombinedArmTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root/'protocol.md').write_text('Synthetic unexecuted plan fixture; no scores.\n')
        shutil.copy(ARCHIVE/'campaign-summary-schema.json',self.root/'campaign-summary-schema.json')
        for batch in (16,64):
            path=self.root/f'batch{batch}/campaign-status.json';path.parent.mkdir()
            cells=[dict(zip(('dataset','variant','epsilon','seed'),c),status='not_executed',reason='fixture') for c in planned_cells()]
            value=dict(schema='dsflower-segmentation-campaign-summary-v1',status='not_executed',contract='pytorch_resnet18_segmentation',task='segmentation',protocol_sha256=sha256(self.root/'protocol.md'),nominal_batch_size=batch,cells=cells,promotion='fixture')
            path.write_text(json.dumps(value))

    def test_keeps_both_complete_plans_without_scores(self):
        value=combine(self.root)
        jsonschema.validate(value,json.loads((ARCHIVE/'campaign-combined-schema.json').read_text()))
        self.assertEqual(len(value['cells']),2*len(planned_cells()))
        self.assertEqual(value['status'],'not_executed')
        self.assertEqual({c['nominal_batch_size'] for c in value['cells']},{16,64})
        self.assertNotIn('scores',value)

    def test_rejects_wrong_arm_identity(self):
        path=self.root/'batch64/campaign-status.json';value=json.loads(path.read_text());value['nominal_batch_size']=16;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'identity'):
            combine(self.root)

    def test_rejects_status_that_hides_failed_cells(self):
        path=self.root/'batch64/campaign-status.json';value=json.loads(path.read_text());value['cells'][0]['status']='failed';path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'contradicts'):
            combine(self.root)
