"""Guard the holdout boundary and pooled twin semantics before cohort fitting."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import protocol
import importlib.util
spec=importlib.util.spec_from_file_location('hazard_twins',Path(__file__).with_name('central_and_score.py'))
twins=importlib.util.module_from_spec(spec)
spec.loader.exec_module(twins)


def config(optimizer='sgd',strategy='fedavg'):
    b64=lambda x:base64.b64encode(json.dumps(x).encode()).decode()
    cfg={'num-features':2,'num-classes':2,'num-labels':2,'loss-name':'discrete_hazard_nll',
         'num-server-rounds':2,'local-epochs':2,'batch-size':4,'learning-rate':.05,
         'optimizer-name':optimizer,'optimizer-momentum':0.,'optimizer-nesterov':False,
         'optimizer-beta1':.9,'optimizer-beta2':.999,'optimizer-eps':1e-8,'optimizer-amsgrad':False,
         'strategy':strategy,'model-spec-b64':b64({'kind':'sequential','layers':[{'op':'linear','out':'@out'}]}),
         'survival-config-b64':b64(dict(schema_version=1,time_unit='days',time_origin='baseline',t_min=1,horizon=1825,edges=[0,30,1825]))}
    return cfg


class ProtocolTests(unittest.TestCase):
    def test_inner_never_opens_outer_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'outer';source.mkdir()
            protocol_file=root/'protocol';protocol_file.write_text('frozen')
            parts=[pd.DataFrame(dict(subject_id=[f'{i}-{j}' for j in range(10)],time=np.arange(1,11),event=np.ones(10))) for i in range(3)]
            for i,part in enumerate(parts,1):part.to_csv(source/f'site{i}.csv',index=False)
            meta=dict(seed=1101,train_sha256='outer-train',sites=[dict(site=i,split_sha256=protocol.sha(source/f'site{i}.csv')) for i in range(1,4)])
            (source/'split.json').write_text(json.dumps(meta))
            (source/'test.csv').write_text('POISON: must never open')
            real_open=Path.open
            def guarded(path,*a,**kw):
                if path==source/'test.csv':raise AssertionError('outer test read')
                return real_open(path,*a,**kw)
            with patch.object(Path,'open',guarded):
                protocol.prepare_inner(source,root/'inner',protocol_file)
            train=pd.read_csv(root/'inner/train.csv');valid=pd.read_csv(root/'inner/test.csv')
            self.assertEqual((len(train),len(valid)),(24,6))
            self.assertFalse(set(train.subject_id)&set(valid.subject_id))

    def test_grid_ignores_validation_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            pd.DataFrame(dict(time=np.arange(1,101),event=np.ones(100))).to_csv(path/'train.csv',index=False)
            (path/'test.csv').write_text('POISON')
            cfg=protocol.grid()[1]
            first=protocol.config_for_seed(cfg,path)
            (path/'test.csv').unlink()
            self.assertEqual(first,protocol.config_for_seed(cfg,path))
            self.assertEqual(first['edges'],[0.,20.8,40.6,60.4,80.2,1825.])

    def test_selection_requires_both_finite_seeds(self):
        cfg=protocol.grid()[0]
        for scores in ({'1101':.6},{'1101':.6,'1102':float('nan')}):
            with self.assertRaises(AssertionError):protocol.select([dict(config=cfg,scores=scores)])
        self.assertEqual(len(protocol.grid()),35)

    def test_nonprivate_fedavg_matches_canonical_twin(self):
        torch.set_num_threads(1)
        cfg=config()
        x=np.random.default_rng(3).uniform(-1,1,(16,2)).astype(np.float32)
        conf=twins.survival.config_from_run(cfg,cfg['loss-name'])
        y=twins.survival.period_targets(np.arange(1,17)*20,np.arange(16)%2,np.ones(16),conf)
        first,_=twins.original.train(cfg,x,y,8,1101,False)
        second,_=twins.train(cfg,x,y,8,1101,False)
        for a,b in zip(first.parameters(),second.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_all_public_server_strategies_accept_pooled_endpoint(self):
        for name in ('fedavg','fedavgm','fedadam','fedyogi'):
            cfg=config(strategy=name);model=twins.build(cfg);strategy=twins.aggregation(cfg)
            old=[x.copy() for x in twins.params.get_torch_params(model)]
            endpoint=[x+.2 for x in old]
            first=twins.aggregate(strategy,model,endpoint,1)
            self.assertTrue(all(np.isfinite(x).all() for x in first))
            if name in ('fedavg','fedavgm'):
                for a,b in zip(first,endpoint):np.testing.assert_allclose(a,b,rtol=0,atol=1e-7)
            twins.params.set_torch_params(model,first)
            second=twins.aggregate(strategy,model,[x+.2 for x in first],2)
            if name=='fedavgm':
                for a,b in zip(second,old):np.testing.assert_allclose(a,b+.58,rtol=0,atol=1e-6)


if __name__=='__main__': unittest.main()
