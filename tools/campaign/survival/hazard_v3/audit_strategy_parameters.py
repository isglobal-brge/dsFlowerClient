#!/usr/bin/env python3
"""Check pooled strategy parameters against the installed R API before confirmation."""
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1])
archive=root/'dsFlowerClient/inst/extdata/campaign/survival/hazard-v3'
assert not (archive/'confirmation_started.json').exists()
path=Path(__file__).with_name('central_and_score.py')
spec=importlib.util.spec_from_file_location('hazard_v3_twins',path)
twins=importlib.util.module_from_spec(spec);spec.loader.exec_module(twins)
expression='''
library(dsFlowerClient)
rows <- list()
for (name in c('fedavg','fedavgm','fedadam','fedyogi')) for (lr in c(.02,.05,.1)) {
  strategy <- if (name=='fedavgm') ds.flower.strategy(name,server_momentum=.9) else ds.flower.strategy(name)
  rows[[length(rows)+1L]] <- dsFlowerClient:::.strategy_config_values(strategy,client_learning_rate=lr)
  rows[[length(rows)]][['learning-rate']] <- lr
}
cat(jsonlite::toJSON(rows,auto_unbox=TRUE,digits=NA))
'''
env=dict(os.environ,R_LIBS_USER=str(root/'runtime/rlib'))
rows=json.loads(subprocess.check_output(['Rscript','-e',expression],env=env,text=True))
keys={'strategy-eta':'eta','strategy-eta-l':'eta_l','strategy-beta-1':'beta_1',
    'strategy-beta-2':'beta_2','strategy-tau':'tau','strategy-server-learning-rate':'server_learning_rate',
    'strategy-server-momentum':'server_momentum'}
for cfg in rows:
    strategy=twins.aggregation(cfg)
    for key,attribute in keys.items():
        if key in cfg:
            assert getattr(strategy,attribute)==cfg[key],(cfg['strategy'],key)
result=dict(status='verified',verified_utc=datetime.now(timezone.utc).isoformat(),
    source='Installed dsFlowerClient 0.5.0 .strategy_config_values versus installed Flower objects used by pooled twins',
    checked_configurations=rows,
    repair='Before confirmation, pooled adaptive eta_l was changed from Flower constructor defaults to the local learning rate, matching the R API. No completed development fit or score used pooled aggregation; the synthetic pilot used FedAvgM. No fit was repeated.',
    twins_sha256=twins.sha(path))
with (archive/'strategy_parameter_audit.json').open('x') as handle:
    json.dump(result,handle,indent=2)
print('STRATEGY_PARAMETERS_VERIFIED',len(rows))
