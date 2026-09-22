"""Record the unchanged runner's full-horizon calibration for the selected schedule."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/cells/dsFlower/inst/flower_app')
from dsflower_runner.dp_harness import effective_dpsgd_mechanism
here=Path(__file__).resolve().parent
params=json.loads((here/'selection.json').read_text())['selected_params']
rows=[]
for n in [9600,12000,36000]:
    for eps in ([8] if n==9600 else [1,4,8]):
        result=effective_dpsgd_mechanism(eps,1e-6,1,n,params['batch_size'],params['local_epochs'],5)
        steps=math.ceil(n/params['batch_size'])
        rows.append(dict(n_units=n,epsilon=eps,delta=1e-6,steps_per_epoch=steps,
            total_steps=5*params['local_epochs']*steps,expected_batch_size=max(1,int(n/steps)),
            mechanism=result))
out=Path('/workspace/cells/r4/corrected_evidence')
out.mkdir(exist_ok=True)
(out/'cdcbmi_r4_accounting.json').write_text(json.dumps(dict(selected_params=params,
    source='unchanged released effective_dpsgd_mechanism; complete five-round horizon',
    calibrations=rows),indent=2)+'\n')
print(json.dumps(rows,indent=2))
