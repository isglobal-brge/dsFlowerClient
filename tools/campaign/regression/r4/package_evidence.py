"""Validate saved scalar evidence and update the track without rescoring models."""
import hashlib
import json
from pathlib import Path
import statistics
import math

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = REPO / 'inst/extdata/campaign/regression'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)

def msd(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values))

selection = json.loads((HERE / 'selection.json').read_text())
params = selection['selected_params']
assert not selection['sealed_test_opened']
assert selection['selected'] == min(selection['candidates'], key=lambda c:c['rmse_mean'])['candidate']
for candidate in selection['candidates']:
    values=[]
    for rep in candidate['replicates']:
        fed=rep['federated']
        assert not rep['sealed_test_opened']
        assert fed['n_clients']==3 and fed['n_rounds_run']==5 and fed['n_failures']==0 and fed['cleanup_ok']
        values.append(fed['metrics']['validation']['rmse'])
        for node in fed['node_privacy']:
            assert node['policy']['per_training_epsilon']==8 and node['policy']['per_training_delta']==1e-6
            assert node['policy']['dp_unit']=='row' and node['clipping_norm']==1
            for cfg in node['staged_configurations']:
                assert cfg['n_samples']==9600 and cfg['local-epochs']==5
                assert cfg['batch-size']==candidate['model_params']['batch_size']
                assert cfg['learning-rate']==candidate['model_params']['learning_rate']
    close(candidate['rmse_mean'],statistics.mean(values))
    close(candidate['rmse_sd'],statistics.stdev(values))
records = []
for eps in [1, 4, 8]:
    path = OUT / f'cdcbmi_r4_pytorch_linear_regression_eps{eps}.json'
    obj = json.loads(path.read_text())
    assert obj['schema'] == 'dsflower-campaign-v2'
    assert obj['model_params_overrides'] == params
    assert obj['versions']['runner_sha256'] == '2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724'
    assert obj['predeclaration_commit'] == (HERE / 'declaration_commit.txt').read_text().strip()
    assert obj['seeds'] == [20260820, 20260821, 20260822]
    assert obj['privacy'] == dict(epsilon=eps, delta=1e-6, unit='row', clipping_norm=1)
    for rep in obj['per_replicate']:
        assert rep['history'] == dict(n_clients=3, n_failures=0, n_rounds=5, cleanup_ok=True)
        assert rep['split']['n_train'] == 36000 and rep['split']['n_test'] == 9000
        assert rep['split']['membership_sha256'] == sha(OUT / rep['split']['membership_file'])
        for filename,key in [('train.csv','train_sha256'),('test.csv','test_sha256')]:
            assert rep['baseline_provenance'][key] == obj['dataset']['checksums'][f"cdcbmi_seed{rep['seed']}/{filename}"]
        pool = rep['pooled_training']
        assert pool['n_clients'] == 1 and pool['n_rounds_run'] == 5 and pool['n_failures'] == 0 and pool['cleanup_ok']
        for nodes, n in [(rep['node_privacy'],12000),(pool['node_privacy'],36000)]:
            for node in nodes:
                pol = node['policy']
                assert pol['dp_unit'] == 'row' and pol['adjacency'] == 'replace_one'
                assert pol['per_training_epsilon'] == eps and pol['per_training_delta'] == 1e-6
                assert node['clipping_norm'] == 1
                for cfg in node['staged_configurations']:
                    for key, value in {'n_samples':n,'n_units':n,'num-server-rounds':5,
                        'local-epochs':params['local_epochs'], 'batch-size':params['batch_size'],
                        'learning-rate':params['learning_rate'], 'optimizer-name':params['optimizer'],
                        'weight-decay':0,'l1-penalty':0,'scheduler-name':'none',
                        'privacy-epsilon':eps,'privacy-delta':1e-6,'privacy-clipping_norm':1,
                        'dp_enabled':True,'fixed_client_sampling':True}.items():
                        assert cfg[key] == value, (key,cfg[key],value)
                    assert cfg['feature-bounds'] == obj['protocol']['feature_bounds']
                    assert cfg['target-bounds'] == obj['target_bounds']
        close(rep['gap_rmse'], rep['federated_dp']['rmse'] - rep['central']['rmse'])
        assert rep['diagnostics']['rmse_below_trivial'] == (rep['federated_dp']['rmse'] < rep['trivial']['rmse'])
        assert rep['diagnostics']['r2_positive'] == (rep['federated_dp']['r2'] > 0)
        for arm in ['central','nonprivate_federated','federated_dp','pooled_dp','trivial']:
            for metric in ['rmse','mae','r2']:
                assert math.isfinite(rep[arm][metric])
        dec = rep['federated_dp']['decomposition']
        close(rep['federated_dp']['rmse']**2, dec['residual_variance'] + dec['mean_residual']**2)
    for arm, field in [('central','central'),('nonprivate_federated','nonprivate_federated'),
                       ('federated_dp','federated'),('pooled_dp','pooled_dp'),('trivial','trivial')]:
        for metric in ['rmse','mae','r2']:
            expected = msd([rep[arm][metric] for rep in obj['per_replicate']])
            for key, value in expected.items():
                close(obj['summary'][field+'_mean_sd'][metric][key], value)
    for key,value in msd([r['gap_rmse'] for r in obj['per_replicate']]).items():
        close(obj['summary']['gap_rmse_mean_sd'][key],value)
    for name, digest in obj['tooling_sha256'].items():
        assert sha(HERE / name) == digest, name
    records.append((path,obj))
for i in range(3):
    for arm in ['central','nonprivate_federated','trivial']:
        assert records[0][1]['per_replicate'][i][arm] == records[1][1]['per_replicate'][i][arm] == records[2][1]['per_replicate'][i][arm]

effective_protocol = dict(records[0][1]['protocol'])
effective_protocol.update(cell='cdcbmi_r4', model_params_overrides=params,
    selection=selection['criterion'],
    schedule_rationale='Training-only real-DP selection increased local epochs from one to five. Five rounds, SGD .01, batch 32; 9375 steps per 12000-row site. No post-test choice.',
    base_protocol_sha256=records[0][1]['protocol_sha256'])
(OUT / 'cdcbmi_r4_protocol.json').write_text(json.dumps(effective_protocol,indent=2)+'\n')
summary_path = OUT / 'summary.json'
summary = json.loads(summary_path.read_text())
for cell in summary['cells']:
    if cell['id'] == 'cdcbmi_public_units':
        cell['r4_diagnosis'] = dict(file='REGRESSION_DIAGNOSIS_R4.md',
            interpretation='Training-only released-step emulation reproduces the above-trivial error without noise. The default one-local-epoch clipped schedule is optimization-limited and initialization-sensitive; no released-code shape, divisor or prediction-transform defect was found. Its DP-minus-OLS gap is not a noise-only privacy cost.')
cell = dict(id='cdcbmi_r4', contract='pytorch_linear_regression',
    dataset='UCI CDC Diabetes Health Indicators', thesis_citation_key='uci_cdc_diabetes_health_indicators',
    n=45000, n_train=36000, n_test=9000, privacy_unit='row', delta=1e-6,
    selected_params=params, selection_file='cdcbmi_r4_selection.json',
    effective_protocol_file='cdcbmi_r4_protocol.json',
    protocol_inheritance_note='Execution records retain the untouched public-unit base protocol and its checksum. Top-level model_params/model_params_overrides and node-reported configurations are the effective corrected schedule; the separate R4 protocol resolves that inheritance.',
    interpretation='Schedule selected using real epsilon-8 federated-DP inner-validation RMSE on training rows only. OLS measures the optimum of the public-unit least-squares objective; the non-private finite-schedule twin exposes remaining optimization/clipping error. The federated-DP minus noiseless-twin difference also contains independent initialization and sampling variation, so it is not an exactly paired causal noise effect. Pooled DP uses one site with all 36000 training rows and recalibrates its complete horizon. SDs across overlapping splits are descriptive; privacy accounting is per training, with no composed-grid or private-selection guarantee.',
    budgets=[dict(epsilon=obj['privacy']['epsilon'], evidence=dict(file=path.name,sha256=sha(path)),
                  summary=obj['summary']) for path,obj in records])
summary['cells'] = [c for c in summary['cells'] if c['id'] != 'cdcbmi_r4']+[cell]
summary['policy'] = 'Historical scored artifacts are unchanged. R4 was separately authorized, selected on training-only inner validation, declared before scoring, and executed once per epsilon/seed. No post-test alternative or scored-cell rerun.'
summary_path.write_text(json.dumps(summary,indent=2)+'\n')
(OUT / 'cdcbmi_r4_selection.json').write_bytes((HERE / 'selection.json').read_bytes())
print('| ε | OLS | Non-private federated | Federated DP | Pooled DP | Trivial | DP R² | Gap |')
print('|---:|---:|---:|---:|---:|---:|---:|---:|')
for _,obj in records:
    s=obj['summary']
    values=[str(obj['privacy']['epsilon'])]
    for key in ['central','nonprivate_federated','federated','pooled_dp','trivial']:
        x=s[key+'_mean_sd']['rmse']; values.append(f"{x['mean']:.6f} ± {x['sd']:.6f}")
    for x in [s['federated_mean_sd']['r2'],s['gap_rmse_mean_sd']]:
        values.append(f"{x['mean']:.6f} ± {x['sd']:.6f}")
    print('| '+' | '.join(values)+' |')
print('Validated all arms, node contracts, three seeds, metrics, gaps, summaries and tooling hashes.')
