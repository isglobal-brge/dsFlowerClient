"""Audit completed public campaign artifacts without rescoring any model."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

root = Path('/workspace/cells-sequence')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
tools = root / 'src/dsFlowerClient/tools/campaign/sequence'
protocol = read(tools / 'protocol.json')
assert sha(tools / 'protocol.json') == '8c518e23b3eccdc026ad21022c1d062a7f8b91b3dc3db9f5e5a5d5d18a0f8b7e'
runtime = read(root / 'runtime.json')
assert runtime['packages']['dsFlower'] == '0.5.1'
assert runtime['packages']['dsFlowerClient'] == '0.5.0'
assert runtime['packages']['server_runner_sha256'] == runtime['packages']['client_runner_sha256'] == '2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724'
assert sha(root / 'Rlib/dsFlower/python/sitecustomize.py') == runtime['packages']['server_guard_sha256']
for path, digest in runtime['tooling_sha256'].items():
    assert sha(tools / path) == digest
records = []
initial_by_seed = {}
central_by_seed = {}
for epsilon in protocol['epsilon_order']:
    for seed in protocol['seeds']:
        run = root / 'runs' / f'pytorch_lstm-eps{epsilon}-seed{seed}'
        federation = read(run / 'federation-status.json')
        central = read(run / 'central/status.json')
        execution = read(run / 'execution-status.json')
        assert federation['status'] == central['status'] == execution['status'] == 'trained_unscored'
        assert federation['cleanup_ok']
        assert sha(Path(federation['output_dir']) / 'model.pt') == federation['model_sha256']
        assert sha(run / 'central/model.pt') == central['model_sha256']
        initial = read(run / 'public-capture/public-initial.json')['tensor_sha256']
        assert initial_by_seed.setdefault(seed, initial) == initial
        assert central_by_seed.setdefault(seed, central['model_sha256']) == central['model_sha256']
        captures = [read(p) for p in (run / 'public-capture').glob('accountant-*.json')]
        assert len(captures) == 15
        counts = {str(i): sum(c['round'] == i for c in captures) for i in range(1, 6)}
        assert set(counts.values()) == {3}
        assert all(c['mechanism']['accounting_population'] == 7 and c['mechanism']['total_steps'] == 5
                   and c['privacy_config']['clipping_norm'] == 1 and c['privacy_config']['delta'] == 1e-6
                   and c['training_pins']['local_epochs'] == 1 and c['training_pins']['batch_size'] == 32
                   and c['training_pins']['learning_rate'] == .001 for c in captures)
        sigma = {c['mechanism']['noise_multiplier'] for c in captures}
        assert len(sigma) == 1
        records.append({'epsilon': epsilon, 'seed': seed, 'cleanup_ok': True,
                        'node_round_captures': len(captures), 'captures_per_round': counts,
                        'noise_multiplier': sigma.pop(), 'federated_elapsed_s': federation['elapsed_s'],
                        'central_elapsed_s': central['elapsed_s'], 'execution': execution,
                        'federated_model_sha256': federation['model_sha256'],
                        'central_model_sha256': central['model_sha256']})
assert sha(root / 'data_cache/uci-har-240.zip') == 'c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031'
result = {'schema': 'dsflower-sequence-execution-audit-v1', 'recorded_at': datetime.now(timezone.utc).isoformat(),
          'task_token': 'FLOWER_CELLS_SEQUENCE_2026-09-22', 'completed_replicates': len(records),
          'node_round_captures': sum(r['node_round_captures'] for r in records),
          'protocol_unchanged': True, 'settings_changed': False, 'local_epochs_reduced': False,
          'all_model_hashes_verified': True, 'central_checkpoint_identical_across_epsilon_for_each_seed': True,
          'test_scoring_started': (root / 'test-scoring-started.json').exists(),
          'per_replicate': records}
(root / 'execution-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k,v in result.items() if k != 'per_replicate'}))
