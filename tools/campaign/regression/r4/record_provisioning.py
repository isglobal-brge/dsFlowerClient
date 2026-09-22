"""Record the installed diagnostic runtime without opening any dataset rows."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/workspace/cells')
env = dict(os.environ, R_LIBS=str(ROOT / 'Rlib'),
           DSFLOWER_VENV_ROOT=str(ROOT / 'venvs'),
           DSFLOWER_CLIENT_VENV_ROOT=str(ROOT / 'client'))


def run(*args):
    return subprocess.check_output(args, env=env, text=True).strip()


r_state = json.loads(run('Rscript', '-e', '''
p <- c("dsFlower","dsFlowerClient","DSI","DSLite","resourcer","arrow", "digest", "jsonlite", "processx", "ps", "filelock")
z <- list(version=R.version.string,
  packages=setNames(lapply(p,function(x) as.character(packageVersion(x))),p),
  server_runner_sha256=dsFlower:::.compute_harness_hash(),
  client_runner_sha256=dsFlowerClient:::.compute_local_runner_hash(),
  health=list(native_tree=dsFlower:::.venv_is_healthy("/workspace/cells/venvs/native-tree","native-tree"),
    pytorch=dsFlower:::.venv_is_healthy("/workspace/cells/venvs/pytorch","pytorch"),
    client=dsFlowerClient:::.client_venv_is_healthy()))
cat(jsonlite::toJSON(z,auto_unbox=TRUE))
'''))
python_state = {}
for name, path in [('native_tree', 'venvs/native-tree'),
                   ('pytorch', 'venvs/pytorch'), ('client', 'client/venv')]:
    python = str(ROOT / path / 'bin/python')
    python_state[name] = json.loads(run(python, '-c', '''
import importlib.metadata as m, json, sys
names = ['flwr','numpy','pandas','pyarrow','cryptography','torch','opacus','optuna','scipy']
present = {d.metadata['Name'].lower(): d.version for d in m.distributions()}
print(json.dumps({'path':sys.executable,'version':sys.version,'packages':{n:present[n] for n in names if n in present}}))
'''))
paths = ['regression-r4-server-v050.tar', 'regression-r4-client-v050.tar',
         'wheels.tar', 'wheels/CHECKSUMS.sha256',
         'dsFlowerClient/tools/campaign/regression/runtime-constraints.txt',
         'dsFlowerClient/tools/campaign/regression/pylock.toml']
hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
state = {
    'pod': 'pod-flower-regression', 'root': str(ROOT),
    'before': {'r_version': '4.6.1', 'uv_version': '0.11.33',
               'python_3_11_installed': False, 'package_sources_extracted': False,
               'r_dependencies_installed': True, 'r_arrow_roundtrip_verified': True,
               'python_venvs_present': False, 'source_tag_archives_present': True,
               'wheel_transfer_incomplete': True},
    'cpu_count': int(run('nproc')),
    'cgroup_memory_bytes': int(Path('/sys/fs/cgroup/memory.max').read_text()),
    'nvidia_devices': [str(p) for p in Path('/dev').glob('nvidia*')],
    'uv_version': run('uv', '--version'), 'r': r_state, 'python': python_state,
    'tags': {
        'dsFlower': {'tag':'v0.5.0','commit':'408f08c539329e2711260050ab40a6567aa4d89e'},
        'dsFlowerClient': {'tag':'v0.5.0','commit':'50dda000a32ffcbdd039c2b74c909df451392bfb'}},
    'sha256': hashes,
    'times': {n: (ROOT / 'logs' / f'r4-provision-{n}.txt').read_text().strip()
              for n in ['start','resume','end']},
    'status': 'complete', 'left_running': True,
}
assert all(r_state['health'].values())
assert r_state['server_runner_sha256'] == r_state['client_runner_sha256']
target = ROOT / 'r4/provisioning.json'
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(state, indent=2) + '\n')
print(target.read_text())
