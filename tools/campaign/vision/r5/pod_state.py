#!/usr/bin/env python3
"""Record/clean only stopped vision training processes on this isolated pod."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import os

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--clean',action='store_true');a=p.parse_args()
assert Path('/workspace/cells-vision').is_dir()
rows=[]
for line in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines():
    pid,ppid,command=line.strip().split(None,2)
    if any(x in command for x in ['flower-superlink','flower-supernode','flower-superexec',
            'flwr.superlink','flwr.supernode','flwr.serverapp','flwr.clientapp',
            'flwr-serverapp','flwr-clientapp','parallel:::.workRSOCK',
            'r4/run_matrix.py','r4/run_federated.R','r5/run_federated.R']):
        if int(pid)!=os.getpid() and 'pod_state.py' not in command:
            rows.append(dict(pid=int(pid),ppid=int(ppid),command=command))
terminated=[]
if a.clean:
    for row in rows:
        os.kill(row['pid'],signal.SIGTERM);terminated.append(row['pid'])
record=dict(time=datetime.now(timezone.utc).isoformat(),matching_processes=rows,terminated_pids=terminated,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used','--format=csv'],text=True),pod_left_running=True)
a.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
