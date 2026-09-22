#!/usr/bin/env python3
"""Verify 0.5.0 source fingerprints and archive public runtime provenance."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

root=Path(sys.argv[1]).resolve()
archive=root/'dsFlowerClient/inst/extdata/campaign/survival/hazard-v3/provenance'
expected=json.loads((archive/'package-source.json').read_text())
for name,item in expected.items():
    for relative,digest in item['files'].items():
        assert hashlib.sha256((root/name/relative).read_bytes()).hexdigest()==digest,(name,relative)
for relative,name in [('runtime/build.json','installed-build.json'),('runtime/pip-freeze.txt','python-freeze.txt'),
    ('runtime/hazard_v3/run_started.json','run-started.json')]:
    shutil.copyfile(root/relative,archive/name)
resume=root/'runtime/hazard_v3/infrastructure_resume.json'
if resume.exists():
    shutil.copyfile(resume,archive/'infrastructure-resume.json')
tools=root/'dsFlowerClient/tools/campaign/survival/hazard_v3'
(archive/'final-tool-sha256.json').write_text(json.dumps({
    p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(tools.iterdir()) if p.is_file()},indent=2)+'\n')
packages=subprocess.check_output(['Rscript','-e',
    '.libPaths(c("/workspace/hazard/runtime/rlib",.libPaths())); p<-installed.packages(); write.table(p[,c("Package","Version")],row.names=FALSE,sep="\\t",quote=FALSE)'],text=True)
(archive/'r-packages.tsv').write_text(packages)
value=dict(verified_utc=datetime.now(timezone.utc).isoformat(),package_source_matches_v050=True,
    matched_file_counts={k:len(v['files']) for k,v in expected.items()},
    pod_id='x0w6ewmpinpsuk',platform=platform.platform(),
    cpu_count=subprocess.check_output(['nproc'],text=True).strip(),memory_limit_bytes=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
    cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
    historical_device='CUDA A40; packaged v2 used older package labels 0.4.5/0.4.4 and runner ac08384b65fe18eeb1a1bbc7c757f8103cb59990f164f32cc7b047405d8e4ac8',
    current_device='CPU; 0.5.0 source fingerprint verified; no claim of byte-identical historical runner',
    initial_harness_commit_before_branch_rebase='6047074d99ece82884e4d91d0474e1877611a48e',
    tools_revision_note='Pod base HEAD retained from preprovisioned source. The shared evidence branch was rebased while additive tooling ran; per-cell, run-started, infrastructure-resume and final-tool hashes identify the actual files independently of rewritten commit IDs.',
    provisioning='Reused existing environment. Matrix/lme4/jomo/mitml/mice installed in runtime/rlib; dsBase6.3.5 then canonical install_and_freeze.sh succeeded.',
    preflight=dict(campaign_tests=5,survival_contract_tests=30,dp_safety_checks=102,synthetic_federation='executed once, Adam + FedAvgM',
        repaired_harness_issues=['Unit-test import path and NumPy reference alias corrected before cohort fitting',
        'Synthetic artifact export path corrected after successful fit; existing release copied without retraining']))
(archive/'verification.json').write_text(json.dumps(value,indent=2)+'\n')
print('SOURCE_PROVENANCE_VERIFIED',value['matched_file_counts'])
