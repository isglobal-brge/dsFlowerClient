#!/usr/bin/env python3
"""Reproduce public model construction with the installed gate and no observer.

No data are staged and no training function is called. The child has the real
runner pin, the real default-deny hook, and an otherwise empty neural manifest.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main(root):
    library = root / "Rlib/dsFlower"
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    with tempfile.TemporaryDirectory(prefix="vision-import-check-") as temporary:
        manifest = Path(temporary)
        (manifest / "manifest.json").write_text(json.dumps({"dp-track": "neural"}))
        (manifest / "pinned_packages.json").write_text(json.dumps({
            "dsflower_runner": protocol["release"]["runner_sha256"]}))
        env = dict(os.environ)
        env.pop("DSFLOWER_NODE_SECRET_FILE", None)
        env["DSFLOWER_MANIFEST_DIR"] = temporary
        env["PYTHONPATH"] = os.pathsep.join([str(library / "python"), str(library / "flower_app")])
        code = ("import json,sys; import dsflower_runner.client_app; "
                "print('CLIENTAPP_IMPORT_PASSED', flush=True); from dsflower_runner import params; "
                "cfg=json.load(open(sys.argv[1]))['config']; "
                "params.load_user_model(cfg,512,'cross_entropy'); print('PUBLIC_MODEL_BUILD_PASSED')")
        config = root / "runs/pytorch_resnet18-eps1-seed20260919/public-capture/public-initial.json"
        result = subprocess.run([sys.executable, "-c", code, str(config)],
                                env=env, capture_output=True, text=True, timeout=180)
    evidence = dict(schema="dsflower-vision-import-check-v1", returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr, benchmark_observer_enabled=False,
        public_model_construction_called=True, training_called=False,
        data_staged=False, integrity_gate_modified=False)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
