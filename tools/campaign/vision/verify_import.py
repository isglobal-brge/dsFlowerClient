#!/usr/bin/env python3
"""Reproduce public model construction with the installed gate and no observer.

No data are staged and no training function is called. The child has the real
runner pin, the real default-deny hook, and an otherwise empty neural manifest.
"""
import argparse
import base64
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
        code = ("import json,sys,sitecustomize; "
                "assert any(isinstance(f, sitecustomize._IntegrityFinder) for f in sys.meta_path); "
                "import dsflower_runner.client_app; "
                "print('CLIENTAPP_IMPORT_PASSED', flush=True); from dsflower_runner import params; "
                "cfg=json.load(open(sys.argv[1]))['config']; "
                "model=params.load_user_model(cfg,512,'cross_entropy'); "
                "assert sum(p.numel() for p in model.parameters()) == 1026; "
                "assert 'dsflower_runner' in sitecustomize._verified_packages; "
                "print('PUBLIC_MODEL_BUILD_PASSED')")
        config = manifest / "public-config.json"
        spec = {"kind": "sequential", "layers": [{"op": "linear", "out": "@out"}]}
        config.write_text(json.dumps({"config": {
            "backbone": "resnet18", "image-size": 224, "num-features": 512,
            "num-classes": 2, "loss-name": "cross_entropy",
            "model-spec-b64": base64.b64encode(json.dumps(spec).encode()).decode()}}))
        result = subprocess.run([sys.executable, "-c", code, str(config)],
                                env=env, capture_output=True, text=True, timeout=180)
    evidence = dict(schema="dsflower-vision-import-check-v1", returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr, benchmark_observer_enabled=False,
        public_model_construction_called=True, training_called=False,
        data_staged=False, integrity_gate_modified=False)
    print(json.dumps(evidence, indent=2))
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    raise SystemExit(main(parser.parse_args().root))
