"""Opt-in PUBLIC BENCHMARK instrumentation, never installed in the runner.

Place this directory and the byte-identical runner parent on PYTHONPATH only
for the public campaign. No private tensors, labels or secrets are captured.
Defer heavy imports until Flower actually loads the ServerApp, so generic
SuperLink/CLI startup does not initialize the training runtime.
"""
import importlib.machinery
import os
import sys


def _attach(server_app):
    import hashlib
    import json
    from pathlib import Path
    import time
    import numpy as np
    import torch
    from dsflower_runner.params import get_torch_params

    directory = Path(os.environ["F_SEQUENCE_CAPTURE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ["F_SEQUENCE_INIT_SEED"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    original_initial = server_app._initial_arrays

    def initial(cfg, track):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model, record = original_initial(cfg, track)
        arrays = get_torch_params(model)
        destination = directory / "public-initial-arrays.npz"
        np.savez(destination, **{str(i): value for i, value in enumerate(arrays)})
        (directory / "public-initial.json").write_text(json.dumps({
            "seed": seed, "model_spec_b64": cfg["model-spec-b64"],
            "tensor_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
            "config": dict(cfg)},
            indent=2, sort_keys=True) + "\n")
        return model, record

    server_app._initial_arrays = initial


class _ObservedLoader:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        _attach(module)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class _ServerAppObserver:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "dsflower_runner.server_app":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            raise RuntimeError("public benchmark cannot resolve the ServerApp")
        spec.loader = _ObservedLoader(spec.loader)
        return spec


if os.environ.get("F_SEQUENCE_PUBLIC_BENCHMARK") == "1":
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    sys.meta_path.insert(0, _ServerAppObserver())
