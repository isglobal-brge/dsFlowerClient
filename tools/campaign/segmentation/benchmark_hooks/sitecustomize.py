"""Opt-in PUBLIC BENCHMARK instrumentation, never installed in the runner.

Place this directory and the byte-identical runner parent on PYTHONPATH only
for the public campaign. No private tensors, labels or secrets are captured.
"""
import os

if os.environ.get("F_SEG_PUBLIC_BENCHMARK") == "1":
    import hashlib
    import json
    from pathlib import Path
    import time
    import numpy as np
    import torch
    from dsflower_runner import client_app, dp_harness, server_app
    from dsflower_runner.params import get_torch_params

    directory = Path(os.environ["F_SEG_CAPTURE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ["F_SEG_INIT_SEED"])
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
            "config": dict(cfg)}, indent=2, sort_keys=True) + "\n")
        return model, record

    server_app._initial_arrays = initial
    engines = []
    original_private = dp_harness.make_private_dpsgd

    def make_private(*args, **kwargs):
        result = original_private(*args, **kwargs)
        engines.append(result[3])
        return result

    dp_harness.make_private_dpsgd = make_private
    original_fit = client_app._dp_fit

    def fit(model, X, y, pcfg, pins, n_staged, cfg, master, noise_multiplier, **kwargs):
        before = len(engines)
        result = original_fit(model, X, y, pcfg, pins, n_staged, cfg, master, noise_multiplier, **kwargs)
        if len(engines) != before + 1:
            raise RuntimeError("benchmark expected exactly one accountant per node round")
        engine = engines[-1]
        mechanism = dp_harness.effective_dpsgd_mechanism(
            pcfg["epsilon"], pcfg["delta"], pcfg["clipping_norm"], len(X),
            pins["batch_size"], pins["local_epochs"], pins["num_rounds"])
        history = engine.accountant.history
        observed = sum(item[2] for item in history)
        if observed != mechanism["steps_per_epoch"] * pins["local_epochs"]:
            raise RuntimeError("observed accountant steps do not match logical batches")
        payload = {"schema": "dsflower-segmentation-benchmark-accountant-v1",
                   "public_fixture_only": True, "round": pins["round_index"],
                   "mechanism": mechanism, "observed_round_steps": observed,
                   "accountant_type": type(engine.accountant).__name__,
                   "accountant_history": history,
                   "features_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
                   "targets_sha256": hashlib.sha256(y.tobytes()).hexdigest()}
        path = directory / ("accountant-%d-%d.json" % (os.getpid(), time.time_ns()))
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return result

    client_app._dp_fit = fit
