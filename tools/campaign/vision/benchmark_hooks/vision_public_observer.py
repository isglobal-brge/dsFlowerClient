"""Custodian-installed observer for this isolated PUBLIC vision campaign.

The .pth bootstrap imports only this stdlib module. It never imports the runner;
the mandatory sitecustomize verifier must admit the runner before attachment.
"""
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import stat
import sys
import time


CONFIG_NAME = "vision-public-benchmark.json"
CAMPAIGN_ROOT = Path("/workspace/cells-vision")


def load_config(allowed_root=CAMPAIGN_ROOT):
    secret = os.environ.get("DSFLOWER_NODE_SECRET_FILE")
    if not secret:
        return None
    parent = Path(secret).parent
    path = parent / CONFIG_NAME
    if not path.exists():
        return None
    parent_stat = parent.lstat()
    if (not stat.S_ISDIR(parent_stat.st_mode)
            or stat.S_IMODE(parent_stat.st_mode) != 0o700
            or parent_stat.st_uid != os.getuid()):
        raise RuntimeError("public observer configuration directory is unsafe")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_size > 16384):
            raise RuntimeError("public observer configuration file is unsafe")
        config = json.load(stream)
    if (not isinstance(config, dict)
            or set(config) != {"public_fixture_only", "dataset", "capture_dir", "seed"}
            or config["public_fixture_only"] is not True
            or config["dataset"] not in ("busbra",)
            or type(config["seed"]) is not int
            or not isinstance(config["capture_dir"], str)):
        raise RuntimeError("public observer requires the explicit public campaign configuration")
    capture = Path(config["capture_dir"])
    if (not capture.is_absolute() or capture.name != "public-capture"
            or allowed_root.resolve() not in capture.resolve().parents
            or capture.resolve() != capture):
        raise RuntimeError("public observer capture directory is outside the campaign")
    capture.mkdir(parents=True, exist_ok=True)
    return config


def verified_guard(observer):
    guard = sys.modules.get("sitecustomize")
    guard_type = getattr(guard, "_IntegrityFinder", None)
    active = [item for item in sys.meta_path
              if guard_type is not None and isinstance(item, guard_type)]
    if (not active or observer not in sys.meta_path
            or sys.meta_path.index(active[0]) >= sys.meta_path.index(observer)
            or "dsflower_runner" not in getattr(guard, "_verified_packages", set())
            or "dsflower_runner" not in (getattr(guard, "_PINNED_MAP", None) or {})
            or getattr(guard, "_CANONICAL_CLIENTAPP_REF", None) != "dsflower_runner.client_app:app"):
        raise RuntimeError("public observer requires the active mandatory runner integrity pin")
    return guard


def observe_fallback(original, config, observer):
    def fallback(*args, **kwargs):
        verified_guard(observer)
        kind, _error, trace = sys.exc_info()
        if kind is not None:
            frames = []
            while trace is not None:
                code = trace.tb_frame.f_code
                frames.append({"file": Path(code.co_filename).name,
                               "function": code.co_name, "line": trace.tb_lineno})
                trace = trace.tb_next
            payload = {"public_fixture_only": True, "exception_type": kind.__name__, "error": str(_error),
                       "frames": frames}
            path = Path(config["capture_dir"]) / ("failure-%d-%d.json" % (os.getpid(), time.time_ns()))
            try:
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(payload, stream, indent=2)
                    stream.write("\n")
            except OSError:
                pass  # A diagnostic write must not replace the unchanged fallback reply.
        return original(*args, **kwargs)
    return fallback


def attach(client_app, config, observer):
    verified_guard(observer)
    import torch

    if getattr(client_app._dp_fit, "_vision_public_observer", False):
        raise RuntimeError("public observer cannot attach twice")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    harness = client_app.dp_harness
    original_private = harness.make_private_dpsgd
    original_fit = client_app._dp_fit
    original_train = client_app._train_neural
    engines = []

    def train(context, cfg, pcfg, pins, *args, **kwargs):
        verified_guard(observer)
        manifest = client_app.task_module._load_manifest(context)
        public_fields = {key: value for key, value in manifest.items()
                         if key.startswith("privacy-") or key in
                         ("dp-unit", "patient_column", "patient-id-canonicalization",
                          "n_units", "n_samples", "data_type", "target-levels",
                          "backbone", "image-size", "vision-extractor-profile", "num-features")}
        result = original_train(context, cfg, pcfg, pins, *args, **kwargs)
        path = Path(config["capture_dir"]) / ("manifest-%d-%d.json" % (os.getpid(), time.time_ns()))
        path.write_text(json.dumps(dict(round=pins["round_index"], manifest=public_fields), indent=2) + "\n")
        return result

    def make_private(*args, **kwargs):
        result = original_private(*args, **kwargs)
        engines.append(result[3])
        return result

    def fit(model, X, y, pcfg, pins, n_staged, cfg, master, noise_multiplier, **kwargs):
        verified_guard(observer)
        if cfg.get("loss-name") != "cross_entropy":
            raise RuntimeError("public observer is limited to vision")
        started = time.monotonic()
        before = len(engines)
        result = original_fit(model, X, y, pcfg, pins, n_staged, cfg, master,
                              noise_multiplier, **kwargs)
        if len(engines) != before + 1:
            raise RuntimeError("benchmark expected exactly one accountant per node round")
        engine = engines[-1]
        mechanism = harness.effective_dpsgd_mechanism(
            pcfg["epsilon"], pcfg["delta"], pcfg["clipping_norm"], len(X),
            pins["batch_size"], pins["local_epochs"], pins["num_rounds"])
        history = engine.accountant.history
        observed = sum(item[2] for item in history)
        if observed != mechanism["steps_per_epoch"] * pins["local_epochs"]:
            raise RuntimeError("observed accountant steps do not match logical batches")
        payload = {"schema": "dsflower-vision-benchmark-accountant-v1",
                   "public_fixture_only": True, "dataset": config["dataset"],
                   "round": pins["round_index"], "source_rows": n_staged,
                   "mechanism": mechanism, "privacy_config": pcfg, "training_pins": pins, "observed_round_steps": observed,
                   "accountant_type": type(engine.accountant).__name__,
                   "accountant_history": history,
                   "elapsed_s": time.monotonic() - started,
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                   "features_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
                   "targets_sha256": hashlib.sha256(y.tobytes()).hexdigest()}
        path = Path(config["capture_dir"]) / ("accountant-%d-%d.json" % (os.getpid(), time.time_ns()))
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        return result

    fit._vision_public_observer = True
    harness.make_private_dpsgd = make_private
    client_app._dp_fit = fit
    client_app._train_neural = train
    client_app._safe_fallback_reply = observe_fallback(client_app._safe_fallback_reply, config, observer)


class ObservedLoader:
    def __init__(self, wrapped, observer):
        self.wrapped = wrapped
        self.observer = observer

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        verified_guard(self.observer)
        self.wrapped.exec_module(module)
        attach(module, self.observer.config, self.observer)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class PublicObserver:
    def __init__(self, config):
        self.config = config

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "dsflower_runner.client_app":
            return None
        verified_guard(self)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            raise RuntimeError("public observer cannot resolve the verified ClientApp")
        spec.loader = ObservedLoader(spec.loader, self)
        return spec


def install():
    config = load_config()
    if config is None:
        return
    # Limit native pools before numerical imports; the trusted clean environment
    # intentionally strips inherited analyst thread controls. Public fixture only.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    if "dsflower_runner" in sys.modules:
        raise RuntimeError("public observer must precede any runner import")
    # sitecustomize runs after .pth processing and inserts its mandatory finder
    # at index zero, ahead of this observer and Python's normal PathFinder.
    sys.meta_path.insert(0, PublicObserver(config))
