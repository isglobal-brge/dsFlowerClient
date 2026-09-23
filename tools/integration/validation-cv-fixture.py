#!/usr/bin/env python3
"""Synthetic DSLite fixture and in-process transport for production Flower apps.

Only network/process orchestration is replaced. Admission, private staging,
trusted predictors, DP training, release guards and coordinator pooling are real.
Fixture-only access to node staging is supplied by the DSLite custodian test.
"""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture(root, encoder):
    import numpy as np
    from dsflower_runner import segmentation_checkpoints as checkpoints
    spec = importlib.util.spec_from_file_location(
        "public_fixture", Path(__file__).with_name("public-initialisation-fixture.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identities = {"segmentation": module.fixture(root, encoder)}
    segmentation_manifest = json.loads((root / "bundle/manifest.json").read_text())
    for kind in ("tabular", "survival"):
        directory = root / kind
        directory.mkdir()
        manifest = {key: value for key, value in segmentation_manifest.items()
                    if key not in {"decoder", "decoder_spec_sha256", "encoder", "encoder_sha256"}}
        manifest["checkpoint_id"] = "synthetic-valcv-" + kind
        manifest["role"] = "tabular_model"
        manifest["model_id"] = "declarative_neural"
        manifest["model_spec"] = {"kind": "sequential", "layers": [{"op": "linear", "out": "@out"}]}
        manifest["model_spec_sha256"] = sha(checkpoints._canonical(manifest["model_spec"]))
        manifest["model_config"] = {"loss-name": "bce_logits", "num-features": 2,
                                    "num-classes": 2, "num-labels": 2}
        manifest["feature_contract"] = {"features": ["x1", "x2"], "feature_lower": [-1, -1],
                                        "feature_upper": [1, 1], "target_levels": [0, 1],
                                        "target_bounds": None}
        if kind == "survival":
            manifest["model_config"]["loss-name"] = "aft_weibull_nll"
            survival = {"schema_version": 1, "time_unit": "days", "time_origin": "baseline",
                        "t_min": 1, "horizon": 10, "time_scale": 1,
                        "distribution": "weibull", "dispersion": 1}
            manifest["model_config"]["survival-config-b64"] = base64.b64encode(
                json.dumps(survival).encode()).decode()
            manifest["feature_contract"]["target_levels"] = None
        arrays = [np.array([[0.25, -0.1]], dtype=np.float32), np.array([0.1], dtype=np.float32)]
        stream = io.BytesIO()
        np.savez(stream, **{str(i): array for i, array in enumerate(arrays)})

        def artifact(name, content):
            (directory / name).write_bytes(content)
            return {"file": name, "size_bytes": len(content), "sha256": sha(content)}

        manifest["checkpoint"] = artifact("checkpoint.npz", stream.getvalue())
        manifest["tensors"] = [{"name": str(i), "shape": list(array.shape), "dtype": "float32",
                                "sha256": sha(array.tobytes())} for i, array in enumerate(arrays)]
        for name, record in manifest["evidence"].items():
            if name != "original_manifest":
                (directory / record["file"]).write_bytes((root / "bundle" / record["file"]).read_bytes())
        original = {"dataset": manifest["dataset"]["dataset"],
                    "dataset_sha256": manifest["dataset"]["sha256"], "privacy": "public_nonprivate",
                    "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                    "tensor_sha256": [item["sha256"] for item in manifest["tensors"]],
                    "model_spec_sha256": manifest["model_spec_sha256"],
                    **{key + "_sha256": manifest["evidence"][key]["sha256"]
                       for key in ("protocol", "provenance", "audit")}}
        manifest["evidence"]["original_manifest"] = artifact("original.json", json.dumps(original).encode())
        (directory / "manifest.json").write_text(json.dumps(manifest))
        bundle = root / (kind + ".zip")
        summary = checkpoints.pack_bundle(directory, bundle)
        identities[kind] = {"bundle_sha256": sha(bundle.read_bytes()),
                            "manifest_sha256": summary["provenance"]["manifest_sha256"]}
    return identities


def execute(app_directory, node_file, report_file):
    import numpy as np
    import torch
    from flwr.common import RecordDict
    from dsflower_runner import client_app, server_app, seeding, resampling, validation, task, dp_harness
    torch.set_num_threads(1)
    cfg = tomllib.loads((app_directory / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    nodes = json.loads(node_file.read_text())
    manifests = [json.loads((Path(node["directory"]) / "manifest.json").read_text()) for node in nodes]
    contexts = {index: SimpleNamespace(run_config=cfg, state=RecordDict(),
                                      node_config={"manifest-dir": node["directory"]})
                for index, node in enumerate(nodes, 1)}
    exchanges = []
    first_arrays = {}
    private_vectors, plain_vectors, metric_sigmas = [], [], []

    def plain_metric_vector(context, arrays):
        """Independent synthetic reference arithmetic, kept inside the fixture."""
        from dsflower_runner import segmentation
        from dsflower_runner.params import load_user_model, set_torch_params
        effective = task.load_pinned_run_config(context)
        layout = validation.layout_from_config(effective)
        if layout["task"] not in ("segmentation", "survival"):
            return None
        model = load_user_model(effective, int(effective["num-features"]), effective["loss-name"])
        set_torch_params(model, arrays)
        model.eval()
        if layout["task"] == "segmentation":
            encoder, device = segmentation.prepare_encoder(effective)
            x, y, _, _ = segmentation.load_subject_tensors(context, effective, encoder, device)
            with torch.no_grad():
                foreground = model(torch.as_tensor(x, dtype=torch.float32)).numpy() >= 0
            reference = y[:, :1] >= .5
            valid = y[:, 1, 0, 0] == 1
            intersection = (foreground & reference).sum((1, 2, 3)) * valid
            predicted = foreground.sum((1, 2, 3)) * valid
            truth = reference.sum((1, 2, 3)) * valid
            raw = np.array([len(y), intersection.sum() / 16384,
                            predicted.sum() / 16384, truth.sum() / 16384])
            exact = 2 * intersection.sum() / max(predicted.sum() + truth.sum(), 1e-12)
            assert abs(validation.validation_metrics(raw, layout)["foreground_dice"] - exact) < 1e-12
        else:
            x, y, _, _ = task.load_survival_data(context, metric_targets=True)
            bounds = effective["feature-bounds"]
            low, high = np.array(bounds["lower"]), np.array(bounds["upper"])
            x = ((np.clip(x, low, high) - (low + high) / 2) / ((high - low) / 2)).astype(np.float32)
            with torch.no_grad():
                mu = model(torch.as_tensor(x)).numpy().reshape(-1).clip(-10, 10)
            public = effective["survival-config"]
            assert public["distribution"] == "weibull"
            time, event, valid = y.T
            k, scale = public["dispersion"], public["time_scale"]
            a = np.log(time / scale) - mu
            cumulative = np.exp(k * a)
            nll = cumulative - event * (np.log(k) - mu + (k - 1) * a - np.log(scale))
            bound = layout["nll_bound"]
            terms = [valid.sum(), (valid * (nll.clip(-bound, bound) + bound) / (2 * bound)).sum()]
            errors, denominators = [], []
            for horizon in layout["horizons"]:
                observed = (valid == 1) & ((time >= horizon) | ((event == 1) & (time <= horizon)))
                alive = ~((event == 1) & (time <= horizon))
                prediction = np.exp(-np.exp(k * (np.log(horizon / scale) - mu)))
                errors.append(float(((prediction - alive) ** 2 * observed).sum()))
                denominators.append(float(observed.sum()))
            raw = np.array(terms + errors + denominators, dtype=float)
            exact = validation.validation_metrics(raw, layout)
            assert abs(exact["negative_log_likelihood"] - float((nll.clip(-bound, bound) * valid).sum() / valid.sum())) < 1e-6
            np.testing.assert_allclose(exact["brier"]["scores"], np.array(errors) / denominators, atol=1e-12)
        privacy = task.load_privacy_config(context)
        metric_sigmas.append(dp_harness.compute_output_sigma(
            privacy["epsilon"], privacy["delta"], layout["sensitivity"], num_releases=1))
        return raw

    class Grid:
        def get_node_ids(self):
            return list(contexts)

        def send_and_receive(self, messages, timeout):
            replies = []
            for message in messages:
                node = int(message.metadata.dst_node_id)
                os.environ["DSFLOWER_NODE_SECRET_FILE"] = nodes[node - 1]["secret"]
                control = dict(message.content["config"])
                operation = control.get("dsflower-operation", "train")
                fold = int(control.get("dsflower-fold", 0))
                round_index = int(control.get("server-round", 0))
                if operation == "cv-train" and round_index == 1:
                    arrays = message.content["arrays"].to_numpy_ndarrays()
                    first_arrays.setdefault(fold, [array.copy() for array in arrays])
                response = client_app.train(message, contexts[node])
                assert not response.has_error(), response.error
                metrics = dict(response.content["metrics"])
                assert not metrics.get("public-preflight-unavailable", 0), (operation, node, metrics)
                assert not metrics.get("execution-unavailable", 0), (operation, node, metrics)
                if cfg.get("dp-track") == "validation":
                    plain = plain_metric_vector(contexts[node], message.content["arrays"].to_numpy_ndarrays())
                    if plain is not None:
                        plain_vectors.append(plain)
                        private_vectors.append(response.content["arrays"].to_numpy_ndarrays()[0])
                exchanges.append((operation, fold, round_index, node))
                replies.append(response)
            return replies

    grid = Grid()
    if cfg.get("dp-track") == "validation":
        metrics, n_nodes, available = server_app._run_validation(grid, cfg)
        assert available and n_nodes == len(nodes), "complete validation pooling failed"
        server_app._save_validation(cfg, metrics, n_nodes, available)
    elif cfg.get("cv-contract-sha256"):
        server_app._run_cross_validation(grid, cfg, "neural")
        assert len(first_arrays) == 2
        assert all(np.array_equal(a, b) for a, b in zip(first_arrays[1], first_arrays[2])), "fold starts differ"
        assert sum(op == "cv-train" for op, *_ in exchanges) == 4 * len(nodes)
        assert sum(op == "cv-release" for op, *_ in exchanges) == len(nodes)
    else:
        server_app._run_fedavg(grid, cfg, "neural")

    # These checks observe only fixture state, never add an exported node API.
    identity_checks = []
    for node, manifest in zip(nodes, manifests):
        os.environ["DSFLOWER_NODE_SECRET_FILE"] = node["secret"]
        selection = seeding.request_selection(manifest)
        changed = dict(manifest, **{"public-initialisation-manifest-sha256": "e" * 64})
        if "public-initialisation-manifest-sha256" in manifest:
            assert seeding.request_selection(changed) != selection
        alias = dict(manifest, **{"checkpoint-resource-symbol": "another", "checkpoint-url": "file:///alias"})
        assert seeding.request_selection(alias) == selection
        if "cv-contract-sha256" in manifest:
            ids = np.array(["p%d" % index for index in range(64)])
            contract = resampling.cross_validation_contract(2, "patient")
            before = resampling.cross_validation_folds(contract, n_rows=len(ids), unit_ids=ids)
            # Initialization is deliberately not a partition input.
            changed_contract = resampling.cross_validation_contract(changed["cv-folds"], changed["dp-unit"])
            after = resampling.cross_validation_folds(changed_contract, n_rows=len(ids), unit_ids=ids)
            np.testing.assert_array_equal(before, after)
        identity_checks.append(True)
    report = {"nodes": len(nodes), "exchanges": len(exchanges),
              "dp_training_rounds": sum(op in ("train", "cv-train") for op, *_ in exchanges)
              if cfg.get("dp-track") != "validation" else 0,
              "folds_start_from_same_checkpoint": len(first_arrays) == 2,
              "identity_and_partition_checks": all(identity_checks)}
    if plain_vectors:
        standardized = np.abs(np.sum(private_vectors, axis=0) - np.sum(plain_vectors, axis=0)) / np.linalg.norm(metric_sigmas)
        assert np.max(standardized) < 6, "pooled metric deviation exceeds six calibrated standard deviations"
        report["plain_metrics_match"] = True
        report["pooled_noise_within_six_sigma"] = True
    report_file.write_text(json.dumps(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("fixture", "execute"))
    parser.add_argument("--runner", required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--app", type=Path)
    parser.add_argument("--nodes", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, args.runner)
    report = fixture(args.root, args.encoder) if args.operation == "fixture" else execute(args.app, args.nodes, args.report)
    print(json.dumps(report))
