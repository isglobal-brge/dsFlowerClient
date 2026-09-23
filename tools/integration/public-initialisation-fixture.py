#!/usr/bin/env python3
"""Synthetic public bundle and direct trusted-runner DSLite integration driver.

Uses an explicitly supplied, full-SHA verified encoder. Never downloads weights.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture(root, encoder):
    import numpy as np
    from PIL import Image
    from dsflower_runner import segmentation as seg, segmentation_checkpoints as checkpoints
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / "bundle"
    bundle.mkdir()
    encoder_data = encoder.read_bytes()
    assert sha(encoder_data) == seg.CHECKPOINT_SHA256
    assert len(encoder_data) == 46830571

    def artifact(name, data):
        (bundle / name).write_bytes(data)
        return {"file": name, "size_bytes": len(data), "sha256": sha(data)}

    shapes = [(8, 128, 3, 3), (8,), (4, 8, 3, 3), (4,), (1, 4, 1, 1), (1,)]
    arrays = [np.full(shape, (i + 1) / 100, dtype=np.float32)
              for i, shape in enumerate(shapes)]
    stream = io.BytesIO()
    np.savez(stream, **{str(i): value for i, value in enumerate(arrays)})
    checkpoint = artifact("checkpoint.npz", stream.getvalue())
    encoder_record = artifact("encoder.pth", encoder_data)
    evidence = {key: artifact(key + ".txt", ("synthetic public fixture " + key).encode())
                for key in ("protocol", "audit", "licence", "mirror_metadata")}
    dataset = {"dataset": "dsFlower synthetic integration", "release": "v1",
               "sha256": "a" * 64, "dataset_url": "https://example.invalid/synthetic",
               "attribution": "dsFlower tests", "licence": "synthetic test fixture",
               "licence_sha256": evidence["licence"]["sha256"],
               "metadata_sha256": evidence["mirror_metadata"]["sha256"]}
    evidence["provenance"] = artifact("provenance.json", json.dumps(dataset).encode())
    original = {"dataset": dataset["dataset"], "dataset_sha256": dataset["sha256"],
                "decoder": "narrow", "privacy": "public_nonprivate",
                "checkpoint_sha256": checkpoint["sha256"],
                "tensor_sha256": [sha(value.tobytes()) for value in arrays],
                "encoder_sha256": seg.CHECKPOINT_SHA256,
                **{key + "_sha256": evidence[key]["sha256"]
                   for key in ("protocol", "provenance", "audit")}}
    evidence["original_manifest"] = artifact("original.json", json.dumps(original).encode())
    manifest = {"schema_version": checkpoints.SCHEMA, "checkpoint_id": "synthetic-dslite-v1",
                "model_id": "pytorch_resnet18_segmentation", "decoder": "narrow",
                "feature_contract": seg.PROFILE, "encoder_sha256": seg.CHECKPOINT_SHA256,
                "dataset": dataset,
                "licence": {"declaration": dataset["licence"], "scope": "synthetic tests only"},
                "checkpoint": checkpoint, "encoder": encoder_record, "evidence": evidence,
                "role": "segmentation_decoder",
                "decoder_spec_sha256": sha(checkpoints._canonical(seg.decoder_spec("narrow"))),
                "pretraining_protocol_sha256": evidence["protocol"]["sha256"],
                "creation": {"creator": "dsFlower synthetic tests", "created_at": "2026-09-23"},
                "tensors": [{"name": str(i), "shape": list(value.shape), "dtype": "float32",
                             "sha256": sha(value.tobytes())} for i, value in enumerate(arrays)]}
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = checkpoints.pack_bundle(bundle, root / "bundle.zip")
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for kind in ("images", "masks"):
        (root / kind).mkdir()
    for i in range(2):
        pixels = np.full((128, 128, 3), 60 + i * 60, dtype=np.uint8)
        pixels[32:96, 32:96] += 30
        Image.fromarray(pixels).save(root / "images" / (str(i) + ".png"))
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[40:80, 40:80] = 255
        Image.fromarray(mask).save(root / "masks" / (str(i) + ".png"))
    return {"bundle_sha256": sha((root / "bundle.zip").read_bytes()),
            "manifest_sha256": summary["provenance"]["manifest_sha256"]}


def train(root, manifest_directory):
    import numpy as np
    import torch
    from flwr.common import ArrayRecord, ConfigRecord, Message, RecordDict
    from dsflower_runner import client_app, segmentation_checkpoints as checkpoints, seeding, segmentation, task
    torch.set_num_threads(1)
    manifest = json.loads((manifest_directory / "manifest.json").read_text())
    local = checkpoints.client_payload(root / "bundle")
    arrays = checkpoints.server_initialization({
        "segmentation-decoder-init": manifest["segmentation-decoder-init"],
        "model-spec-b64": manifest["model-spec-b64"],
        checkpoints.TRANSPORT_KEY: __import__("base64").b64encode(
            json.dumps(local).encode()).decode()})
    initial = [value.copy() for value in arrays]
    public_keys = set(segmentation.PIN_KEYS) | {
        "dp-track", "task-type", "data_type", "loss-name", "model-spec-b64", "backbone",
        "vision-extractor-profile", "num-features", "image-size", "num-classes",
        "segmentation-decoder-init", "num-server-rounds", "batch-size", "local-epochs",
        "learning-rate", "optimizer-name", "scheduler-name"}
    config = {key: manifest[key] for key in public_keys if key in manifest}
    config["data-kind"] = "image"
    context = SimpleNamespace(run_config=config,
        node_config={"manifest-dir": str(manifest_directory)}, state=RecordDict())
    effective = task.load_pinned_run_config(context)
    checkpoints.verify_node_checkpoint(effective, manifest)
    segmentation.prepare_encoder(effective)
    rounds = []
    for index in (1, 2):
        message = Message(content=RecordDict({
            "arrays": ArrayRecord(numpy_ndarrays=arrays),
            "config": ConfigRecord({"server-round": index})}),
            dst_node_id=1, message_type="train", group_id="public-init-integration-%d" % index)
        response = client_app.train(message, context)
        assert not response.has_error(), response.error
        metrics = dict(response.content["metrics"])
        assert not metrics.get("public-preflight-unavailable", 0), (index, metrics)
        assert not metrics.get("execution-unavailable", 0), metrics
        arrays = response.content["arrays"].to_numpy_ndarrays()
        release = json.loads((manifest_directory /
            ("segmentation-public-init-release-%06d.json" % index)).read_text())
        assert release["initialisation"] == manifest["initialisation"]
        assert release["public_initialisation"] == manifest[checkpoints.PROVENANCE_KEY]
        assert release["public_initialisation_policy"] == "analyst_or_resource"
        assert release["round"] == index
        assert [x["sha256"] for x in release["released_tensors"]] == [sha(x.tobytes()) for x in arrays]
        rounds.append({"round": index, "initialisation": release["initialisation"],
                       "manifest_sha256": release["public_initialisation"]["provenance"]["manifest_sha256"]})
    assert any(not np.array_equal(a, b) for a, b in zip(initial, arrays))
    selected = seeding.request_selection(manifest)
    changed = dict(manifest)
    changed[checkpoints.MANIFEST_KEY] = "e" * 64
    assert seeding.request_selection(changed) != selected
    alias = dict(manifest, **{"checkpoint-resource-symbol": "other", "checkpoint-url": "file:///other"})
    assert seeding.request_selection(alias) == selected
    return {"rounds": rounds, "identity_changed_with_digest": True,
            "identity_unchanged_with_locator": True, "private_training_executed": True}


def serve(root, certificate, key):
    import functools
    import http.server
    import ssl
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
        functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root)))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    (root / "https-port.json").write_text(json.dumps(server.server_address[1]))
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("fixture", "train", "serve"))
    parser.add_argument("--runner", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--key", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, args.runner)
    if args.operation == "serve":
        serve(args.root, args.certificate, args.key)
        sys.exit(0)
    result = (fixture(args.root, args.encoder) if args.operation == "fixture"
              else train(args.root, args.manifest))
    print(json.dumps(result))
