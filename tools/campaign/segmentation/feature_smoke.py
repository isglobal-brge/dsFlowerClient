#!/usr/bin/env python3
"""Public release/encoder smoke, no prediction or utility scoring."""
import argparse
import base64
import gc
import hashlib
import json
import os
import platform
import torchvision
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from dsflower_runner import segmentation, params


def config():
    return {"task-type": "segmentation", "loss-name": "segmentation_bce_dice",
            "data-kind": "image", "backbone": segmentation.BACKBONE,
            "vision-extractor-profile": segmentation.PROFILE,
            "num-features": segmentation.FEATURE_DIM, "image-size": 128,
            "num-classes": 2, "segmentation-selection": segmentation.SELECTION,
            "segmentation-preprocessing": segmentation.PREPROCESSING,
            "segmentation-checkpoint-sha256": segmentation.CHECKPOINT_SHA256,
            "segmentation-output-shape": "1,128,128", "segmentation-alpha": .5,
            "segmentation-smooth": 1., "mask-vocabulary": "0,255",
            "model-spec-b64": base64.b64encode(json.dumps(segmentation.decoder_spec()).encode()).decode(),
            "batch-size": 16, "local-epochs": 2, "num-server-rounds": 5,
            "learning-rate": .01, "optimizer-name": "sgd", "scheduler-name": "none"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cfg = config()
    encoder, device = segmentation.prepare_encoder(cfg)
    before = {key: value.clone() for key, value in encoder.state_dict().items()}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("breast", "busbra"):
        start = time.monotonic()
        prepared = args.prepared / name
        audit = json.loads((prepared / "audit.json").read_text())
        import pandas as pd
        frame = pd.read_csv(prepared / "samples.csv")
        output = args.out / name
        output.mkdir(exist_ok=True)
        frame.to_csv(output / "samples.csv", index=False)
        manifest = dict(cfg, **{"data_type": "image", "data_format": "csv",
            "samples_file": "samples.csv", "n_samples": len(frame),
            "n_units": frame.subject_id.nunique(), "dp-unit": "patient",
            "patient_column": "subject_id", "patient-id-canonicalization": "trim-utf8-v2",
            "sample_id_col": "image_id",
            "mask_empty_col": "mask_empty", "target_column": "mask_path",
            "assets": {"images": {"root": audit["image_root"], "path_col": "relative_path"},
                       "masks": {"root": audit["mask_root"], "path_col": "mask_path"}}})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        context = SimpleNamespace(node_config={"manifest-dir": str(output)})
        X, y, subjects, n_staged = segmentation.load_subject_tensors(context, cfg, encoder, device)
        assert len(X) == audit["public_subjects"]
        assert y.shape == (len(X), 2, 128, 128)
        assert np.isfinite(X).all() and np.isfinite(y).all()
        # Audited public fixtures: any invalid retained subject is a pipeline defect.
        assert (y[:, 1] == 1).all(), "public fixture unexpectedly totalized"
        assert all(torch.equal(value, before[key]) for key, value in encoder.state_dict().items())
        assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
        np.savez(output / "public-subject-tensors.npz", X=X, y=y, subjects=np.asarray(subjects))
        result = {"schema": "dsflower-segmentation-public-feature-smoke-v1",
                  "status": "passed", "dataset": name, "device": str(device),
                  "torch": torch.__version__, "torchvision": torchvision.__version__,
                  "python": platform.python_version(), "cuda_runtime": torch.version.cuda,
                  "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                  "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                  "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "profile": segmentation.PROFILE,
                  "checkpoint_sha256": segmentation.CHECKPOINT_SHA256,
                  "subject_n": len(X), "source_rows": n_staged,
                  "features_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
                  "targets_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                  "subject_order_sha256": hashlib.sha256("\n".join(subjects).encode()).hexdigest(),
                  "encoder_state_unchanged": True, "scoring_performed": False,
                  "elapsed_s": time.monotonic() - start,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
        (output / "feature-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


    # A complete Poisson draw can contain every subject on a public site. This
    # synthetic bound exceeds the protocol's largest actual per-site N (284).
    from opacus import GradSampleModule
    del X, y, encoder, before
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = params.load_user_model(cfg, segmentation.FEATURE_DIM, "segmentation_bce_dice").to(device)
    wrapped = GradSampleModule(model, loss_reduction="mean")
    batch = 285
    x = torch.zeros((batch, segmentation.FEATURE_DIM), device=device)
    target = torch.zeros((batch, 2, 128, 128), device=device)
    target[:, 1] = 1
    loss = segmentation.loss_factory(cfg)(wrapped(x), target)
    loss.backward()
    for name, parameter in wrapped.named_parameters():
        assert parameter.grad_sample.shape[0] == batch, name
        assert torch.isfinite(parameter.grad_sample).all(), name
    if device.type == "cuda":
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    reserved = torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
    total = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None
    # Reserve 1 GiB per node for weights/features/framework overhead above decoder peak.
    conservative = 3 * (peak + 1024 ** 3)
    result = {"schema": "dsflower-segmentation-public-memory-smoke-v1", "status": "passed",
              "synthetic_subject_batch": batch, "all_parameter_grad_samples_finite": True,
              "optimizer_steps": 0, "scoring_performed": False,
              "peak_cuda_allocated_bytes": peak, "peak_cuda_reserved_bytes": reserved,
              "three_node_peak_plus_1gib_each": conservative,
              "device_total_bytes": total, "three_node_bound_fits": total is None or conservative < total,
              "elapsed_s": time.monotonic() - started}
    assert result["three_node_bound_fits"], "whole-draw memory bound requires further review"
    (args.out / "memory-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
