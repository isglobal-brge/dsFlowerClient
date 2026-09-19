#!/usr/bin/env python3
"""Verify actual three-node synthetic integration; never synthesize gate evidence."""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
from PIL import Image
import torch
from dsflower_runner import params, segmentation


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def fixture_tensors(prepared, split, encoder, device):
    """Decode the known public fixture independently of runtime subject staging."""
    audit = json.loads((prepared / "audit.json").read_text())
    subjects = [f"s{i:03d}" for i in range(51)]
    sites = [subjects[start:start + 16] for start in (0, 16, 32)]
    require(audit.get("dataset") == "synthetic", "not the synthetic fixture")
    require(split["train"] == subjects[:48] and split["test"] == subjects[48:]
            and split["sites"] == sites, "synthetic subject split changed")
    require(split["seed"] == 20260919 and split.get("variant", "full") == "full",
            "synthetic split pins changed")
    with (prepared / "samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 57 and {r["subject_id"] for r in rows} == set(subjects),
            "synthetic source census changed")
    X = np.zeros((51, segmentation.FEATURE_DIM), np.float32)
    y = np.zeros((51, 2, 128, 128), np.float32)
    yy, xx = np.indices((128, 128))
    encoder.eval()
    before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    with torch.no_grad():
        for i, subject in enumerate(subjects):
            records = [r for r in rows if r["subject_id"] == subject]
            duplicate = i < 48 and i % 16 == 2
            require(len(records) == (3 if duplicate else 1), "synthetic row census changed")
            selected = [r for r in records if r["image_id"] == min(r["image_id"] for r in records)]
            require(len(selected) == (2 if duplicate else 1)
                    and all(r["image_id"] == "a" for r in selected), "image selection changed")
            require(all(r["relative_path"] == subject + ".png" for r in selected),
                    "selected image pairing changed")
            declared_empty = i % 16 == 0
            require(all(r["mask_empty"] == ("TRUE" if declared_empty else "FALSE")
                        for r in selected), "synthetic empty declaration changed")
            with Image.open(Path(audit["image_root"]) / selected[0]["relative_path"]) as image:
                image = np.asarray(image.convert("RGB"), dtype=np.float32)
            require(image.shape == (128, 128, 3), "synthetic image shape changed")
            union = np.zeros((128, 128), bool)
            valid = True
            for record in selected:
                path = Path(audit["mask_root"]) / record["mask_path"]
                if not path.exists():
                    valid = False
                    continue
                with Image.open(path) as mask:
                    pixels = np.asarray(mask)
                require(pixels.shape == (128, 128) and np.isin(pixels, [0, 255]).all(),
                        "synthetic mask vocabulary or geometry changed")
                union |= pixels == 255
            expected_valid = not (i < 48 and i % 16 == 1)
            require(valid == expected_valid, "synthetic invalid subject changed")
            if not valid:
                continue
            expected = (xx - 64) ** 2 + (yy - 64) ** 2 < (16 + i % 8) ** 2
            if declared_empty:
                expected[:] = False
            if duplicate:
                expected[10:20, 10:20] = True
            require(np.array_equal(union, expected), "synthetic mask union changed")
            # Images already have canonical geometry. Normalize each image
            # directly, without the production pair/selection implementation.
            image = image.transpose(2, 0, 1) / 255.0
            image = (image - np.array([.485, .456, .406], np.float32)[:, None, None]) / (
                np.array([.229, .224, .225], np.float32)[:, None, None])
            value = encoder(torch.from_numpy(image)[None].to(device)).flatten(1).cpu().numpy()[0]
            require(value.shape == (segmentation.FEATURE_DIM,) and np.isfinite(value).all(),
                    "synthetic encoder produced invalid features")
            X[i] = np.clip(value, -1e6, 1e6)
            y[i, 0] = union
            y[i, 1] = 1
    require(all(torch.equal(before[k], v) for k, v in encoder.state_dict().items()),
            "encoder state changed")
    expected_sites = []
    for site, start in enumerate((0, 16, 32), 1):
        sx, sy = X[start:start + 16], y[start:start + 16]
        require(int(sy[:, 1, 0, 0].sum()) == 15 and not sx[1].any() and not sy[1].any(),
                "invalid subject was dropped or contributed data")
        expected_sites.append(dict(site=site, subjects=16, source_rows=18, valid_subjects=15,
                                   declared_empty_subjects=1, features_sha256=tensor_hash(sx),
                                   targets_sha256=tensor_hash(sy)))
    return X, y, expected_sites


def independent_accounting(mechanism, epsilon):
    from opacus.accountants import PRVAccountant
    sigma = mechanism["noise_multiplier"]
    require(math.isfinite(sigma) and sigma > 0, "noise multiplier invalid")
    accountant = PRVAccountant()
    # Reconstruct all four logical steps, not the two-step per-round object.
    for _ in range(4):
        accountant.step(noise_multiplier=sigma, sample_rate=.5)
    delta_add_remove = 1e-5 / (1 + math.exp(epsilon / 2))
    epsilon_add_remove = accountant.get_epsilon(delta=delta_add_remove)
    epsilon_replace_one = 2 * epsilon_add_remove
    delta_replace_one = delta_add_remove * (1 + math.exp(epsilon_add_remove))
    require(epsilon_replace_one <= epsilon and delta_replace_one <= 1e-5,
            "independent full-horizon accountant exceeds budget")
    return dict(accountant="PRVAccountant", total_steps=4,
                epsilon_replace_one=epsilon_replace_one, delta_replace_one=delta_replace_one)


def verify_captures(records, expected_sites, epsilon):
    require(len(records) == 6, "expected six actual node-round captures")
    expected = {(s["features_sha256"], s["targets_sha256"]): s for s in expected_sites}
    require(len(expected) == 3, "synthetic sites must have distinct tensor hashes")
    seen, results = Counter(), []
    geometry = dict(adjacency="replace_one", clipping_norm=1., accounting_population=16,
                    steps_per_epoch=2, sample_rate=.5, expected_batch_size=8,
                    total_epochs=2, total_steps=4)
    for record in records:
        key = (record["features_sha256"], record["targets_sha256"])
        require(key in expected and record["round"] in (1, 2), "unexpected site or round")
        require(record.get("public_fixture_only") is True, "capture is not a public fixture")
        require(record.get("source_rows") == 18, "actual source census was not preserved")
        mechanism = record["mechanism"]
        require(all(mechanism.get(k) == v for k, v in geometry.items()), "accountant geometry changed")
        require(record["observed_round_steps"] == 2 and record["accountant_history"] ==
                [[mechanism["noise_multiplier"], .5, 2]], "actual logical step transcript changed")
        require(record.get("accountant_type") in ("PRVAccountant", "RDPAccountant"),
                "unknown actual accountant")
        seen[(expected[key]["site"], record["round"])] += 1
    require(seen == Counter({(s, r): 1 for s in (1, 2, 3) for r in (1, 2)}),
            "missing or duplicate site-round capture")
    accounting_cache = {}
    for site in expected_sites:
        pair = [r for r in records if r["features_sha256"] == site["features_sha256"]]
        require(pair[0]["mechanism"] == pair[1]["mechanism"], "site accounting changed between rounds")
        sigma = pair[0]["mechanism"]["noise_multiplier"]
        if sigma not in accounting_cache:
            accounting_cache[sigma] = independent_accounting(pair[0]["mechanism"], epsilon)
        results.append(dict(site=site["site"], mechanism=pair[0]["mechanism"],
                            independent_accounting=accounting_cache[sigma]))
    return results


def verify(prepared, run, decoder="current"):
    status_path = run / "federation-status.json"
    status = json.loads(status_path.read_text())
    require(status.get("status") == "predicted_pending_public_metric_summary"
            and status.get("cleanup_ok") is True and status.get("synthetic") is True
            and status.get("dataset") == "synthetic", "actual synthetic federation did not complete cleanly")
    require(isinstance(status.get("output_dir"), str), "artifact output directory missing")
    artifact = Path(status["output_dir"])
    artifact_root = (run / "artifact").resolve()
    require(artifact.is_absolute() and artifact.is_dir()
            and (artifact.resolve() == artifact_root or artifact_root in artifact.resolve().parents),
            "artifact output directory is outside this run")
    require(status.get("epsilon") in (1, 4, 8), "synthetic budget changed")
    split_path = prepared / "split-20260919.json"
    require(status.get("split_sha256") == sha256(split_path), "source split digest changed")
    split = json.loads((run / "effective-split.json").read_text())
    capture = run / "public-capture"
    initial_meta = json.loads((capture / "public-initial.json").read_text())
    cfg = initial_meta["config"]
    segmentation.validate_config(cfg)
    require(json.loads(base64.b64decode(cfg["model-spec-b64"])) == segmentation.decoder_spec(decoder),
            "synthetic decoder differs from requested candidate")
    parameter_tensors = 2 if decoder == "pointwise" else 6
    require(all(cfg.get(k) == v for k, v in {"batch-size": 8, "local-epochs": 1,
            "num-server-rounds": 2, "learning-rate": .01, "segmentation-alpha": .5}.items()),
            "synthetic schedule changed")
    require(initial_meta["seed"] == split["seed"] == status["seed"], "initialization seed changed")
    initial_npz = np.load(capture / "public-initial-arrays.npz", allow_pickle=False)
    require(set(initial_npz.files) == {str(i) for i in range(parameter_tensors)}, "initial parameter count changed")
    initial = [initial_npz[str(i)] for i in range(parameter_tensors)]
    require([tensor_hash(a) for a in initial] == initial_meta["tensor_sha256"], "initial arrays changed")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    encoder, device = segmentation.prepare_encoder(cfg)
    X, _, expected_sites = fixture_tensors(prepared, split, encoder, device)
    records = [json.loads(path.read_text()) for path in sorted(capture.glob("accountant-*.json"))]
    accounting = verify_captures(records, expected_sites, status["epsilon"])
    metadata = json.loads((artifact / "metadata.json").read_text())
    require(metadata.get("status") == "success" and metadata.get("available") is True
            and metadata.get("n_clients") == 3 and metadata.get("model") == "pytorch_resnet18_segmentation"
            and metadata.get("privacy") == "server-enforced-dp", "released artifact metadata invalid")
    history = json.loads((artifact / "history.json").read_text())
    require(isinstance(history, list) and len(history) == 2
            and [r.get("round") for r in history] == [1, 2]
            and all(r.get("n_failures") == 0 and r.get("available") is True for r in history),
            "actual two-round availability history invalid")
    require(all(set(r) <= {"round", "n_failures", "available", "n_examples"}
                and r.get("n_examples", 1) in (1, 3) for r in history), "unexpected history disclosures")
    model_path = artifact / "model.pt"
    require(sha256(model_path) == status.get("model_sha256"), "released artifact digest changed")
    model = params.load_user_model(cfg, segmentation.FEATURE_DIM, "segmentation_bce_dice").cpu()
    require(all(a.shape == tuple(p.shape) and a.dtype == np.float32 and np.isfinite(a).all()
                for a, p in zip(initial, model.parameters())), "initial tensor geometry invalid")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    require(not list(model.buffers()) and len(state) == parameter_tensors
            and set(state) == {n for n, _ in model.named_parameters()}
            and all(torch.isfinite(t).all().item() for t in state.values()),
            "release must contain exact finite trained decoder parameters without buffers")
    model.load_state_dict(state, strict=True)
    require(any(not np.array_equal(a, b) for a, b in zip(initial, params.get_torch_params(model))),
            "release equals untrained initial arrays")
    model.eval()
    with torch.no_grad():
        probability = model(torch.from_numpy(X[48:])).sigmoid().numpy().reshape(3, -1)
    actual = np.loadtxt(run / "public-probabilities.csv", delimiter=",")
    require(actual.shape == (3, 16384) and np.isfinite(actual).all()
            and np.allclose(actual, probability, rtol=0, atol=1e-12), "persisted local predictions differ")
    return dict(schema="dsflower-segmentation-synthetic-integration-v1", status="passed",
                gate="segmentation_6_1_7", scoring_performed=False, sites=expected_sites,
                rounds=2, unique_node_round_captures=6, accounting=accounting,
                model_sha256=sha256(model_path), split_sha256=sha256(split_path),
                source_rows_sha256=sha256(prepared / "samples.csv"),
                federation_status_sha256=sha256(status_path), cleanup_ok=True,
                decoder_parameter_tensors=parameter_tensors, decoder_parameter_count=sum(p.numel() for p in model.parameters()),
                reloaded_predictions_match=True, python=platform.python_version(),
                torch=torch.__version__, encoder_device=str(device), decoder_device="cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decoder", choices=("current", "narrow", "pointwise"), default="current")
    args = parser.parse_args()
    result = verify(args.prepared, args.run, args.decoder)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
