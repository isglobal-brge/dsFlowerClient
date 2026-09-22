#!/usr/bin/env python3
"""One fixed test-scoring pass, after every requested model and twin is trained."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from dsflower_runner import params, validation, vision


def metrics(y, probability):
    y, p = np.asarray(y), np.asarray(probability, dtype=float)
    assert len(y) == len(p) and set(y) == {0, 1}
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    positive, negative = p[y == 1], p[y == 0]
    differences = positive[:, None] - negative[None, :]
    auc = ((differences > 0).sum() + .5 * (differences == 0).sum()) / differences.size
    safe = np.clip(p, 1e-15, 1-1e-15)
    return dict(auc=float(auc), accuracy=float(((p >= .5) == y).mean()),
                brier=float(np.mean((p-y)**2)),
                log_loss=float(-np.mean(y*np.log(safe)+(1-y)*np.log1p(-safe))))


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(tools, artifact, paths_file, predictions):
    started = time.monotonic()
    subprocess.run(["Rscript", str(tools / "predict.R"), str(artifact),
                    str(paths_file), str(predictions)], check=True)
    return time.monotonic() - started


def main(root, epsilons):
    tools = Path(__file__).resolve().parent
    protocol = read_json(tools / "protocol.json")
    lock = root / "scoring-lock.json"
    assert not lock.exists(), "Scoring was already started; never change or repeat a scored configuration"
    frozen = {}
    federated_artifacts = {}
    for epsilon in epsilons:
        for seed in protocol["seeds"]:
            run = root / "runs" / f"pytorch_resnet18-eps{epsilon}-seed{seed}"
            federation = read_json(run / "federation-status.json")
            assert federation["status"] == "trained_unscored"
            artifact = Path(federation["output_dir"])
            assert sha(artifact / "model.pt") == federation["model_sha256"]
            federated_artifacts[epsilon, seed] = artifact
            twins = read_json(run / "twins-status.json")
            assert all(v["status"] == "trained_unscored" for v in twins.values())
            for path in [artifact / "model.pt", artifact / "metadata.json",
                         run / "public-capture/public-initial.json", run / "twins-status.json",
                         *[Path(v["artifact"]) for v in twins.values()]]:
                frozen[str(path)] = sha(path)
    with lock.open("x") as stream:
        json.dump(dict(started_at=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(tools / "protocol.json"), artifacts=frozen, epsilon_order=epsilons,
            scoring_driver_sha256=sha(Path(__file__)), prediction_driver_sha256=sha(tools / "predict.R"),
            configuration_changes_after_scoring_forbidden=True), stream, indent=2)
        stream.write("\n")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    encoder, size, is3d, device = vision.prepare_backbone("resnet18",
        protocol["model"]["extractor_profile"], 512, 224)
    for seed in protocol["seeds"]:
        split = read_json(root / "prepared/busbra" / f"split-{seed}.json")
        test_ids = set(split["test"])
        with (root / "data/busbra/BUSBRA/bus_data.csv").open() as stream:
            rows = [row for row in csv.DictReader(stream) if "busbra:"+str(int(row["Case"])) in test_ids]
        rows.sort(key=lambda row: row["ID"])
        y = np.array([protocol["class_mapping"][row["Pathology"].lower()] for row in rows])
        paths = [str(root / "data/busbra/BUSBRA/Images" / (row["ID"]+".png")) for row in rows]
        directory = root / "scores" / str(seed)
        directory.mkdir(parents=True, exist_ok=False)
        paths_file = directory / "paths.json"
        paths_file.write_text(json.dumps(paths))
        # The canonical local predictor embeds images in outer batches of 1024 (internally bounded to 32).
        features = np.concatenate([vision.extract_features_from_paths(encoder, paths[i:i+1024], size, is3d, device)
                                   for i in range(0, len(paths), 1024)])
        # Independent, fixed models; each canonical predictor runs exactly once.
        # Keep the three epsilon predictions in this foreground scoring job.
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {epsilon: pool.submit(predict, tools, federated_artifacts[epsilon, seed],
                        paths_file, directory / f"federated-eps{epsilon}.csv")
                    for epsilon in epsilons}
            prediction_seconds = {epsilon: job.result() for epsilon, job in jobs.items()}
        central_metric = None
        for epsilon in epsilons:
            started = time.monotonic()
            run = root / "runs" / f"pytorch_resnet18-eps{epsilon}-seed{seed}"
            cfg = read_json(run / "public-capture/public-initial.json")["config"]
            twins = read_json(run / "twins-status.json")
            predictions = directory / f"federated-eps{epsilon}.csv"
            with predictions.open() as stream:
                reader = csv.DictReader(stream)
                assert reader.fieldnames == ["benign", "malignant"], reader.fieldnames
                fed_p = np.array([float(row["malignant"]) for row in reader])
            results = dict(federated_dp=metrics(y, fed_p))
            for branch in ("central", "pooled_dp"):
                if branch == "central" and central_metric is not None:
                    results[branch] = central_metric
                    continue
                data = np.load(twins[branch]["artifact"])
                model = params.load_user_model(cfg, 512, "cross_entropy")
                params.set_torch_params(model, [data[str(i)] for i in range(len(data.files))])
                probability = np.asarray(validation.neural_predictions(model, features, "cross_entropy"))[:, 1]
                results[branch] = metrics(y, probability)
                if branch == "central":
                    central_metric = results[branch]
            prevalence = twins["central"]["training_prevalence"]
            results["trivial"] = metrics(y, np.repeat(prevalence, len(y)))
            majority_rate = float(max(np.mean(y), 1-np.mean(y)))
            record = dict(status="scored", epsilon=epsilon, seed=seed, metrics=results,
                n_train_patients=852, n_test_patients=212, n_test_images=len(y),
                test_prevalence=float(np.mean(y)), test_majority_rate=majority_rate,
                training_prevalence=prevalence, gap=results["federated_dp"]["auc"]-results["central"]["auc"],
                acceptance_diagnostic=(results["federated_dp"]["auc"] > .5 and
                                       results["federated_dp"]["accuracy"] > majority_rate),
                scoring_elapsed_s=prediction_seconds[epsilon] + time.monotonic()-started,
                prediction_elapsed_s=prediction_seconds[epsilon],
                prediction_execution="Three epsilon models concurrently per seed, each scored once after artifact lock.",
                test_prediction_sha256=sha(predictions))
            (run / "scores.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epsilons", type=int, nargs="+", default=[1, 8, 4])
    args = parser.parse_args()
    main(args.root, args.epsilons)
