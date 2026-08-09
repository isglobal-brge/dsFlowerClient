#!/usr/bin/env python3
"""Centralized baselines for dsFlower vision-model validation."""

import argparse
import json
import sys

import pandas as pd
from PIL import Image


def _fail(message):
    print(message, file=sys.stderr)
    return 2


def _as_int(value, default):
    if value is None:
        return default
    return int(value)


def _as_float(value, default):
    if value is None:
        return default
    return float(value)


def _load_params(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _torch_import():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import densenet121, resnet18
    return torch, nn, DataLoader, Dataset, transforms, densenet121, resnet18


def _safe_join(root, rel_path):
    rel_path = str(rel_path)
    if ".." in rel_path or rel_path.startswith("/"):
        raise ValueError(f"Invalid relative path: {rel_path}")
    return f"{root.rstrip('/')}/{rel_path}"


class _ImageClassificationDataset:
    def __init__(self, dataset_base, samples, image_root, target_col,
                 path_col, image_size, transforms):
        Dataset = dataset_base

        class ImageClassificationDataset(Dataset):
            def __init__(self):
                self.samples = samples.reset_index(drop=True)
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                row = self.samples.iloc[idx]
                image = Image.open(_safe_join(image_root, row[path_col])).convert("RGB")
                label = int(row[target_col])
                return self.transform(image), label

        self.dataset = ImageClassificationDataset()


def _build_classifier(method, n_classes, densenet121, resnet18, nn):
    if method == "pytorch_resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model
    if method == "pytorch_densenet121":
        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
        return model
    raise ValueError(f"Unsupported classifier method: {method}")


def _run_classifier(args, params, torch, nn, DataLoader, Dataset,
                    transforms, densenet121, resnet18):
    torch.manual_seed(args.seed)
    torch.set_num_threads(_as_int(params.get("threads"), 1))
    image_size = _as_int(params.get("image_size"), 224)
    batch_size = _as_int(params.get("batch_size"), 4)
    epochs = max(1, args.rounds * _as_int(params.get("local_epochs"), 1))
    n_classes = _as_int(params.get("n_classes"), 2)

    samples = pd.read_csv(args.samples)
    dataset = _ImageClassificationDataset(
        Dataset, samples, args.image_root, args.target_col,
        args.path_col, image_size, transforms
    ).dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_classifier(args.method, n_classes, densenet121, resnet18, nn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_as_float(params.get("learning_rate"), 0.001))
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)
    return {
        "metric_name": "cross_entropy",
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "n_samples": int(total),
    }


def run(args):
    params = _load_params(args.model_params_file)
    torch, nn, DataLoader, Dataset, transforms, densenet121, resnet18 = _torch_import()
    if args.method in ("pytorch_resnet18", "pytorch_densenet121"):
        result = _run_classifier(
            args, params, torch, nn, DataLoader, Dataset,
            transforms, densenet121, resnet18
        )
    else:
        raise ValueError(f"Unsupported vision method: {args.method}")

    result.update({
        "method": args.method,
        "status": "ok",
        "image_size": _as_int(params.get("image_size"), 224),
    })
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--path-col", default="relative_path")
    parser.add_argument("--model-params-file", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
