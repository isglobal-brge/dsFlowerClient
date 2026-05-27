#!/usr/bin/env python3
"""Train/evaluate ResNet-18 on image-table manifests used by dsFlower demos."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--path-col", default="relative_path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def safe_join(root: str, rel_path: str) -> str:
    rel_path = str(rel_path)
    if rel_path.startswith("/") or ".." in rel_path.split(os.sep):
        raise ValueError(f"invalid relative image path: {rel_path}")
    return os.path.join(root, rel_path)


class ImageTable(Dataset):
    def __init__(self, samples: pd.DataFrame, image_root: str, target_col: str,
                 path_col: str, image_size: int):
        self.samples = samples.reset_index(drop=True)
        self.image_root = image_root
        self.target_col = target_col
        self.path_col = path_col
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        path = safe_join(self.image_root, row[self.path_col])
        image = Image.open(path).convert("RGB")
        label = torch.tensor(int(row[self.target_col]), dtype=torch.long)
        return self.transform(image), label


def build_model(n_classes: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    if ckpt_keys and all(str(k).isdigit() for k in ckpt_keys):
        if len(ckpt_keys) != len(model_keys):
            raise ValueError(
                f"checkpoint has {len(ckpt_keys)} tensors; model expects {len(model_keys)}"
            )
        ordered = OrderedDict()
        for i, key in enumerate(model_keys):
            ordered[key] = state_dict[str(i)]
        state_dict = ordered

    model.load_state_dict(state_dict, strict=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * int(labels.numel())
            total += int(labels.numel())
            correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / total if total else None,
        "accuracy": correct / total if total else None,
        "n_samples": int(total),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.threads)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = pd.read_csv(args.samples)
    dataset = ImageTable(
        samples=samples,
        image_root=args.image_root,
        target_col=args.target_col,
        path_col=args.path_col,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss_by_epoch = []
    mode = "centralized"
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
        mode = "checkpoint"
    else:
        model.train()
        for _ in range(max(1, args.epochs)):
            epoch_loss = 0.0
            epoch_n = 0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * int(labels.numel())
                epoch_n += int(labels.numel())
            train_loss_by_epoch.append(epoch_loss / epoch_n if epoch_n else None)

    result = {
        "mode": mode,
        "checkpoint": args.checkpoint,
        "image_root": args.image_root,
        "n_samples": len(dataset),
        "class_counts": {
            str(k): int(v)
            for k, v in samples[args.target_col].value_counts().sort_index().items()
        },
        "epochs": int(max(1, args.epochs)),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "image_size": int(args.image_size),
        "device": str(device),
        "train_loss_by_epoch": train_loss_by_epoch,
        "eval": evaluate(model, eval_loader, device),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
