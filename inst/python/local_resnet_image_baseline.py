#!/usr/bin/env python3
"""Centralized ResNet-18 baseline for dsImaging direct-image demos."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18


TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def find_image(workdir: Path, site: Optional[str], sample_id: str) -> Optional[Path]:
    candidates = []
    if site:
        candidates.append(workdir / "sites" / site / "images" / f"{sample_id}.nii.gz")
        candidates.append(workdir / "sites" / site / "images" / f"{sample_id}.nii")
    candidates.extend(
        [
            workdir / "nifti" / "images" / f"{sample_id}.nii.gz",
            workdir / "nifti" / "images" / f"{sample_id}.nii",
            workdir / "images" / f"{sample_id}.nii.gz",
            workdir / "images" / f"{sample_id}.nii",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_rows(workdir: Path, target: str) -> pd.DataFrame:
    meta_files = sorted((workdir / "sites").glob("*/metadata.csv"))
    if not meta_files:
        root_meta = workdir / "metadata.csv"
        if root_meta.exists():
            meta_files = [root_meta]
    if not meta_files:
        raise FileNotFoundError(
            f"No metadata.csv files found under {workdir}/sites/*/"
        )

    frames = []
    for meta in meta_files:
        df = pd.read_csv(meta)
        if "sample_id" not in df.columns:
            raise ValueError(f"{meta} has no sample_id column")
        if target not in df.columns:
            raise ValueError(f"{meta} has no target column {target!r}")
        if "site" not in df.columns:
            df["site"] = meta.parent.name if meta.parent.name != str(workdir) else ""
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    image_paths = []
    for _, row in data.iterrows():
        path = find_image(workdir, str(row.get("site", "")), str(row["sample_id"]))
        if path is None:
            raise FileNotFoundError(f"No image found for sample_id={row['sample_id']}")
        image_paths.append(str(path))
    data["image_path"] = image_paths
    data[target] = data[target].astype(int)
    data = data[data[target].isin([0, 1])].reset_index(drop=True)
    if data.empty:
        raise ValueError("No binary-labelled rows available for baseline")
    return data


def load_volume_slice(path: str) -> Image.Image:
    vol = nib.load(path).get_fdata(dtype=np.float32)
    if vol.ndim == 3:
        arr = vol[:, :, vol.shape[2] // 2]
    elif vol.ndim == 4:
        arr = vol[:, :, vol.shape[2] // 2, 0]
    else:
        arr = vol
    arr = np.asarray(arr, dtype=np.float32)
    if float(arr.max()) > float(arr.min()):
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    arr = arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


class ImageTableDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        image = TRANSFORM(load_volume_slice(row["image_path"]))
        label = torch.tensor(int(row[self.target]), dtype=torch.long)
        return image, label


def build_model(n_classes: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
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
        "n_samples": total,
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.threads)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    workdir = Path(args.workdir)
    data = load_rows(workdir, args.target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageTableDataset(data, args.target)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    model.train()
    for epoch in range(args.epochs):
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
        losses.append(epoch_loss / epoch_n if epoch_n else None)

    metrics = evaluate(model, eval_loader, device)
    output = {
        "mode": "centralized",
        "workdir": str(workdir),
        "target": args.target,
        "n_samples": int(len(data)),
        "class_counts": {
            str(k): int(v) for k, v in data[args.target].value_counts().sort_index().items()
        },
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "device": str(device),
        "train_loss_by_epoch": losses,
        "eval": metrics,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
