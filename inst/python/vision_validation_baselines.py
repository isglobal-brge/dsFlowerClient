#!/usr/bin/env python3
"""Centralized baselines for dsFlower vision-template validation."""

import argparse
import json
import sys

import numpy as np
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


class _SegmentationDataset:
    def __init__(self, dataset_base, samples, image_root, target_col,
                 path_col, mask_path_col, image_size, transforms):
        Dataset = dataset_base

        class SegmentationDataset(Dataset):
            def __init__(self):
                self.samples = samples.reset_index(drop=True)
                self.image_transform = transforms.Compose([
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
                mask = Image.open(_safe_join(image_root, row[mask_path_col])).convert("L")
                mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
                mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
                mask_arr = mask_arr[None, :, :]
                return self.image_transform(image), mask_arr

        self.dataset = SegmentationDataset()


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


class _DoubleConv:
    @staticmethod
    def build(nn, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


def _build_unet(torch, nn, n_classes=1, in_channels=3, base_channels=8):
    class UNet2D(nn.Module):
        def __init__(self):
            super().__init__()
            c1 = int(base_channels)
            c2 = c1 * 2
            c3 = c1 * 4
            c4 = c1 * 8
            cb = c1 * 16
            self.enc1 = _DoubleConv.build(nn, in_channels, c1)
            self.enc2 = _DoubleConv.build(nn, c1, c2)
            self.enc3 = _DoubleConv.build(nn, c2, c3)
            self.enc4 = _DoubleConv.build(nn, c3, c4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _DoubleConv.build(nn, c4, cb)
            self.up4 = nn.ConvTranspose2d(cb, c4, 2, stride=2)
            self.dec4 = _DoubleConv.build(nn, c4 * 2, c4)
            self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
            self.dec3 = _DoubleConv.build(nn, c3 * 2, c3)
            self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
            self.dec2 = _DoubleConv.build(nn, c2 * 2, c2)
            self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
            self.dec1 = _DoubleConv.build(nn, c1 * 2, c1)
            self.final = nn.Conv2d(c1, n_classes, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.final(d1)

    return UNet2D()


def _dice_bce_loss(torch, nn, logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    smooth = 1e-6
    intersection = (probs * targets).sum()
    dice_loss = 1 - (2 * intersection + smooth) / (
        probs.sum() + targets.sum() + smooth
    )
    return bce + dice_loss


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


def _run_unet(args, params, torch, nn, DataLoader, Dataset, transforms):
    torch.manual_seed(args.seed)
    torch.set_num_threads(_as_int(params.get("threads"), 1))
    image_size = _as_int(params.get("image_size"), 224)
    batch_size = _as_int(params.get("batch_size"), 1)
    epochs = max(1, args.rounds * _as_int(params.get("local_epochs"), 1))
    base_channels = _as_int(params.get("base_channels"), 64)
    n_classes = _as_int(params.get("n_classes"), 1)

    samples = pd.read_csv(args.samples)
    dataset = _SegmentationDataset(
        Dataset, samples, args.image_root, args.target_col,
        args.path_col, args.mask_path_col, image_size, transforms
    ).dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_unet(torch, nn, n_classes=n_classes, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_as_float(params.get("learning_rate"), 0.001))

    model.train()
    for _ in range(epochs):
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss = _dice_bce_loss(torch, nn, model(images), masks)
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss = 0.0
    total = 0
    dice_scores = []
    with torch.no_grad():
        for images, masks in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            total_loss += float(_dice_bce_loss(torch, nn, logits, masks).item()) * len(masks)
            pred = (torch.sigmoid(logits) > 0.5).float()
            intersection = (pred * masks).sum(dim=(1, 2, 3))
            denom = pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = ((2 * intersection + 1e-6) / (denom + 1e-6)).detach().cpu().numpy()
            dice_scores.extend(dice.tolist())
            total += len(masks)
    return {
        "metric_name": "dice_bce_loss",
        "loss": total_loss / max(total, 1),
        "dice": float(np.mean(dice_scores)) if dice_scores else None,
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
    elif args.method == "pytorch_unet2d":
        result = _run_unet(args, params, torch, nn, DataLoader, Dataset, transforms)
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
    parser.add_argument("--mask-path-col", default="mask_path")
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
