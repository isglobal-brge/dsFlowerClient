#!/usr/bin/env python3
"""BUSI-only nonprivate pretraining. This process never opens BUS-BRA data."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from PIL import Image
from dsflower_runner import params, segmentation
from feature_smoke import config
from protocol_v4 import SEEDS


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def train(archive_path, provenance_path, output, protocol_hash):
    provenance = json.loads(provenance_path.read_text())
    if sha256(archive_path) != provenance['sha256']:
        raise ValueError('BUSI release differs from preregistration')
    output.mkdir(parents=True, exist_ok=False)
    images = output / 'images'
    images.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if not (images / entry.filename).resolve().is_relative_to(images.resolve()):
                raise ValueError('archive member outside BUSI extraction root')
        archive.extractall(images)
    files = sorted(p for p in images.rglob('*.png') if '_mask' not in p.stem)
    classes = {name: sum(p.parent.name == name for p in files)
               for name in ('benign', 'malignant', 'normal')}
    if classes != dict(benign=437, malignant=210, normal=133) or len(files) != 780:
        raise ValueError('BUSI version 1 image census differs')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    cfg = config()
    cfg['model-spec-b64'] = base64.b64encode(json.dumps(segmentation.decoder_spec('narrow')).encode()).decode()
    encoder, device = segmentation.prepare_encoder(cfg)
    before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    X = np.zeros((780, segmentation.FEATURE_DIM), np.float32)
    y = np.zeros((780, 2, 128, 128), np.float32)
    census = []
    normalized = output / 'normalized-masks'
    normalized.mkdir()
    with torch.no_grad():
        for i, path in enumerate(files):
            masks = sorted(path.parent.glob(path.stem + '_mask*.png'))
            converted = []
            for source in masks:
                with Image.open(source) as raster:
                    pixels = np.asarray(raster)
                if pixels.ndim == 3:
                    if pixels.shape[2] not in (3, 4) or not np.array_equal(pixels[:, :, 0], pixels[:, :, 1]) or not np.array_equal(pixels[:, :, 0], pixels[:, :, 2]):
                        raise ValueError('BUSI mask RGB channels are not identical silhouettes')
                    pixels = pixels[:, :, 0]  # Alpha is not a class, as in v3/v4 BrEaST.
                if pixels.dtype == bool:
                    pixels = pixels.astype(np.uint8) * 255
                if pixels.ndim != 2 or not np.isin(pixels, [0, 255]).all():
                    raise ValueError('BUSI source mask vocabulary differs')
                destination = normalized / source.relative_to(images)
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels.astype(np.uint8)).save(destination)
                converted.append(destination)
            empty = path.parent.name == 'normal'
            image, mask, valid = segmentation.read_pair(str(path), [str(p) for p in converted],
                                                        [empty] * len(masks), '0,255')
            if valid:
                X[i] = np.clip(encoder(torch.from_numpy(image)[None].to(device)).flatten(1).cpu().numpy()[0], -1e6, 1e6)
                y[i, 0] = mask[0]
                y[i, 1] = 1
            census.append(dict(image=str(path.relative_to(images)), image_sha256=sha256(path),
                               masks={str(p.relative_to(images)): sha256(p) for p in masks},
                               normalized_masks={str(p.relative_to(normalized)): sha256(p) for p in converted},
                               explicit_empty=empty, valid=bool(valid)))
    if not all(torch.equal(before[k], v) for k, v in encoder.state_dict().items()):
        raise ValueError('public encoder state changed')
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('public encoder is not frozen')
    np.savez(output / 'public-busi-tensors.npz', X=X, y=y)
    audit = dict(dataset='BUSI', privacy='public_nonprivate', image_count=780, classes=classes,
                 valid_count=int(y[:, 1, 0, 0].sum()), invalid_retained=int(780-y[:, 1, 0, 0].sum()),
                 patient_mapping='unavailable; image-level public training only, no patient claim',
                 source=provenance, protocol_sha256=protocol_hash,
                 encoder_sha256=segmentation.CHECKPOINT_SHA256, encoder_state_unchanged=True,
                 feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                 target_sha256=hashlib.sha256(y.tobytes()).hexdigest(), sources=census)
    write(output / 'audit.json', audit)
    del encoder, before
    features = torch.from_numpy(X).to(device)
    target = torch.from_numpy(y).to(device)
    loss = segmentation.loss_factory(cfg)
    # Each seed has one 60-epoch path; the 20-epoch candidate is its fixed prefix.
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = params.load_user_model(cfg, segmentation.FEATURE_DIM, 'segmentation_bce_dice').to(device)
        initial_hash = hashlib.sha256(b''.join(a.tobytes() for a in params.get_torch_params(model))).hexdigest()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0., amsgrad=False)
        for epoch in range(1, 61):
            order_seed = int.from_bytes(hashlib.sha256(f'busi-v5-order|{seed}|{epoch}'.encode()).digest()[:8], 'big')
            order = np.random.default_rng(order_seed).permutation(len(X))
            total = 0.
            for start in range(0, len(X), 16):
                index = torch.as_tensor(order[start:start+16], device=device)
                optimizer.zero_grad(set_to_none=True)
                value = loss(model(features[index]), target[index])
                if not torch.isfinite(value):
                    raise ValueError('nonfinite public pretraining loss')
                value.backward()
                optimizer.step()
                total += value.item() * len(index)
            print(json.dumps(dict(stage='public_nonprivate', dataset='BUSI', seed=seed,
                                  epoch=epoch, training_loss=total/len(X))), flush=True)
            if epoch in (20, 60):
                folder = output / f'epochs{epoch}'
                folder.mkdir(exist_ok=True)
                arrays = params.get_torch_params(model)
                path = folder / f'seed{seed}.npz'
                np.savez(path, **{str(i): a for i, a in enumerate(arrays)})
                write(folder / f'seed{seed}.json', dict(
                    dataset='BUSI', privacy='public_nonprivate', seed=seed, decoder='narrow',
                    epochs=epoch, checkpoint=str(path), checkpoint_sha256=sha256(path),
                    tensor_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
                    encoder_sha256=segmentation.CHECKPOINT_SHA256, protocol_sha256=protocol_hash,
                    dataset_sha256=provenance['sha256'], audit_sha256=sha256(output / 'audit.json'),
                    provenance_sha256=sha256(provenance_path), initial_random_tensor_sha256=initial_hash,
                    optimizer='Adam lr=.001 betas=(.9,.999) eps=1e-8 no decay; batch16, no DP',
                    training_loss=total/len(X)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    train(args.archive, args.provenance, args.out, args.protocol_sha256)
