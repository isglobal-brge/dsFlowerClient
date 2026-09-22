#!/usr/bin/env python3
"""Fetch pinned public releases and construct dsImaging training collections.

Reuses segmentation preparation verbatim. No model predictions or test summaries.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segmentation"))
from fetch_public_data import SOURCES, fetch
from prepare_public_data import prepare_busbra, split_subjects, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    names = ("BUSBRA.zip", "busbra-record.json", "resnet18-f37072fd.pth",
             "CC-BY-4.0.txt", "torchvision-LICENSE.txt", "BUS-BRA-README.md")
    records = [fetch((name, SOURCES[name]), data) for name in names]
    write_json(data / "release-manifest.json", {"sources": records})
    with zipfile.ZipFile(data / "BUSBRA.zip") as archive:
        destination = data / "busbra"
        for entry in archive.infolist():
            assert (destination / entry.filename).resolve().is_relative_to(destination.resolve())
        archive.extractall(destination)
    checkpoint = root / "torch/hub/checkpoints/resnet18-f37072fd.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data / checkpoint.name, checkpoint)
    prepared = root / "prepared/busbra"
    (prepared / "masks").mkdir(parents=True, exist_ok=True)
    rows, pathology, audit, image_root = prepare_busbra(data, prepared)
    write_csv(prepared / "samples.csv", rows)
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    audit.update(schema="dsflower-vision-data-audit-v1", dataset="busbra",
                 samples_sha256=sha(prepared / "samples.csv"), image_root=str(image_root),
                 archive_sha256=sha(data / "BUSBRA.zip"), split_hashes={})
    for seed in protocol["seeds"]:
        split = split_subjects(sorted(pathology), seed, pathology)
        split_path = prepared / f"split-{seed}.json"
        write_json(split_path, split)
        assert sha(split_path) == protocol["split"]["sha256_by_seed"][str(seed)]
        assert len(split["train"]) == 852 and len(split["test"]) == 212
        audit["split_hashes"][str(seed)] = sha(split_path)
        for i, ids in enumerate(split["sites"], 1):
            collection = root / "prepared/vision" / str(seed) / f"site{i}"
            (collection / "images").mkdir(parents=True, exist_ok=True)
            metadata, manifests, index = [], [], []
            for row in rows:
                if row["subject_id"] not in ids:
                    continue
                name = Path(row["relative_path"]).name
                image = collection / "images" / name
                shutil.copyfile(image_root / row["relative_path"], image)
                label = pathology[row["subject_id"]].lower()
                assert label in protocol["class_mapping"]
                metadata.append(dict(image_id=row["image_id"], subject_id=row["subject_id"],
                                     relative_path=name, pathology=label,
                                     source_kind="single_file", n_files=1))
                manifests.append(dict(sample_id=row["image_id"], source_kind="single_file",
                                      primary_uri=name, files_json=json.dumps([dict(path=name, role="primary")]),
                                      content_hash=sha(image), n_files=1))
                index.append(dict(sample_id=row["image_id"], uri=str(image), content_hash=sha(image),
                                  size=image.stat().st_size, source_kind="single_file"))
            assert len({r["subject_id"] for r in metadata}) == 284
            write_csv(collection / "samples.csv", metadata)
            write_csv(collection / "sample_manifests.csv", manifests)
            write_csv(collection / "content_hash_index.csv", index)
            dataset_id = f"busbra.site{i}"
            manifest = dict(schema_version=1, dataset_id=dataset_id,
                metadata=dict(uri=str(collection / "samples.csv"), format="csv", id_col="image_id",
                    privacy_unit="patient", privacy_unit_col="subject_id",
                    privacy_unit_canonicalization="trim-utf8-v2", label_col="pathology",
                    label_levels=["benign", "malignant"]),
                assets=dict(images=dict(type="image_root", uri=str(collection / "images"),
                                        path_col="relative_path")),
                sample_manifests=dict(uri=str(collection / "sample_manifests.csv"), format="csv"),
                content_hash_index=dict(uri=str(collection / "content_hash_index.csv"), format="csv"))
            write_json(collection / "manifest.yaml", manifest)
            write_json(collection / "registry.yaml", {"schema_version": 1, dataset_id:
                dict(enabled=True, backend="file", manifest_uri=str(collection / "manifest.yaml"))})
    write_json(prepared / "audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "mask_hashes"}), flush=True)


if __name__ == "__main__":
    main()
