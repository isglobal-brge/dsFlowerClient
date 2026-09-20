#!/usr/bin/env python3
"""Audit release subject IDs; normalize public mask channels; freeze subject splits.

This command computes no model or baseline scores and must precede every scored
cell. Original archives remain unchanged. Output counts describe public fixtures.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

SEEDS = (20260919, 20260920, 20260921)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def ordered(subjects, seed, domain):
    return sorted(subjects, key=lambda s: (digest(f"{domain}|{seed}|{s}".encode()), s))


def split_subjects(subjects, seed, pathology):
    order = ordered(subjects, seed, "split-v1")
    test, train = order[:len(order) // 5], order[len(order) // 5:]
    assignment = ordered(train, seed, "sites-v1")
    sites = [assignment[i::3] for i in range(3)]
    small = ordered(train, seed, "subset-v1")[:192]
    small_order = ordered(small, seed, "sites-v1")
    stress = sorted(assignment, key=lambda s: pathology[s])  # stable public hash ties
    return {"seed": seed, "train": sorted(train), "test": sorted(test),
            "sites": [sorted(s) for s in sites],
            "small_train": sorted(small),
            "small_sites": [sorted(small_order[i::3]) for i in range(3)],
            "heterogeneous_sites": [sorted(s.tolist()) for s in np.array_split(stress, 3)]}


def clinical_rows(path):
    """Read the release's simple XLSX without depending on a spreadsheet runtime."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(x.itertext()) for x in ET.fromstring(
            archive.read("xl/sharedStrings.xml")).findall("s:si", ns)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(".//s:row", ns):
        result = {}
        for cell in row.findall("s:c", ns):
            value = cell.find("s:v", ns)
            text = "" if value is None else value.text
            if cell.get("t") == "s":
                text = strings[int(text)]
            result[re.sub(r"\d", "", cell.get("r"))] = text
        rows.append(result)
    header = rows.pop(0)
    return [{name: row.get(col, "") for col, name in header.items()} for row in rows]


def normalize_mask(data, destination):
    with Image.open(io.BytesIO(data)) as image:
        values = np.asarray(image)
        if image.mode == "RGBA":
            assert np.array_equal(values[:, :, 0], values[:, :, 1])
            assert np.array_equal(values[:, :, 0], values[:, :, 2])
            values = values[:, :, 0]  # publisher RGB silhouette; alpha is not a class
        assert set(np.unique(values)).issubset({0, 1, 255})
        normalized = (values != 0).astype(np.uint8) * 255
        Image.fromarray(normalized).save(destination)
    return {"source_sha256": digest(data),
            "normalized_sha256": digest(destination.read_bytes())}


def prepare_busbra(root, out):
    with zipfile.ZipFile(root / "BUSBRA.zip") as archive:
        metadata = archive.read("BUSBRA/bus_data.csv")
        records = list(csv.DictReader(io.StringIO(metadata.decode())))
        groups = collections.defaultdict(list)
        rows, mask_hashes = [], {}
        for record in records:
            subject = "busbra:" + str(int(record["Case"]))
            image_id = record["ID"]
            assert int(image_id.split("_")[1].split("-")[0]) == int(record["Case"])
            groups[subject].append(record)
            mask_name = image_id.replace("bus_", "mask_") + ".png"
            data = archive.read("BUSBRA/Masks/" + mask_name)
            mask_hashes[mask_name] = normalize_mask(data, out / "masks" / mask_name)
            rows.append({"subject_id": subject, "image_id": image_id,
                         "relative_path": "BUSBRA/Images/" + image_id + ".png",
                         "mask_path": mask_name, "mask_empty": 0})
        assert len(records) == 1875 and len({r["ID"] for r in records}) == 1875
        assert len(groups) == 1064
        assert {int(r["Case"]) for r in records} == set(range(1, 1065))
        assert all(len({r["Pathology"] for r in rs}) == 1 for rs in groups.values())
        for folds in ("5-fold-cv.csv", "10-fold-cv.csv"):
            cv = list(csv.DictReader(io.StringIO(archive.read("BUSBRA/" + folds).decode())))
            by_case = collections.defaultdict(set)
            for row in cv:
                by_case[row["ID"].split("-")[0]].add(row["kFold"])
            assert all(len(value) == 1 for value in by_case.values())
        (out / "BUS-BRA-LICENSE.txt").write_bytes(archive.read("BUSBRA/LICENSE.txt"))
    audit = {"public_source_images": len(records), "public_subjects": len(groups),
             "multiplicity": dict(collections.Counter(map(len, groups.values()))),
             "source_metadata_sha256": digest(metadata), "mask_hashes": mask_hashes,
             "patient_mapping": "released Case column, exact 1..1064, filename prefix agrees; publisher declares 1064 patients; both official CV folds keep Case together",
             "mask_conversion": "source bool silhouette -> L uint8 {0,255}; unchanged native geometry"}
    return rows, {s: rs[0]["Pathology"] for s, rs in groups.items()}, audit, root / "busbra"


def prepare_breast(root, out):
    records = clinical_rows(root / "BrEaST-clinical.xlsx")
    rows, masks, normal = [], {}, []
    with zipfile.ZipFile(root / "BrEaST.zip") as archive:
        names = archive.namelist()
        for record in records:
            image_id = Path(record["Image_filename"]).stem
            assert image_id == f"case{int(record['CaseID']):03d}"
            image_file = "BrEaST-Lesions_USG-images_and_masks/" + image_id + ".png"
            targets = [n for n in names if re.fullmatch(
                "BrEaST-Lesions_USG-images_and_masks/" + image_id + r"_(tumor|other\d*)\.png", n)]
            if not targets:
                assert record["Classification"] == "normal"
                normal.append(image_id)
                with Image.open(io.BytesIO(archive.read(image_file))) as image:
                    Image.new("L", image.size).save(out / "masks" / (image_id + "_empty.png"))
                targets = [None]
            for target in targets:
                mask_name = Path(target).name if target else image_id + "_empty.png"
                if target:
                    masks[mask_name] = normalize_mask(archive.read(target), out / "masks" / mask_name)
                rows.append({"subject_id": "breast:" + str(int(record["CaseID"])),
                             "image_id": image_id, "relative_path": image_file,
                             "mask_path": mask_name, "mask_empty": int(target is None)})
    assert len(records) == 256 and len(normal) == 4
    assert set(normal) == {"case045", "case061", "case209", "case213"}
    return rows, {"breast:" + str(int(r["CaseID"])): r["Classification"] for r in records}, {
        "public_source_images": 256, "public_subjects": 256,
        "public_mask_annotations": len(masks), "public_declared_empty": len(normal),
        "mask_hashes": masks,
        "mask_conversion": "equal RGB silhouette from RGBA -> L uint8 {0,255}; native geometry; all tumor/other suspicious-lesion annotations unioned by runtime",
        "patient_mapping": "CaseID + unique Image_filename; publisher explicitly one scan per patient",
        "empty_declaration": "Only XLSX Classification=normal and no tumor/other annotation; four explicit blank PNGs generated"}, root / "breast"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for name, prepare in (("breast", prepare_breast), ("busbra", prepare_busbra)):
        out = args.out / name
        (out / "masks").mkdir(parents=True, exist_ok=True)
        rows, pathology, audit, image_root = prepare(args.root, out)
        with (out / "samples.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        audit.update({"schema": "dsflower-segmentation-public-audit-v1", "dataset": name,
                      "samples_sha256": digest((out / "samples.csv").read_bytes()),
                      "image_root": str(image_root), "mask_root": str(out / "masks"),
                      "mask_vocabulary": "0,255", "split_hashes": {}})
        for seed in SEEDS:
            path = out / f"split-{seed}.json"
            write_json(path, split_subjects(sorted(pathology), seed, pathology))
            audit["split_hashes"][str(seed)] = digest(path.read_bytes())
        write_json(out / "audit.json", audit)
        print(json.dumps({k: v for k, v in audit.items() if k != "mask_hashes"}), flush=True)


if __name__ == "__main__":
    main()
