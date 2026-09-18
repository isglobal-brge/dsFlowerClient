#!/usr/bin/env python3
"""Download and hash only the public F segmentation benchmark releases."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import requests
import zipfile

SOURCES = {
    "BUSBRA.zip": "https://zenodo.org/api/records/8231412/files/BUSBRA.zip/content",
    "busbra-record.json": "https://zenodo.org/api/records/8231412",
    "BrEaST.zip": "https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip",
    "BrEaST-clinical.xlsx": "https://www.cancerimagingarchive.net/wp-content/uploads/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx",
    "BrEaST-source.html": "https://www.cancerimagingarchive.net/collection/breast-lesions-usg/",
    "CC-BY-4.0.txt": "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
    "resnet18-f37072fd.pth": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "torchvision-LICENSE.txt": "https://raw.githubusercontent.com/pytorch/vision/v0.19.1/LICENSE",
    "torchvision-model-terms.html": "https://docs.pytorch.org/vision/0.19/models.html",
    "BUS-BRA-README.md": "https://raw.githubusercontent.com/wgomezf/BUS-BRA/main/README.md",
}

PINNED_SHA256 = {
    "BUSBRA.zip": "ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff",
    "BrEaST.zip": "c32e49cdfa54042e065582d0cb35978b5fc5e5b5e461b33bf773fb50cfdad355",
    "BrEaST-clinical.xlsx": "89a8874496a6f1f93960390b963f7369b90f7e72ee08a738f2d17e37f5eff8bf",
    "resnet18-f37072fd.pth": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "CC-BY-4.0.txt": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
}


def fetch(item, root):
    name, url = item
    path = root / name
    if not path.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    data = path.read_bytes()
    record = {"filename": name, "url": url, "bytes": len(data),
              "sha256": hashlib.sha256(data).hexdigest()}
    if name in PINNED_SHA256:
        assert record["sha256"] == PINNED_SHA256[name], "pinned release bytes changed"
    if name == "BUSBRA.zip":
        assert hashlib.md5(data).hexdigest() == "1f8b2be6476d58fc97bfb5e5a1ea9bab"
        record["publisher_md5"] = "1f8b2be6476d58fc97bfb5e5a1ea9bab"
    print(json.dumps(record), flush=True)
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda item: fetch(item, args.root), SOURCES.items()))
    for name, subdir in (("BUSBRA.zip", "busbra"), ("BrEaST.zip", "breast")):
        destination = args.root / subdir
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(args.root / name) as archive:
            for entry in archive.infolist():
                assert (destination / entry.filename).resolve().is_relative_to(destination.resolve())
            archive.extractall(destination)
    (args.root / "release-manifest.json").write_text(json.dumps({
        "schema": "dsflower-segmentation-public-provenance-v1", "sources": records}, indent=2) + "\n")

if __name__ == "__main__":
    main()
