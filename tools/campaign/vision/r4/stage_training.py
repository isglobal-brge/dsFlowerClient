#!/usr/bin/env python3
"""Copy only a supplied training collection, verifying raw image byte parity."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def stage(source, destination):
    """Stage three training sites into a new directory; refuse any overwrite."""
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("staging destination must be outside the source collection")
    destination.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    sites, all_images = [], []
    for number in range(1, 4):
        original, copied = source / f"site{number}", destination / f"site{number}"
        copied.mkdir()
        (copied / "images").mkdir()
        with (original / "samples.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        image_hashes = {}
        for row in rows:
            relative = Path(row["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("training image paths must stay within the supplied site")
            incoming, outgoing = original / "images" / relative, copied / "images" / relative
            if incoming.is_symlink() or not incoming.is_file():
                raise ValueError("training image must be a regular file")
            if (original / "images").resolve() not in incoming.resolve().parents:
                raise ValueError("training image resolves outside the supplied site")
            if str(relative) in image_hashes:
                continue
            outgoing.parent.mkdir(parents=True, exist_ok=True)
            source_hash = sha(incoming)
            shutil.copyfile(incoming, outgoing)
            assert sha(outgoing) == source_hash, "staged image bytes differ"
            image_hashes[str(relative)] = source_hash
        hashes = {}
        for name in ("samples.csv", "sample_manifests.csv"):
            shutil.copyfile(original / name, copied / name)
            hashes[name] = dict(source_sha256=sha(original / name),
                                staged_sha256=sha(copied / name))
            assert hashes[name]["source_sha256"] == hashes[name]["staged_sha256"]

        def relocate(value):
            if isinstance(value, dict):
                return {key: relocate(item) for key, item in value.items()}
            if isinstance(value, list):
                return [relocate(item) for item in value]
            if isinstance(value, str) and value.startswith("/"):
                relative = Path(value).resolve().relative_to(original)
                return str(copied / relative)
            return value

        index_name = "content_hash_index.csv"
        with (original / index_name).open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, index = reader.fieldnames, list(reader)
        assert len(index) == len(rows)
        for row in index:
            relative = Path(row["uri"]).resolve().relative_to(original / "images")
            assert image_hashes[str(relative)] == row["content_hash"]
            row["uri"] = relocate(row["uri"])
        with (copied / index_name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(index)
        hashes[index_name] = dict(source_sha256=sha(original / index_name),
                                  staged_sha256=sha(copied / index_name))
        for name in ("manifest.yaml", "registry.yaml"):
            # These prepared dsImaging files are JSON, a YAML subset.
            value = json.loads((original / name).read_text())
            write_json(copied / name, relocate(value))
            hashes[name] = dict(source_sha256=sha(original / name),
                                staged_sha256=sha(copied / name))
        canonical = "\n".join(f"{name}\t{digest}" for name, digest in sorted(image_hashes.items()))
        sites.append(dict(site=number, n_images=len(image_hashes), n_rows=len(rows),
            n_patients=len({row["subject_id"] for row in rows}),
            image_hash_aggregate_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
            source_and_staged_image_hashes_identical=True, file_hashes=hashes))
        all_images.extend(f"site{number}/{name}\t{digest}" for name, digest in sorted(image_hashes.items()))
    audit = dict(schema="dsflower-vision-r4-training-staging-v1", source=str(source),
        destination=str(destination), started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(), sites=sites,
        n_images=sum(site["n_images"] for site in sites),
        n_patients=sum(site["n_patients"] for site in sites),
        image_hash_aggregate_sha256=hashlib.sha256("\n".join(all_images).encode()).hexdigest(),
        policy="Byte-identical raw training images; samples and sample manifests unchanged; only absolute collection paths rewritten. No feature cache.",
        test_accessed=False)
    write_json(destination / "staging-audit.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.destination), indent=2), flush=True)
