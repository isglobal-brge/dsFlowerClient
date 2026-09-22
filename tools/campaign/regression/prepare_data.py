"""Download the public source without inspecting feature or target values."""
import argparse
import hashlib
import json
import pathlib
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/cells")
    args = parser.parse_args()
    protocol_path = pathlib.Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    cache = pathlib.Path(args.root) / "data_cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "parkinsons_telemonitoring.zip"
    if not archive.exists():
        urllib.request.urlretrieve(protocol["dataset"]["source_url"], archive)
    with zipfile.ZipFile(archive) as source:
        for name in ("parkinsons_updrs.data", "parkinsons_updrs.names"):
            (cache / name).write_bytes(source.read(name))
    hashes = {
        name: hashlib.sha256((cache / name).read_bytes()).hexdigest()
        for name in (archive.name, "parkinsons_updrs.data", "parkinsons_updrs.names")
    }
    (cache / "CHECKSUMS.sha256").write_text(
        "".join(f"{value}  {name}\n" for name, value in hashes.items()))
    provenance = dict(protocol["dataset"], checksums=hashes)
    provenance["protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    (cache / "regression_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
