"""Fetch hash-verified Linux wheels from the committed CPython 3.11 lock."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import tomllib
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    lock = tomllib.loads(Path(__file__).with_name("pylock.toml").read_text())

    def fetch(package):
        wheel = package["wheels"][0]
        name = urllib.parse.unquote(urllib.parse.urlsplit(wheel["url"]).path.rsplit("/", 1)[1])
        path = args.destination / name
        expected = wheel["hashes"]["sha256"]
        if not path.exists():
            urllib.request.urlretrieve(wheel["url"], path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Wheel checksum mismatch: {name}")
        return f"{actual}  {name}\n"

    with ThreadPoolExecutor(max_workers=8) as pool:
        checksums = list(pool.map(fetch, lock["packages"]))
    (args.destination / "CHECKSUMS.sha256").write_text("".join(checksums))
    print(f"Verified {len(checksums)} Linux wheels")


if __name__ == "__main__":
    main()
