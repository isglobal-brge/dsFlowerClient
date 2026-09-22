"""Rebuild frozen CDC training inputs without creating or opening test.csv."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/cells"))
    args = parser.parse_args()
    script = Path(__file__).resolve()
    repo = script.parents[4]
    regression = script.parent.parent
    evidence = repo / "inst/extdata/campaign/regression"
    frozen = json.loads((evidence / "cdcbmi_provenance.json").read_text())
    protocol = regression / "cdcbmi_protocol.json"
    assert sha256(protocol) == frozen["protocol_sha256"]
    campaign = regression.parent / "campaign_lib.R"
    assert sha256(campaign) == frozen["logistic_campaign_lib_sha256"]
    cache = args.root / "data_cache"
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / "cdc_diabetes_health_indicators.csv"
    if not source.exists():
        with urllib.request.urlopen(frozen["source_url"], timeout=120) as response:
            with source.with_suffix(".download").open("wb") as output:
                shutil.copyfileobj(response, output)
        source.with_suffix(".download").replace(source)
    assert sha256(source) == frozen["source_sha256"]
    for seed in frozen["splits"]:
        archived = evidence / f"cdcbmi_split_seed{seed}.json"
        assert sha256(archived) == frozen["checksums"][f"cdcbmi_seed{seed}/split.json"]
    with tempfile.TemporaryDirectory(prefix="r4_training_", dir=cache) as temp:
        subprocess.run(["Rscript", "--vanilla", str(script.with_suffix(".R")),
                        str(protocol), str(campaign), str(cache), temp,
                        str(evidence)], check=True)
        temporary = Path(temp)
        preparation = json.loads((temporary / "preparation.json").read_text())
        checksums = {}
        for seed in frozen["splits"]:
            folder = f"cdcbmi_seed{seed}"
            (cache / folder).mkdir(exist_ok=True)
            for filename in ("train.csv", "site1.csv", "site2.csv", "site3.csv"):
                relative = f"{folder}/{filename}"
                checksum = sha256(temporary / relative)
                assert checksum == frozen["checksums"][relative], relative
                destination = cache / relative
                if destination.exists():
                    assert sha256(destination) == checksum, relative
                else:
                    shutil.copyfile(temporary / relative, destination)
                checksums[relative] = checksum
            shutil.copyfile(evidence / f"cdcbmi_split_seed{seed}.json",
                            cache / folder / "split.json")
    preparation.update(source_url=frozen["source_url"], source_sha256=sha256(source),
                       training_checksums=checksums, frozen_hashes_verified=True,
                       heldout_policy="Source read only for deterministic routing; no test.csv opened or created, no held-out summaries or scoring.")
    output = args.root / "r4" / "preparation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(preparation, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
