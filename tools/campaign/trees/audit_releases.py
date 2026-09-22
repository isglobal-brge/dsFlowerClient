#!/usr/bin/env python3
"""Verify saved releases and repair the harness's absent fit$n_clients field.

Reads public model artifacts and saved training metadata only, never holdouts.
Preserves every metric and setting; records the original field and JSON hash.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

output, runs = map(Path, sys.argv[1:])
assert not (output / "release-audit.json").exists(), "Release audit already recorded"
audit = []
for path in sorted(output.glob("pilot_*.json")):
    raw = path.read_bytes()
    cell = json.loads(raw)
    records = []
    for index, rep in enumerate(cell["per_replicate"], 1):
        directory = runs / path.stem / f"rep{index}" / "artifact"
        models = list(directory.glob("*/*.rds"))
        assert len(models) == 1, directory
        saved = models[0]
        count = subprocess.run([
            "Rscript", "-e", "cat(readRDS(commandArgs(TRUE)[1])$n_clients)",
            str(saved)], capture_output=True, text=True, check=True).stdout.strip()
        assert count == "3", (saved, count)
        artifact = saved.parent / rep["release_metadata"]["artifact"]["file"]
        encoded = artifact.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        assert digest == rep["model_sha256"] == rep["release_metadata"]["artifact"]["sha256"]
        assert len(encoded) == rep["release_metadata"]["artifact"]["size_bytes"]
        ensemble = json.loads(encoded)
        assert len(ensemble["models"]) == int(count) == len(rep["node_privacy"]) == 3
        assert ensemble["public_schema_sha256"] == rep["release_metadata"]["public_schema_sha256"]
        for model in ensemble["models"]:
            assert model["depth"] == cell["model_params"]["max_depth"]
            assert len(model["trees"]) == cell["model_params"]["n_estimators"]
        previous = rep["history"]["n_clients"]
        assert previous in ([], 3), previous
        rep["history"]["n_clients"] = int(count)
        rep["history"]["n_clients_source"] = "saved training RDS and SHA-256-bound ensemble member count"
        rep["history"]["n_clients_original"] = previous
        records.append({"replicate": index, "saved_rds_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                        "ensemble_sha256": digest, "ensemble_members": len(ensemble["models"]),
                        "saved_n_clients": int(count), "original_n_clients": previous})
    # Assert explicitly that reporting repair leaves all scores/settings intact.
    restored = json.loads(json.dumps(cell))
    for rep in restored["per_replicate"]:
        rep["history"]["n_clients"] = rep["history"].pop("n_clients_original")
        del rep["history"]["n_clients_source"]
    assert restored == json.loads(raw)
    path.write_text(json.dumps(cell, indent=2, allow_nan=False) + "\n")
    audit.append({"file": path.name, "original_json_sha256": hashlib.sha256(raw).hexdigest(),
                  "corrected_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "replicates": records})
(output / "release-audit.json").write_text(json.dumps({
    "reason": "fit$n_clients is absent; as.integer(NULL) serialized as [] in the original campaign harness",
    "metrics_or_settings_changed": False, "retrained_or_rescored": False,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cells": audit}, indent=2) + "\n")
print(f"Verified and repaired client counts in {len(audit)} cells without rescoring.")
