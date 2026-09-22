#!/usr/bin/env python3
"""Assemble recorded scores without opening data, predicting, or training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


DECLARATION_COMMIT = "c6136371e644bd0c9a9e945c53f8d5d544475115"
ORIGINAL_INTERPRETATION = (
    "Pipeline/specification did not learn: the noiseless central twin was at "
    "chance-like utility (macro AUC 0.435143, accuracy 0.169551, log-loss 1.799129). "
    "It trained 21 modal-labelled subject averages for only five SGD updates at "
    "learning rate 0.001. This is not a privacy utility result. Original records "
    "are preserved byte-for-byte.")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_integrity(evidence):
    manifest = evidence / "r3/original-evidence-hashes.json"
    hashes = read(manifest)
    backups = {"README.md": "README-r1.md", "summary.json": "summary-r1.json"}
    checked = {}
    for original, expected in hashes.items():
        retained = backups.get(original, original)
        actual = sha(evidence / retained)
        assert actual == expected, f"Original evidence changed: {retained}"
        checked[retained] = actual
    return {"status": "verified", "files_checked": len(checked),
            "manifest": str(manifest.relative_to(evidence)),
            "manifest_sha256": sha(manifest), "verified_sha256": checked,
            "original_name_to_backup": backups}


def triple(summary, branch):
    return " / ".join(f"{summary[branch][metric]['mean']:.6f}"
                      for metric in ("macro_auc", "accuracy", "log_loss"))


def annotation(cell):
    summary = cell["summary"]
    auc = summary["federated_dp"]["macro_auc"]["mean"] > .5
    accuracy = (summary["federated_dp"]["accuracy"]["mean"] >
                summary["trivial"]["accuracy"]["mean"])
    return f"AUC > .5: {'yes' if auc else 'no'}; accuracy > majority: {'yes' if accuracy else 'no'}"


def main(evidence):
    integrity = original_integrity(evidence)
    previous = read(evidence / "summary-r1.json")
    old_index = {entry["file"]: entry for entry in previous["cells"]}
    cells, records = [], []
    for revision, unit, prefix in (("original", "subject", "pilot_uci_har"),
                                   ("R3", "subject", "har_subject"),
                                   ("R3", "window", "har_window")):
        for epsilon in (1, 4, 8):
            name = f"{prefix}_pytorch_lstm_eps{epsilon}.json"
            path = evidence / name
            cell = read(path)
            assert cell["status"] == "executed" and cell["contract"] == "pytorch_lstm"
            assert cell["epsilon"] == epsilon and len(cell["per_replicate"]) == 3
            if revision == "original":
                assert sha(path) == old_index[name]["sha256"]
                interpretation = ORIGINAL_INTERPRETATION
                diagnostics = cell["utility_diagnostics"]
            else:
                assert cell["unit"] == unit
                interpretation, diagnostics = cell["interpretation"], cell["diagnostics"]
            for branch in ("central", "federated_dp", "trivial"):
                for metric in ("macro_auc", "accuracy", "log_loss"):
                    assert all(math.isfinite(cell["summary"][branch][metric][key])
                               for key in ("mean", "sd"))
            entry = {"file": name, "sha256": sha(path), "revision": revision,
                     "status": "executed", "contract": "pytorch_lstm",
                     "dataset": "UCI HAR", "unit": unit, "epsilon": epsilon,
                     "interpretation": interpretation, "summary": cell["summary"],
                     "diagnostics": diagnostics, "diagnostics_annotation_only": True,
                     "limitations": cell["limitations"]}
            cells.append(entry)
            records.append(cell)
    corrected = records[3:]
    reference = corrected[0]
    protocol = reference["protocol"]
    runtime = read(evidence / "r3/runtime.json")
    assert runtime == reference["runtime"]
    assert protocol["units"] == ["subject", "window"]
    assert protocol["rounds"] == 5 and protocol["delta"] == 1e-6 and protocol["clipping_norm"] == 1
    shared = {rep["seed"]: {"seed": rep["seed"], "metrics": rep["central"],
              "model_sha256": rep["central_training"]["model_sha256"],
              "training": rep["central_training"],
              "diagnostics": rep["utility_diagnostics"]["central"]}
              for rep in reference["per_replicate"]}
    assert set(shared) == set(protocol["seeds"])
    for cell in corrected:
        assert cell["protocol"] == protocol and cell["runtime"] == runtime
        assert cell["protocol_sha256"] == reference["protocol_sha256"]
        assert cell["summary"]["central"] == reference["summary"]["central"]
        assert cell["summary"]["trivial"] == reference["summary"]["trivial"]
        assert {rep["seed"] for rep in cell["per_replicate"]} == set(shared)
        for rep in cell["per_replicate"]:
            central = shared[rep["seed"]]
            assert rep["central"] == central["metrics"]
            assert rep["central_training"]["model_sha256"] == central["model_sha256"]
            assert len(rep["node_round_captures"]) == 15
    scoring_path = evidence / "r3/test-scoring-started.json"
    scoring = read(scoring_path)
    model_hashes = [rep["federated_model_sha256"] for cell in corrected
                    for rep in cell["per_replicate"]]
    assert sorted(scoring["federated_models"]) == sorted(model_hashes)
    assert sorted(scoring["central_models"]) == sorted(value["model_sha256"] for value in shared.values())
    assert scoring["protocol_sha256"] == reference["protocol_sha256"]
    diagnosis = read(evidence / "r3/diagnosis/audit.json")
    assert diagnosis["test_accessed"] is False
    assert diagnosis["archive_alignment_exact"] and diagnosis["layout_exact"]
    assert diagnosis["prior_was_affine_scaled"]
    blocked = evidence / "blocked-0.5.0/summary.json"
    result = {"schema": "dsflower-sequence-summary-all-v1", "status": "executed",
              "token": protocol["token"], "contract": "pytorch_lstm",
              "dataset": "UCI Human Activity Recognition Using Smartphones",
              "n_scored_cells": len(cells), "cells": cells,
              "interpretations": {"original": ORIGINAL_INTERPRETATION,
                  "subject": protocol["subject_cell"], "window": protocol["window_cell"]},
              "shared_corrected_central": list(shared.values()),
              "declaration_commit": DECLARATION_COMMIT,
              "scoring_marker_sha256": sha(scoring_path),
              "protocol_sha256": reference["protocol_sha256"], "protocol": protocol,
              "runtime": runtime, "original_evidence_integrity": integrity,
              "blocked_archive": {"file": "blocked-0.5.0/summary.json",
                  "sha256": sha(blocked), "included_in_scored_cell_count": False},
              "reporting": {"created_at": datetime.now(timezone.utc).isoformat(),
                  "source": "Existing score JSONs only; no data access or metric recomputation."}}
    selected = protocol["inner_selection"]
    schedule = protocol["model_params"]
    packages = runtime["packages"]
    sources = runtime["release_sources"]
    rows = []
    for cell in cells:
        gap = cell["summary"]["gap_macro_auc"]
        unit = "subject pooled" if cell["unit"] == "subject" else "window (row)"
        rows.append(f"| [{cell['revision']}]({cell['file']}) | pytorch_lstm | UCI HAR | {unit} | {cell['epsilon']} | "
                    f"{triple(cell['summary'], 'central')} | {triple(cell['summary'], 'federated_dp')} | "
                    f"{triple(cell['summary'], 'trivial')} | {gap['mean']:+.6f} ± {gap['sd']:.6f} | "
                    f"{annotation(cell)} |")
    document = f"""# UCI HAR sequence cells: original and corrected R3

Token: `{protocol['token']}`. All nine scored cells are indexed in
[summary.json](summary.json), with individual replicates and diagnostics in the
linked JSON files. The original pipeline did not learn: its noiseless central
twin gave **{triple(records[0]['summary'], 'central')}** (macro OVR AUC / accuracy /
log-loss), near chance utility. These original results are a pipeline/specification
failure, not a privacy utility finding. Their records remain unchanged.

## Diagnosis and declaration

TRAIN-only checks ruled out **(a) layout** and **(d) label misalignment**: the
contract receives `[N,128,9]` from C-order token-major flat rows, and prepared
features, targets and subjects match the archive exactly. **(b) unstandardised
inputs was not the cause:** the old contract already applied clipped affine
scaling using TRAIN-derived bounds. **(c) the schedule was inadequate:** only
five updates at SGD 0.001. In addition, the patient path collapsed all activities
within each subject to one average sequence and its modal label, leaving only
21 central training examples. No package plumbing defect was found.

The same contract-built GPU LSTM learned on 5,564 windows from 16 TRAIN subjects.
On 1,788 inner-validation windows from TRAIN subjects 1, 8, 17, 25 and 30,
the selected central-only run achieved **{selected['selected_macro_auc']:.9f} /
{selected['selected_accuracy']:.9f} / {selected['selected_log_loss']:.9f}** after
20 epochs. The schedule was selected by final inner-validation macro AUC between
Adam learning rates 0.003 and 0.01. TEST was not accessed for diagnosis or selection.
See [diagnostic audit](r3/diagnosis/audit.json),
[recorded experiments](r3/diagnosis/experiments.json), and
[the declaration committed before corrected training](../../../../tools/campaign/sequence/README.md)
at `{DECLARATION_COMMIT}`; the binding
[protocol](../../../../tools/campaign/sequence/r3/protocol.json) is also embedded
in every corrected cell.

The corrected declaration is **Adam {schedule['learning_rate']}, batch
{schedule['batch_size']}, {schedule['local_epochs']} local epochs × five rounds**,
hidden size 32, six classes, no scheduler or penalties. Inputs use the same
128 × 9 time-major layout. Fixed public design bounds are ±1 g for body
acceleration, ±1 rad/s for gyro, and ±2 g for total acceleration; the unchanged
contract applies `clip(x,-b,b)/b` once. These constants are declared design
choices, not empirical extrema or guaranteed archive limits.

Three subject-disjoint sites contain seven subjects each (2,553 / 2,397 / 2,402
windows). Both corrected units use seeds 20260922–20260924, epsilon 1 / 4 / 8,
delta 1e-6 and clipping norm 1. **Subject:** the released patient contract clips
each pooled subject's gradient, seven units/site and 20 steps/site. It does not
aggregate losses or gradients over the original windows within each subject;
that requested behavior is unsupported without a mechanism change. **Window:**
the released row contract clips each window, with 200 steps/site. This is a
window-level mechanism measurement and provides no subject-level protection.

The corrected central comparator is trained and scored once per seed on all
7,352 original TRAIN windows, without DP or federation, with identical
architecture, initialization, bounds and nominal 20-epoch schedule. Adam resets
every four epochs; ordinary shuffled minibatches give 580 updates. Its three
models are shared across all six corrected cells. The subject comparison
therefore includes the pooling/task mismatch. Trivial probabilities are TRAIN
class frequencies; classification is the TRAIN majority class.

## All scored cells

Metric triples are mean **macro one-vs-rest AUC / accuracy / log-loss** over three
training seeds. The gap is federated-DP minus paired central macro AUC, with
sample SD. Diagnostic thresholds are annotations only. The original central
comparator used 21 subject averages; corrected rows use the shared full-window
central comparator described above.

| Run | Contract | Dataset | Privacy unit | Epsilon | Central | Federated-DP | Trivial | AUC gap mean ± SD | Diagnostics |
|---|---|---|---|---:|---|---|---|---|---|
{chr(10).join(rows)}

Full per-seed predicted-class counts, mean probabilities, probability spans,
metrics, timing, initialization hashes, release policies and accountant captures
are in the corrected JSONs. Original diagnostics remain in their pilot JSONs.

## Interpretation and limits

- Original: no useful central learning, so near-zero DP-versus-central gaps do
  not establish privacy utility.
- Subject: a valid subject-unit measurement of the released pooled surrogate,
  in the 21-unit regime. Its gap combines pooling/task change, federation,
  clipping and noise; it cannot isolate the cost of privacy.
- Window: window privacy only, despite subject-disjoint sites. The gap combines
  federation, sampling, clipping and noise; no noiseless federated control or
  pooled-DP twin was scheduled.
- Metrics use held-out windows; the split is fixed. SD measures training
  variation across three seeds, not split or population uncertainty. Central
  results repeated across corrected table rows are the same three models.
- Each epsilon/seed/unit is a separate per-training mechanism contract. No
  composed campaign guarantee is claimed. Non-DP inner selection on public
  TRAIN data does not provide an end-to-end private model-selection guarantee.
- Public seeds determine initialization. Node-owned cryptographic sampling and
  noise remain unchanged; no deterministic noise replacement is used.

## Provenance and preservation

dsFlower **{packages['dsFlower']}**, server patch
`{sources['dsFlower']['source_commit']}` on
`fix/import-guard-torch-generated-modules`; dsFlowerClient
**{packages['dsFlowerClient']}**, release
`{sources['dsFlowerClient']['source_commit']}`. Declaration commit:
`{DECLARATION_COMMIT}`. Runtime tooling commit:
`{runtime.get('tooling_commit')}`. Both installed canonical runner hashes:
`{packages['server_runner_sha256']}`. No R3 package code, guard, privacy
mechanism or accountant changes. Exact environment and tooling hashes are in
[r3/runtime.json](r3/runtime.json).

All 18 corrected federations and three central models were verified before the
exclusive [R3 scoring marker](r3/test-scoring-started.json). The one final pass
loaded TEST once and predicted each of those 21 models once. Every scored cell
is retained; none was rerun or retuned after scoring. The public observer's
270 node-round captures and independent accounting are included in the cells.
The pod `pod-flower-sequence` remains running at `/workspace/cells-sequence`.

All {integrity['files_checked']} original files match
[the frozen hash manifest](r3/original-evidence-hashes.json). The original report
and index are retained byte-for-byte as [README-r1.md](README-r1.md) and
[summary-r1.json](summary-r1.json); pilot JSONs and both old scoring and failure
records are unchanged. The pre-training dsFlower 0.5.0 import failure remains
under [blocked-0.5.0](blocked-0.5.0/README.md) and is indexed separately from the
nine scored cells. Report assembly reads recorded JSONs only and does not score
or open any dataset split.

## Dataset citation

[UCI HAR, dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
[official archive](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip),
DOI [10.24432/C54S4K](https://doi.org/10.24432/C54S4K), CC BY 4.0.
Archive SHA-256: `{reference['dataset']['archive_sha256']}`.

Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013).
*A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN.
"""
    (evidence / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (evidence / "README.md").write_text(document)
    assert original_integrity(evidence) == integrity
    print(json.dumps({"scored_cells": len(cells), "original_files_verified": integrity["files_checked"],
                      "summary_sha256": sha(evidence / "summary.json"),
                      "readme_sha256": sha(evidence / "README.md")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    main(parser.parse_args().evidence)
