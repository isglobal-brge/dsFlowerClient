"""Completion-gate unit fixtures are constructed in memory, never archived evidence."""
import base64
import copy
import hashlib
import json
import math
import unittest

from metrics import envelopes
from validate_completion import (CELLS, COHORTS, EDGES, GROUPS, VARIANTS, flagged_keys,
                                 validate_cell, validate_completion)


def b64(value):
    return base64.b64encode(json.dumps(value).encode()).decode()


def cell(dataset, subset, variant, epsilon, seed):
    """Schema test data only. These numbers make no empirical utility claim."""
    synthetic = dataset == "synthetic-public"
    original, release, licence, features, upper = ((90, "a" * 64, "CC0-1.0", ["x"], [1])
                                                  if synthetic else COHORTS[dataset])
    n = 600 if subset == "small600" else math.floor(.8 * original)
    sites = [dict(site=i+1, source_rows=n//3+(i < n%3), n_subjects=n//3+(i < n%3), split_sha256="a"*64)
             for i in range(3)]
    meta = dict(dataset=dataset, subset=subset, seed=seed, public_fixture=True,
                source=dict(sha256=release, licence=licence, release="unit fixture", url="https://example.org/unit",
                            licence_url="https://example.org/unit", attribution="unit fixture"),
                protocol_sha256="b"*64, subject_provenance="unit fixture", split_rule="unit fixture",
                n_original_rows=original, n_train=n, n_test=original-math.floor(.8*original), sites=sites,
                train_sha256="c"*64, test_sha256="d"*64, features=features,
                feature_bounds=dict(lower=[0]*len(features), upper=upper))
    config = dict(schema_version=1, time_unit="days", time_origin="baseline", t_min=1, horizon=1825)
    if variant == "hazard":
        config["edges"] = EDGES
        contract, loss = "pytorch_discrete_hazard", "discrete_hazard_nll"
    else:
        config.update(distribution=variant, time_scale=365, dispersion=1)
        contract, loss = "pytorch_aft", "aft_"+variant+"_nll"
    cfg = {"learning-rate": .05, "optimizer-name": "sgd", "optimizer-momentum": 0, "weight-decay": 0,
           "l1-penalty": 0, "scheduler-name": "none", "batch-size": 128, "num-server-rounds": 2 if synthetic else 10,
           "local-epochs": 1 if synthetic else 2, "loss-name": loss, "num-features": len(features),
           "survival-config-b64": b64(config),
           "model-spec-b64": b64({"kind": "sequential", "layers": [{"op": "linear", "out": "@out"}]})}
    def mechanism(n):
        steps = math.ceil(n/128)
        value = dict(adjacency="replace_one", noise_multiplier=2., clipping_norm=1., accounting_population=n,
                     steps_per_epoch=steps, sample_rate=1./steps, expected_batch_size=max(1,n//steps),
                     total_epochs=cfg["num-server-rounds"]*cfg["local-epochs"],
                     total_steps=steps*cfg["num-server-rounds"]*cfg["local-epochs"])
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        value.update(policy_hash=hashlib.sha256(b"dsflower/effective-dpsgd-policy/v1\x00"+payload).hexdigest(),
                     independently_recomputed_replace_one_epsilon=epsilon-.01,
                     independently_recomputed_replace_one_delta=9e-6, verification_accountant="PRV", calibration="unit fixture")
        return value
    score = dict(c_index=.5, heldout_nll=1., n_evaluated_public_subjects=meta["n_test"], n_invalid_public_subjects=0)
    result = dict(seed=seed, n_train=n, minimum_site_n=min(s["n_subjects"] for s in sites), model_sha256="e"*64,
                  central=score.copy(), central_dp=score.copy(), null=score.copy(), federated_dp=score.copy(),
                  site_mechanisms=[dict(site=s["site"], **mechanism(s["n_subjects"])) for s in sites],
                  pooled_mechanism=mechanism(n), elapsed_s=1., max_rss_native_units=1,
                  memory_measurement=dict(scope="unit fixture", native_unit="bytes", federation_peak_memory_measured=False),
                  versions={name: "unit fixture" for name in ("python", "torch", "opacus", "flwr", "numpy", "scipy", "pandas",
                                                             "platform", "nonprivate_twin_device", "dp_twin_device", "federation_device_rule")},
                  twin_matching=dict(architecture_loss_preprocessing_initialization_optimizer_schedule="exact",
                                     initialization_seed=0, pooled_epochs=cfg["num-server-rounds"]*cfg["local-epochs"],
                                     differences="unit fixture"))
    result["versions"].update(deterministic_algorithms=True, cuda_available=False)
    versions = {name: "unit fixture" for name in ("dsFlower", "dsFlowerClient", "R", "DSI", "DSLite", "dsBase", "resourcer")}
    commits = dict(dsFlower="a"*40, dsFlowerClient="b"*40)
    return dict(schema_version=1, task="survival", record_type="cell", status="executed", dataset=meta,
                variant=variant, contract=contract, epsilon=epsilon, delta=1e-5, clip=1, privacy_unit="patient",
                adjacency="replace_one", site_count=3, cleanup_ok=True, public_config=cfg,
                started_utc="2026-01-01T00:00:00Z" if synthetic else "2026-01-02T00:00:00Z",
                finished_utc="2026-01-01T00:01:00Z" if synthetic else "2026-01-02T00:01:00Z", elapsed_s=60.,
                campaign_tools_commit="c"*40, package_commits=commits, runner_sha256="d"*64, package_versions=versions,
                installed_build=dict(commits=commits.copy(), runner_sha256="d"*64, built_utc="2025-12-31T00:00:00Z",
                                     package_versions=[versions["dsFlower"], versions["dsFlowerClient"]]),
                artifact_checksum="e"*64, results=result,
                federation=dict(n_clients=3, n_failures=0, n_rounds_run=cfg["num-server-rounds"], cleanup_ok=True, model_sha256="e"*64),
                score_conventions=dict(time_ties="excluded", risk_ties=.5, gap="central minus federated (historic sign reversed)",
                                       hazard_risk="negative left-endpoint restricted mean"),
                outcome_semantics=dict(target_order=["time", "event"], event=1, censored=0, time_unit="days", baseline="unit fixture",
                                       administrative_censor="unit fixture", invalid="unit fixture", preprocessing="unit fixture", interval_convention="unit fixture"),
                mechanism_provenance="unit fixture", topology="unit fixture", evaluation="channel B, public held-out subjects only")


def complete_fixture():
    records = [(f"unit-{i}.json", cell(*key)) for i, key in enumerate(sorted(CELLS))]
    groups = []
    for dataset, subset, variant in sorted(GROUPS):
        by_epsilon = {}
        for _, record in records:
            if (record["dataset"]["dataset"], record["dataset"]["subset"], record["variant"]) == (dataset, subset, variant):
                by_epsilon.setdefault(record["epsilon"], []).append(record["results"])
        env = envelopes(by_epsilon, primary=(dataset, subset) == ("support2", "full"), small_n=subset == "small600")
        groups.append(dict(dataset=dataset, subset=subset, variant=variant, status="executed", envelopes=env,
                           investigations={key: dict(status="reviewed", explanation="Unit test only; no empirical claim.")
                                           for key in flagged_keys(env)}))
    records += [("unit-synthetic-"+variant+".json", cell("synthetic-public", "synthetic", variant, 8, 1101)) for variant in VARIANTS]
    records.append(("summary.json", dict(schema_version=1, task="survival", record_type="summary", status="executed",
                                         date_utc="2026-01-03T00:00:00Z", expected_matrix_cells=90, groups=groups,
                                         failed_attempts=[], envelope_interpretation="historic sign reversed")))
    return records


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.records = complete_fixture()

    def test_complete_fixture_can_report_failed_floors(self):
        self.assertEqual(validate_completion(self.records), dict(cohort_cells=90, synthetic_cells=3, groups=12, failed_utility_floors=3))

    def test_missing_duplicate_or_wrong_seed_cell_fails(self):
        for mutation in (lambda r: r.pop(0), lambda r: r.insert(0, copy.deepcopy(r[0])),
                         lambda r: r[0][1]["dataset"].update(seed=1104)):
            records = copy.deepcopy(self.records)
            mutation(records)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_completion(records)

    def test_only_failures_cannot_complete(self):
        record = copy.deepcopy(self.records[0][1])
        record.update(status="failed", error="unit test failure")
        del record["results"]
        validate_cell(record)
        with self.assertRaisesRegex(ValueError, "90 cohort cells"):
            validate_completion([("failure.json", record)])

    def test_missing_synthetic_or_wrong_group_fails(self):
        records = copy.deepcopy(self.records)
        records.pop(90)
        with self.assertRaisesRegex(ValueError, "three executed synthetic"):
            validate_completion(records)
        self.records[-1][1]["groups"][0]["dataset"] = "other"
        with self.assertRaisesRegex(ValueError, "12 preregistered groups"):
            validate_completion(self.records)

    def test_ci_tampering_and_unreviewed_flags_fail(self):
        records = copy.deepcopy(self.records)
        records[-1][1]["groups"][0]["envelopes"]["summaries"]["8"]["gap"]["ci95"][0] += .01
        with self.assertRaisesRegex(ValueError, "differs from executed replicates"):
            validate_completion(records)
        group = next(g for g in self.records[-1][1]["groups"] if g["investigations"])
        next(iter(group["investigations"].values()))["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "reviewed explanation"):
            validate_completion(self.records)

    def test_subject_census_build_artifact_and_horizon_bindings(self):
        mutations = [lambda r: r["results"]["site_mechanisms"][0].update(accounting_population=999),
                     lambda r: r["installed_build"].update(runner_sha256="f"*64),
                     lambda r: r.update(artifact_checksum="f"*64),
                     lambda r: r["public_config"].update({"num-server-rounds": 1}),
                     lambda r: r["results"]["site_mechanisms"][0].update(total_epochs=2),
                     lambda r: r["results"]["site_mechanisms"][0].update(policy_hash="f"*64),
                     lambda r: r["dataset"]["features"].append("hospdead")]
        for mutation in mutations:
            record = copy.deepcopy(self.records[0][1])
            mutation(record)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_cell(record)

    def test_missing_versions_and_null_score_fail(self):
        record = copy.deepcopy(self.records[0][1])
        record["results"]["versions"]["pandas"] = ""
        with self.assertRaisesRegex(ValueError, "runtime version"):
            validate_cell(record)
        record = copy.deepcopy(self.records[0][1])
        record["results"]["central_dp"]["c_index"] = None
        with self.assertRaisesRegex(ValueError, "finite executed score"):
            validate_cell(record)

    def test_json_integer_noise_retains_runtime_float_policy_hash(self):
        record = copy.deepcopy(self.records[0][1])
        for value in record["results"]["site_mechanisms"] + [record["results"]["pooled_mechanism"]]:
            value["noise_multiplier"] = 2
        validate_cell(record)


if __name__ == "__main__":
    unittest.main()
