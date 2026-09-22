#!/usr/bin/env python3
"""Write the training-only R5 diagnosis from completed, immutable result records."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def number(value, digits=6):
    return f"{float(value):.{digits}f}"


def batch_name(candidate):
    return "full site (227)" if candidate["batch_size"] == 227 else str(candidate["batch_size"])


def metric_row(row):
    score, mechanism = row["inner_validation"], row["mechanism"]
    return (f"| {row['candidate']['id']} | {row['epsilon']} | {mechanism['total_steps']} | "
            f"{number(mechanism['noise_multiplier'])} | {number(score['auc'])} | "
            f"{number(score['accuracy'])} | {number(score['brier'])} | {number(score['log_loss'])} |")


def optional_record(results, names):
    for name in names:
        path = results / name
        if path.exists():
            return name, read(path)
    return None, None


def build(results):
    ranked = read(results / "ranked-candidates.json")
    declaration = read(results / "search-declaration.json")
    validation = read(results / "validation.json")
    confirmations = read(results / "dp-confirmation.json")
    selection = read(results / "selection.json")
    selected = selection["selected"]
    candidate, mechanism = selected["candidate"], selected["mechanism"]
    best = selected["inner_validation"]
    actual8 = [row for row in confirmations if row["epsilon"] == 8]
    assert validation["passed"] and len(ranked) == len(declaration["candidates"]) == 736
    assert len(actual8) == 3
    assert {row["candidate"]["id"] for row in actual8} == {row["candidate"]["id"] for row in ranked[:3]}
    assert selected == sorted(actual8, key=lambda row: (-row["inner_validation"]["auc"], row["candidate"]["id"]))[0]
    assert all(row["feature_parity"] and not row["test_accessed"] for row in confirmations)
    assert not validation["test_accessed"] and not selection["test_accessed"]
    if best["auc"] < .60:
        assert {row["epsilon"] for row in confirmations if row["candidate"]["id"] == candidate["id"]} == {1, 4, 8}
    real_by_id = {row["candidate"]["id"]: row for row in actual8}
    proceed = best["auc"] >= .60
    lines = [
        "# Vision R5: diagnosis and selection under the privacy contract",
        "",
        f"Selected `{candidate['id']}` by the highest actual ε=8 federated-DP inner-validation AUC: "
        f"**{number(best['auc'])}**, accuracy {number(best['accuracy'])}, Brier {number(best['brier'])}, "
        f"log loss {number(best['log_loss'])}. "
        + ("This exceeds the prespecified 0.60 diagnostic threshold." if proceed else
           "The best confirmed result does not reach the prespecified 0.60 diagnostic threshold."),
        "",
        "This is a training-only diagnosis and schedule selection. No final cell was run. "
        "The fixed outer split remains 852 training patients and 212 test patients, unopened in R4/R5; "
        "the R4 inner split remains 681 fitting patients, three sites of 227, and 171 validation patients. "
        "All reported new evaluation uses cached training-patient features. The original R3/R4 records are retained byte-for-byte.",
        "",
        "## Verified contract and the R4 failure",
        "",
        "The unchanged contract is `pytorch_resnet18`: frozen ImageNet ResNet18, patient-mean 512-dimensional "
        "features and deterministic modal patient labels, two-logit affine head with its architectural "
        "FiniteClamp output in [-30, 30], five rounds, equal weight 1 for each of three sites, "
        "δ=10⁻⁶ and clipping norm 1. The backbone uses the runner's per-image preprocessing. "
        "Its pooled features are not normalized to [0, 1]: the cached maximum is approximately 8.587. "
        "Finite totalization bounds are [-10⁶, 10⁶].",
        "",
        "For site population n and requested batch b, the runner uses s=ceil(n/b), "
        "Poisson inclusion q=1/s, expected-batch mean divisor max(1, floor(n/s)), and "
        "5 × local_epochs × s accounted updates per site. The actual q is not b/n at ceil boundaries. "
        "Each sample gradient is coordinate-clamped to [-1, 1], then clipped to global L2 norm 1 "
        "across weight and bias; independent Gaussian noise is added to the summed gradients before "
        "division. The accountant calibrates replace-one privacy, converting to add/remove "
        "(ε/2, δ/(1+exp(ε/2))) before Opacus calibration. SGD momentum or Adam moments reset each round. "
        "The emulator retains finite-output derivatives, parameter saturation and weight decay after noising.",
        "",
        "| Inner requested batch | Updates per local epoch | Executed q | Mean divisor |",
        "|---:|---:|---:|---:|",
        "| 32 | 8 | 0.125 | 28 |",
        "| 64 | 4 | 0.25 | 56 |",
        "| 128 | 2 | 0.5 | 113 |",
        "| full site = 227 | 1 | 1 | 227 |",
        "",
        "R4 pruned 25 schedules without privacy and selected 20 local epochs with Adam at 0.003/0.01, "
        "batch 32. Under the real inner contract those schedules require 800 updates per site, "
        "q=0.125 and noise multiplier 4.98046875; their recorded AUCs were 0.500940 and 0.478683. "
        "The high update count raises the accountant's required noise, and Adam's coordinate scaling "
        "changes the impact of noisy gradients. Non-private pruning therefore did not identify a useful "
        "private schedule. Those results motivated direct DP-aware ranking here. The R4 converged "
        "central logistic head reached 0.779405 ± 0.025903 inner-CV AUC, so the frozen representation carries signal.",
        "",
        "## Emulator validation",
        "",
        "The emulator and unchanged runner were compared over all five rounds using captured initial "
        "arrays, exact cached-feature/label hashes, the same accountant, and identical fresh independent "
        "per-site/per-round SecureNumpyRng sampling and Gaussian streams. The actual historical custodial "
        "secrets were not reconstructed. These numerical parity checks precede ranking.",
        "",
        "The reordered outer feature cache did not match the captured inner-site tensor hashes. "
        "Inner training features were therefore re-extracted from the same raw inner collections in "
        "their exact manifest order and verified against the actual R4 captures. The outer cache remains "
        "the source for the unchanged 171 validation patients and the R3 training-only comparison. "
        "`inner-feature-parity.json` records this cache correction; no test data was involved.",
        "",
        "| Historical schedule | Updates/site | σ | Maximum CUDA parameter error | Maximum CPU parameter error | Maximum probability error | Passed |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for check in validation["numerical_parity"]:
        rounds = check["rounds"]
        parameter_cuda = max(row["cuda_parameter_max_abs"] for row in rounds)
        parameter_cpu = max(row["cpu_parameter_max_abs"] for row in rounds)
        probability = max(max(row["cuda_probability_max_abs"], row["cpu_probability_max_abs"]) for row in rounds)
        m = check["mechanism"]
        lines.append(f"| {check['name']} | {m['total_steps']} | {number(m['noise_multiplier'])} | "
                     f"{parameter_cuda:.3g} | {parameter_cpu:.3g} | {probability:.3g} | {check['passed']} |")
    lines += ["", "The absolute pass tolerance is 10⁻⁴. Independent-noise comparisons use 12 fresh replicates per schedule:", "",
        "| Comparison population/schedule | Historical AUC | Emulated mean ± sample SD | Emulated range | Historical AUC in range |",
        "|---|---:|---:|---:|:---:|"]
    for row in validation["historic_r4_inner_validation"] + [validation["historic_r3_training_only"]]:
        bounds = row["empirical_auc_range"]
        scope = "R3, 852 training patients" if row is validation["historic_r3_training_only"] else "R4, 171 inner-validation patients"
        lines.append(f"| {scope}; {row['candidate']['id']} | {number(row['historical_metrics']['auc'])} | "
                     f"{number(row['mean_auc'])} ± {number(row['sd_auc'])} | "
                     f"{number(bounds[0])}–{number(bounds[1])} | {row['historical_auc_within_empirical_range']} |")
    lines += ["", "The R3 default check uses its actual n=284 site mechanism (45 updates, q=1/9, divisor 31), "
        "captured initialization and training features. The previously published R3 test AUC 0.533 is "
        "context only: no test records or predictions were opened, and that test AUC was not recomputed. "
        "Historical-noise compatibility is descriptive; numerical parity is the emulator validation gate."]
    clamp_path = results / "finiteclamp-validation.json"
    if clamp_path.exists():
        clamp = read(clamp_path)
        lines += ["", "The additional FiniteClamp stress check verifies the output-clamp derivative in the "
            "high-learning-rate regime. Its recorded results are:", "", "```json", json.dumps(clamp, indent=2), "```"]
    rank_check = read(results / "shortlist-validation.json")
    assert rank_check["passed"]
    comparisons = [r[d] for c in rank_check["checks"] for s in c["replications"]
        for r in s["rounds"] for d in ("cpu", "cuda")]
    lines += ["", "A further check covers all three shortlisted schedules, all three search seeds and "
        "all five rounds, comparing unchanged CUDA training with CPU/CUDA emulation. It checks native "
        "prediction ranks because tiny saturated probabilities can hide ranking errors in an absolute "
        f"probability tolerance. Maximum observed AUC difference: {max(r['auc_abs_difference'] for r in comparisons):.9g}; "
        f"maximum cross-class pair-order disagreements: {max(r['cross_class_pair_order_disagreements'] for r in comparisons)}. "
        "Full results are in `shortlist-validation.json`."]
    aggregation = read(results / "aggregation-validation.json")
    assert aggregation["passed"]
    aggregate_rows = [r for c in aggregation["checks"] for s in c["replications"]
        for r in s["rounds"] + s["final_round_reply_orders"]]
    lines += ["", "Installed Flower normalizes each unit weight to 1/3 before addition; the emulator "
        "adds then divides. A separate five-round check uses the installed Flower aggregator and "
        "unchanged CUDA training for all shortlisted schedules and search seeds. It also tests all "
        "six final-round reply orders on the same node arrays. Maximum AUC difference is "
        f"{max(r['auc_abs_difference'] for r in aggregate_rows):.9g}; maximum parameter error is "
        f"{max(r['parameter_max_abs'] for r in aggregate_rows):.3g}. "
        "`aggregation-validation.json` records this float32 arithmetic check. Earlier-round arrival "
        "orders use canonical site order; historical secret streams are not replayed."]
    lines += ["", "## Search and real confirmation", "",
        "The complete declared grid has 736 schedules: SGD with momentum 0 or 0.9 at learning rates "
        "{0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}; Adam at "
        "{0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1}; local epochs {1, 2, 4, 8}; "
        "batch {32, 64, 128, full site}; weight decay {0, 0.0001}. Scheduler is none, L1 is zero, "
        "and five rounds are fixed. Ranking uses ε=8 mean inner-validation AUC over three "
        "independent noise/sampling seeds " + ", ".join(map(str, declaration["noise_seeds"])) +
        "; the captured initialization seed is 20260922. Ties use ascending candidate ID.", "",
        "The three highest emulated means were confirmed once each through the actual dsFlower federation. "
        "All 15 node-round captures per federation verify feature/target hashes, optimizer pins, "
        "observed updates, privacy configuration and accountant geometry. Selection uses the highest "
        "real ε=8 AUC; the same ID tie-break applies.", "",
        "| Actual candidate | ε | Updates/site | σ | AUC | Accuracy | Brier | Log loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines.extend(metric_row(row) for row in confirmations)
    largest_gap = max(actual8, key=lambda r: abs(r['inner_validation']['auc'] -
        next(e['mean_auc'] for e in ranked if e['candidate']['id'] == r['candidate']['id'])))
    predicted = next(r['mean_auc'] for r in ranked if r['candidate']['id'] == largest_gap['candidate']['id'])
    lines += ["", f"The largest confirmation discrepancy is `{largest_gap['candidate']['id']}`: "
        f"emulated mean AUC {number(predicted)}, real AUC {number(largest_gap['inner_validation']['auc'])}. "
        "Three emulated seeds did not characterize the full observed outcome spread. The real runs "
        "use fresh custodial randomness, and one real fit per candidate cannot separate a small "
        "weight-decay effect from noise-realization variability. The actual feature, mechanism and "
        "optimizer checks passed; no scored configuration was repeated or adjusted."]
    boundary_gap = ranked[2]["mean_auc"] - ranked[3]["mean_auc"]
    lines += ["", f"The third/fourth emulated mean gap is only {boundary_gap:.9f}. "
        "The shortlist is the declared numerical ranking, not evidence that the third schedule is "
        "materially better than the fourth. In emulation, all three shortlisted candidates have "
        "threshold-0.5 accuracy equal to the inner majority rate (116/171 = 0.678363), Brier about "
        "0.3214–0.3216, and log loss about 2.5–11.1. A positive PROCEED decision establishes "
        "above-chance discrimination under the contract; it does not establish calibrated "
        "probabilities or useful threshold-0.5 classification."]
    lines += ["", ("Because the best ε=8 AUC was below 0.60, the selected schedule was also confirmed at ε=4 "
        "and ε=1; those real results above characterize budget dependence. No candidate reached the "
        "diagnostic threshold among the three real confirmations selected from this declared finite grid. "
        "This does not prove that every point in the contract's continuous parameter surface is unable to learn."
        if not proceed else "The best real ε=8 AUC exceeds 0.60, so the prespecified conditional ε=4/ε=1 "
        "diagnosis was not triggered."), "",
        "These are selection estimates: 736 schedules were ranked on the same 171 validation patients, "
        "and the best of three real noisy fits was chosen on those patients. Their AUCs are optimistic "
        "for a future independent evaluation. One real fit per candidate does not estimate real-fit "
        "noise variance, and the three-seed emulation SD is not a patient-sampling confidence interval. "
        "The quoted ε is per federation, not a composed privacy budget for the entire public-data "
        "benchmark search and selection process.", "",
        "## Selected schedule and implementation comparator definitions", "",
        f"Selected optimizer **{candidate['optimizer']}**, learning rate **{candidate['learning_rate']:g}**, "
        f"momentum **{candidate.get('momentum', 0):g}**, local epochs **{candidate['local_epochs']}**, "
        f"batch **{batch_name(candidate)}**, weight decay **{candidate['weight_decay']:g}**, five rounds. "
        f"At the inner ε=8 contract: {mechanism['total_steps']} updates/site, "
        f"q={mechanism['sample_rate']:g}, divisor {mechanism['expected_batch_size']}, "
        f"σ={number(mechanism['noise_multiplier'])}. The noise standard deviation per mean-gradient "
        f"coordinate is {number(mechanism['noise_multiplier']/mechanism['expected_batch_size'])}, "
        "compared with 0.177874 in R4. Batch geometry, learning rate and optimizer all change across "
        "these schedules; this comparison does not isolate an individual causal effect.", "",
        "The batch search dimension `full site` is a prespecified logical schedule choice: 227 on the "
        "inner sites, 284 on each original training site, and 852 for the pooled-DP twin. It retains "
        "q=1 and the same updates per local epoch. Numeric batch choices 32, 64 and 128 remain literal "
        "when moving to the full training cohort. This rule is fixed at diagnosis; no implementation "
        "or selected-schedule fit on the outer cohort has been performed. The R3 check above was "
        "diagnostic replay of its default schedule on training features only."]
    outer_batch = 284 if candidate["batch_size"] == 227 else candidate["batch_size"]
    pooled_batch = 852 if candidate["batch_size"] == 227 else candidate["batch_size"]
    outer_steps, pooled_steps = math.ceil(284 / outer_batch), math.ceil(852 / pooled_batch)
    lines += ["", f"For this selected schedule the later full-cohort federation would use batch {outer_batch}, "
        f"{outer_steps * candidate['local_epochs'] * 5} updates/site, q={1 / outer_steps:.9g}, "
        f"divisor {284 // outer_steps}. Its noise multiplier must be recalibrated for that mechanism. "
        f"The pooled-DP twin would use batch {pooled_batch}, {pooled_steps * candidate['local_epochs'] * 5} "
        f"updates, q={1 / pooled_steps:.9g}, divisor {852 // pooled_steps}; it requires its own n=852 accountant calibration.", "",
        "| Comparator | Fixed definition for the implementation |",
        "|---|---|",
        "| Converged central logistic head | Fit a pooled C=1 L2 logistic head on the 852 frozen patient feature vectors; unpenalized intercept, converged solver, same malignant-positive task. This is a representation/optimization comparator, not a finite-step twin. |",
        "| Non-private federated finite-schedule twin | Three original 284-patient sites, the selected logical schedule, same initial affine head and architectural FiniteClamp, same Poisson sampling geometry, expected-batch division, equal site weights and optimizer reset each round. Remove DP per-gradient coordinate/global-norm clipping and Gaussian noise; retain the selected regularization and public numerical totalization. |",
        "| Pooled-DP finite-schedule twin | Pool all 852 training patients; same selected optimizer, learning rate, local epochs, five optimizer-reset rounds, logical batch rule, initial head, architecture, patient preprocessing and DP gradient operations. Recompute q, divisor, updates and σ using n=852 with its own ε/δ accountant. Do not reuse the inner-site noise multiplier. |",
        "| Trivial majority | For primary image metrics, predict the training-image majority class and training-image prevalence of malignancy as the constant probability, matching R4. For secondary patient metrics, use the 852-patient training majority and prevalence. Determine neither class nor probability from test patients. |", "",
        "The implementation retains the previously declared per-image primary metrics, patient-level "
        "secondary metrics and the already fixed 852/212 split. Selection here uses patient-level AUC. "
        "Implementation training must extract features afresh; diagnosis caches remain diagnostic inputs only. "
        "These comparator definitions do not authorize test access within this diagnosis.", "",
        "## Pod and record state", ""]
    preflight_name, preflight = optional_record(results, ["runtime-preflight.json", "preflight.json", "runtime_preflight.json"])
    if preflight is not None:
        lines += [f"Runtime/package verification is recorded in `{preflight_name}`:", "", "```json", json.dumps(preflight, indent=2), "```", ""]
    else:
        lines += ["Runtime/package preflight record was not found in this results directory; attach the archived verification before publication.", ""]
    finish = read(results / "runner-final.json")
    assert finish["status"] == "verified"
    lines += ["Final package versions and both installed runner hashes were reverified in `runner-final.json`. "
        "The original 45-second Torch import probe timed out without reporting an import error; its "
        "record is preserved in `runner-final-import-timeout.json`. The identical import probe passed "
        "with a 120-second allowance. No package or model was changed or rerun.", ""]
    pod_name, pod = optional_record(results, ["pod-final.json", "pod-state-final.json", "pod-final-state.json", "pod-state-after.json", "pod-after.json"])
    if pod is not None:
        lines += [f"Final pod state is recorded in `{pod_name}`: pod left running = {pod.get('pod_left_running')}; "
            f"matching training processes = {len(pod.get('matching_processes', []))}. "
            + str(pod.get("gpu", "")).strip().replace("\n", "; "), ""]
    else:
        lines += ["Final pod process-state record was not found in this results directory; verify and attach it before publication.", ""]
    lines += ["The isolated vision pod is `/workspace/cells-vision` on `pod-flower-vision` (A40). "
        "The real federation drivers record successful SuperLink/SuperNode cleanup. The pod is left running; "
        "no other pod is part of this work. Package source is unchanged. R3/R4 records and failed-attempt "
        "evidence are preserved; R5 tooling and new records are separate. `preservation.json` verifies "
        "111 pre-existing local vision files byte-for-byte, including pre-existing uncommitted R4 "
        "working edits. Those existing edits remain unchanged and outside the R5 commit.", "",
        "## Complete ranked DP-aware grid", "",
        "Mean ± SD is across the three emulated ε=8 noise seeds. Updates and σ are per inner site "
        "over all five rounds. A dash means no actual federation was run for that candidate. "
        "The table contains every declared candidate in the recorded rank order.", "",
        "| Rank | Candidate | Optimizer | Momentum | LR | Local epochs | Batch | Weight decay | Updates/site | σ | Emulated AUC mean ± SD | Real ε8 AUC |",
        "|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(ranked, 1):
        c, m = row["candidate"], row["mechanism"]
        real = real_by_id.get(c["id"])
        real_auc = number(real["inner_validation"]["auc"]) if real else "—"
        lines.append(f"| {rank} | {c['id']} | {c['optimizer']} | {c.get('momentum', 0):g} | "
            f"{c['learning_rate']:g} | {c['local_epochs']} | {batch_name(c)} | {c['weight_decay']:g} | "
            f"{m['total_steps']} | {number(m['noise_multiplier'])} | {number(row['mean_auc'])} ± "
            f"{number(row['sd_auc'])} | {real_auc} |")
    lines += ["", "PROCEED: yes" if proceed else "PROCEED: no"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", "--root", dest="results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "VISION_DIAGNOSIS_R5.md"
    output.write_text(build(args.results))
    print(output)
