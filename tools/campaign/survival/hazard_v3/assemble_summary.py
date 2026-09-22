#!/usr/bin/env python3
"""Write the reviewable report from locked, verified public evidence."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive',type=Path);args=ap.parse_args();root=args.archive
    summary=json.loads((root/'summary.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    audit=json.loads((root/'artifact_verification.json').read_text())
    assert summary['status']=='executed' and audit['verified_confirmation_cells']==summary['expected_matrix_cells']
    cfg=selection['selected']
    path=root/'HAZARD_V3_SUMMARY.md'
    diagnosis=path.read_text().split('## Development and confirmation')[0].rstrip()
    rows=json.loads((root/'development_diagnostics.json').read_text())['results']
    text=[diagnosis,'','## Development-only mechanism controls','',
        'These controls were declared before any live development score returned. They use only inner training/validation data. '
        'They are unnoised public linear-SGD calculations, with direct patient gradients checked against the frozen autograd loss; '
        'they are not actual federations or DP releases and never enter selection.','',
        '| Grid | Pooled, full step count | Pooled, site step count | Unnoised equal-site averaging | Same, unit-clipped |',
        '|---|---:|---:|---:|---:|']
    for cid in ('g01','g02'):
        group=[r for r in rows if r['config']==cid]
        values=[statistics.mean(r['controls'][k]['c_index'] for r in group) for k in group[0]['controls']]
        text.append('| '+('K10 equal-width' if cid=='g01' else 'K5 event quantiles')+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    text+=['',
        'For h06, reducing only the pooled sequential update count explains a 0.028008 C-index drop on these development splits; '
        'equal-site averaging differs from that step-matched pooled control by +0.000072. '
        'The h06 initial mean gradient norm is 0.3070/0.3081 and maximum 0.7664/0.7647; no sampled gradient exceeds 1 throughout these unnoised controls. '
        'This supports insufficient sequential optimization as the dominant tested explanation, rather than heavy clipping or equal-site weighting itself. '
        'It does not assign an exact causal fraction of the historical noisy test gap.', '',
        'At K10, first-bin event counts are 2761/2741; last-bin counts 19/16, with 221/216 exposed patients. '
        'K5 quantiles redistribute events to roughly 772–831 per interval. K5 initial clipping fractions are23.52%/23.69%, '
        'but only 0.150%/0.142% of sampled visits exceed1 over the clipped unnoised trajectories. '
        'Grid coarsening improves both pooled and federated controls; its effect is not simply removing clipping.','',
        '## Frozen development sweep','',
        f'{len(selection["ranked"])} configurations × two seeds (1101/1102), epsilon 8, '
        'three sites of 1942 inner-training patients and 1458 pooled validation patients. '
        'Every selected score comes from actual isolated DSLite federation with patient DP. '
        'All 35 planned configurations and per-seed quantile boundaries are in `frozen_configurations.json`; '
        '`sweep.csv` and `development/` retain the full executed table and cell records. '
        f'{len(selection["omitted_configurations"])} configurations were omitted by the predeclared time cutoff.', '',
        'Selection: maximum mean federated-DP inner C; exact ties use fewer total epochs, smaller K, then ID. '
        'No pooled/control score or outer confirmation metric entered selection. Public-bound scaling was already enabled and remains fixed. '
        'Quantile edges use each seed’s inner-training events, never validation events; the resulting public grids remain unchanged at confirmation.', '',
        '| Rank | ID | K/grid | Strategy | Optimizer/LR | Rounds × local epochs | Batch | Seed1101 C | Seed1102 C | Mean C |',
        '|---:|---|---|---|---|---|---:|---:|---:|---:|']
    for rank,row in enumerate(selection['ranked'],1):
        c=row['config']
        text.append(f'| {rank} | {c["id"]} | {c["K"]}/{c["grid"]} | {c["strategy"]} | {c["optimizer"]}/{c["learning_rate"]} | {c["rounds"]} × {c["local_epochs"]} | {c["batch_size"]} | {row["scores"]["1101"]:.6f} | {row["scores"]["1102"]:.6f} | {row["mean"]:.6f} |')
    text+=['','## One confirmation pass','',
        f'Selected **{cfg["id"]}: K={cfg["K"]}, {cfg["grid"]}, {cfg["strategy"]}, '
        f'{cfg["optimizer"]} LR {cfg["learning_rate"]}, {cfg["rounds"]} rounds × {cfg["local_epochs"]} local epochs, '
        f'batch {cfg["batch_size"]}**. Selection locked at {selection["selected_utc"]}. '
        'Each twin matches the grid, architecture, public feature transform, initialization 0, local optimizer reset schedule and server post-processing; '
        'pooled q and sequential step counts differ by population. Seeds define subject splits, not published DP noise seeds.', '',
        '**This is the third confirmation of the hazard contract, after the v1 matrix and v2 schedule h06.** '
        'The historical holdouts are reused, not new independent validation. Development opens only the training files for each split; '
        'replicate subject splits overlap, so all intervals and the final verdict remain descriptive benchmark evidence. '
        'There was one selected configuration and one confirmation matrix, with no confirmation-driven tuning or repeat fit.', '',
        '| Arm | ε | Federated-DP C | Pooled-DP C | Pooled nonprivate C | Null C |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary['groups']:
        values=' | '.join(f'{row["c_index"][b]["mean"]:.6f} ± {row["c_index"][b]["sd"]:.6f}' for b in ('federated_dp','central_dp','central','null'))
        text.append(f'| {row["arm"]} ({row["n_sites"]} sites; '+','.join(map(str,row['site_n']))+f'/site) | {row["epsilon"]} | {values} |')
    primary=next(r for r in summary['groups'] if r['arm']=='full' and r['epsilon']==8)
    c=primary['c_index']['federated_dp'];null=primary['c_index']['null']['mean']
    text+=['',f'**Three-site verdict: {summary["three_site_verdict"]}.** Mean epsilon 8 C={c["mean"]:.6f}; '
        f'null={null:.6f}; required C≥0.600000 and C≥{null+.05:.6f}. '
        f'Split-replicate 95% t interval: [{c["ci95"][0]:.6f}, {c["ci95"][1]:.6f}]. '
        'The two-site arm is a patients-per-site envelope and cannot substitute for admission of the three-site route.', '',
        'Held-out NLL, paired central/federated gaps, sample SDs, and Student-t 95% intervals are in `summary.json`. '
        'Only three overlapping split replicates support these intervals. NLL/K is not comparable between grids or likelihoods. '
        'Ranking performance does not establish probability calibration.', '',
        '## Execution incidents and validation','',
        'The preprovisioned pod lacked dsBase because Matrix 1.4-0 blocked lme4. '
        'Updating the missing R dependencies in the task library restored dsBase 6.3.5; canonical install/freeze succeeded. '
        'No package source was changed. Five campaign checks, 30 survival-contract tests, 102 DP-safety checks, '
        'runner byte synchronization and tag source fingerprints passed.', '',
        'One development attempt (g01/1102) failed at SuperNode startup, before server training submission, with no model or scores. '
        'Its unchanged failure record and redacted diagnostics are in `failures/`. The exact low-level cause is unavailable; '
        'shared spawn-lock contention is plausible, not established. Seven successful first-wave cells were reused. '
        'Only the failed pre-training attempt was recovered, once, in a new directory. '
        'Subsequent launches are spaced 30 seconds and each federation receives one CPU by OS affinity; '
        'the trusted launcher strips OMP/BLAS environment settings, so those alone did not limit the original workers. '
        'The grid, split data, privacy mechanism and selection rule stayed fixed.', '',
        'A synthetic artifact-copy path error occurred after successful fitting/scoring; the existing model was exported without retraining. '
        'Unit-test import resolution and a NumPy alias in the test expectation were corrected before cohort fitting. '
        f'All {summary["expected_matrix_cells"]} confirmation cells were validated and all four model artifacts per cell were independently reloaded/rescored. '
        'No confirmation fit was repeated. Details are in `artifact_verification.json` and `infrastructure_investigation.json`.', '',
        '## Provenance','',
        '- Packages: dsFlower 0.5.0 and dsFlowerClient 0.5.0. 101 server and 105 client source files match tag v0.5.0; full fingerprints and installed commits are in `provenance/`.',
        '- Runner SHA256: `2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.',
        '- Pod: `x0w6ewmpinpsuk`, Ubuntu 22.04, 32 available CPUs, 64000000000-byte memory limit, CPU only; left running.',
        '- Runtime: R 4.6.1, Python 3.11.10, Torch 2.4.1+cpu, Opacus 1.5.2, Flower 1.31.0. Full Python/R versions are archived.',
        '- SUPPORT2 SHA256: `9da794bbd5c3a6a816e677cc17535e58c122d9ef4cbefd404489330a9f9cd2de`; UCI source and license attribution retained in each cell.',
        '- Development seeds 1101/1102; confirmation 1101/1102/1103. All three outer train/test file hashes reproduce the historical splits exactly.',
        '- `preregistration.json`, `frozen_configurations.json`, `selection.json`, confirmation start/completion records and per-cell hashes bind the execution order and settings.',
        '- Historical v2 ran on CUDA A40 with package labels 0.4.5/0.4.4 and runner `ac08384…e4ac8`; current 0.5.0 source and CPU runtime are recorded separately. Cross-version changes are not a controlled device-only comparison.',
        '- Each reported epsilon is a per-fit contract, not a privacy budget for the public development/confirmation campaign as a whole. Private-data selection would compose; data-derived private quantile grids would need their own accounted release.',
        '- Only public evidence and whitelisted released models are archived. No node secrets or staged private manifests are exported.','']
    path.write_text('\n'.join(text))


if __name__=='__main__':main()
