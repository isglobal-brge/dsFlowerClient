#!/usr/bin/env python3
"""Write the reviewable report from locked, verified public evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive',type=Path);args=ap.parse_args();root=args.archive
    summary=json.loads((root/'summary.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    audit=json.loads((root/'artifact_verification.json').read_text())
    overlap=json.loads((root/'split_overlap.json').read_text())
    assert summary['status']=='executed' and audit['verified_confirmation_cells']==summary['expected_matrix_cells']
    assert audit['verified_development_cells']==2*len(selection['ranked'])
    cfg=selection['selected']
    strategy_settings={
        'fedavg':'fixed unit site weights',
        'fedavgm':'server learning rate 1, server momentum 0.9, fixed unit site weights',
        'fedadam':f'server eta 0.1, eta_l {cfg["learning_rate"]}, beta1 0.9, beta2 0.99, tau 0.001, fixed unit site weights',
        'fedyogi':f'server eta 0.01, eta_l {cfg["learning_rate"]}, beta1 0.9, beta2 0.99, tau 0.001, fixed unit site weights'}
    path=root/'HAZARD_V3_SUMMARY.md'
    initial=(root/'diagnosis_before_training.md').read_bytes()
    preregistration=json.loads((root/'preregistration.json').read_text())
    assert hashlib.sha256(initial).hexdigest()==preregistration['diagnosis_sha256']
    diagnosis=initial.decode().split('## Development and confirmation')[0].rstrip()
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
    geometry=json.loads((root/'gradient_geometry.json').read_text())['rows']
    text+=['', 'Initial gradient geometry on the same inner training splits (1942 patients/site):', '',
        '| Contract | Parameters | Patients clipped at initialization | Norm of mean clipped gradient | Site noise RMS L2 per update |',
        '|---|---:|---:|---:|---:|']
    for name in ('hazard','lognormal','weibull'):
        entries=[r for r in geometry if r['variant']==name]
        avg=lambda key:statistics.mean(r[key] for r in entries)
        label='hazard h06 (K10)' if name=='hazard' else name
        text.append(f'| {label} | {entries[0]["n_parameters"]} | {100*avg("fraction_clipped_at_initialization"):.2f}% | {avg("norm_of_mean_clipped_gradient"):.6f} | {avg("site_noise_rms_l2"):.6f} |')
    text+=['', 'This is an initial geometry diagnostic, not a final-model signal-to-noise decomposition. '
        'AFT clips far more patients initially yet has denser, stronger mean signal and much smaller total noise magnitude. '
        'The hazard problem is weak /K signal relative to fixed unit-clip noise across many coordinates, not large raw covariate scales.', '',
        'For h06, reducing only the pooled sequential update count explains a 0.028008 C-index drop on these development splits; '
        'equal-site averaging differs from that step-matched pooled control by +0.000072. '
        'The h06 initial mean patient-gradient norm is 0.3070/0.3081 and maximum 0.7664/0.7647; no sampled gradient exceeds 1 throughout these unnoised controls. '
        'This supports insufficient sequential optimization as the dominant tested explanation, rather than heavy clipping or equal-site weighting itself. '
        'It does not assign an exact causal fraction of the historical noisy test gap.', '',
        'At K10, first-bin event counts are 2761/2741; last-bin counts 19/16, with 221/216 exposed patients. '
        'K5 quantiles redistribute events to roughly 772–831 per interval. K5 initial clipping fractions are 23.52%/23.69%, '
        'but only 0.150%/0.142% of sampled visits exceed 1 over the clipped unnoised trajectories. '
        'Grid coarsening improves both pooled and federated controls; its effect is not simply removing clipping.','',
        '## Frozen development sweep','',
        f'{len(selection["ranked"])} configurations × two seeds (1101/1102), epsilon 8, '
        'three sites of 1942 inner-training patients and 1458 pooled validation patients. '
        'Every selected score comes from actual isolated DSLite federation with patient DP. '
        'All 35 planned configurations and per-seed quantile boundaries are in `frozen_configurations.json`; '
        '`sweep.csv` and `development/` retain the full executed table and cell records; `sweep-full.csv` also marks every unstarted planned configuration. '
        f'{len(selection["omitted_configurations"])} configurations were omitted by the predeclared time cutoff.', '',
        'No new wave starts at or after 02:40 UTC; every started pair and wave must finish. '
        'The optional envelope arms are included only when selection locks before 02:50 UTC. '
        'Both time rules were fixed before the sweep and operate before any confirmation outcome is available.', '',
        'Selection: maximum mean federated-DP inner C; exact ties use fewer total epochs, smaller K, then ID. '
        'No pooled/control score or outer confirmation metric entered selection. Public-bound scaling was already enabled and remains fixed. '
        'Quantile edges use each seed’s inner-training events, never validation events; the resulting public grids remain unchanged at confirmation.', '',
        '| Rank | ID | K/grid | Strategy | Optimizer/LR | Rounds × local epochs | Batch | Seed1101 C | Seed1102 C | Mean C |',
        '|---:|---|---|---|---|---|---:|---:|---:|---:|']
    for rank,row in enumerate(selection['ranked'],1):
        c=row['config']
        text.append(f'| {rank} | {c["id"]} | {c["K"]}/{c["grid"]} | {c["strategy"]} | {c["optimizer"]}/{c["learning_rate"]} | {c["rounds"]} × {c["local_epochs"]} | {c["batch_size"]} | {row["scores"]["1101"]:.6f} | {row["scores"]["1102"]:.6f} | {row["mean"]:.6f} |')
    dev={row['config']['id']:row['mean'] for row in selection['ranked']}
    if all(key in dev for key in ('g02','g06','g16','g18','g20')):
        text+=['',f'For the same K5 quantile grid and plain SGD, the baseline mean is {dev["g02"]:.6f}. '
            f'Doubling learning rate gives {dev["g06"]:.6f}; halving batch size gives {dev["g16"]:.6f}; '
            f'doubling total epochs gives {dev["g20"]:.6f}. '
            f'Doubling learning rate while halving total epochs leaves learning rate × update count unchanged and gives {dev["g18"]:.6f}. '
            'These development comparisons support the update-count diagnosis, together with the unnoised controls. '
            'Batch/epoch changes also change calibrated noise, and all DP draws are independent; these are not isolated noise-only effects.']
    text+=['','## One confirmation pass','',
        f'Selected **{cfg["id"]}: K={cfg["K"]}, {cfg["grid"]}, {cfg["strategy"]}, '
        f'{cfg["optimizer"]} LR {cfg["learning_rate"]}, {cfg["rounds"]} rounds × {cfg["local_epochs"]} local epochs, '
        f'batch {cfg["batch_size"]}**. Selection locked at {selection["selected_utc"]}. '
        f'Aggregation settings: {strategy_settings[cfg["strategy"]]}. '
        'Each twin matches the grid, architecture, public feature transform, fixed initialization seed 0, local optimizer reset schedule and server post-processing; '
        'pooled q and sequential step counts differ by population. Seeds define subject splits, not published DP noise seeds.', '',
        'Optional envelope arms omitted by the predeclared time rule: '+
        (', '.join(arm for arm in ('heterogeneous','small600') if arm not in selection['confirmation_arms']) or 'none')+'.', '',
        '**This is the third confirmation of the hazard contract, after the v1 matrix and v2 schedule h06.** '
        'The historical holdouts are reused, not new independent validation. Development opens only the training files for each split. '
        'This is disjointness within each seed, not global disjointness across seeds: another seed’s training set can include a held-out patient. '
        'The shared configuration is selected across development replicates, so this is not independent nested validation. '
        'All intervals and the final verdict remain descriptive benchmark evidence. '
        'There was one selected configuration and one confirmation matrix, with no confirmation-driven tuning or repeat fit.', '',
        'Training-ID-only overlap audit (no outer test file or outcome read): '+
        '; '.join(f'development {r["development_seed"]} includes {r["development_outer_training_subjects_outside_confirmation_training"]} subjects held out by seed {r["confirmation_seed"]}'
            for r in overlap['rows'] if r['development_seed']!=r['confirmation_seed'])+'. See `split_overlap.json`.', '',
        '| Arm | ε | Federated-DP C | Pooled-DP C | Pooled nonprivate C | Null C |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary['groups']:
        values=' | '.join(f'{row["c_index"][b]["mean"]:.6f} ± {row["c_index"][b]["sd"]:.6f}' for b in ('federated_dp','central_dp','central','null'))
        text.append(f'| {row["arm"]} ({row["n_sites"]} sites; '+','.join(map(str,row['site_n']))+f'/site) | {row["epsilon"]} | {values} |')
    text+=['', 'Epsilon 8 accounting geometry (seed 1101; the other seeds have the same population/schedule geometry):', '',
        '| Arm / population | Patients | Poisson q | Total sequential steps | Noise multiplier | Gradient noise SD per coordinate |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in selection['confirmation_arms']:
        record=json.loads((root/f'cell-support2-{arm}-hazard-v3-eps8-seed1101.json').read_text())
        for label,m in [('site',record['results']['site_mechanisms'][0]),('pooled',record['results']['pooled_mechanism'])]:
            text.append(f'| {arm} / {label} | {m["accounting_population"]} | {m["sample_rate"]:.8f} | {m["total_steps"]} | {m["noise_multiplier"]:.8f} | {m["noise_multiplier"]/m["expected_batch_size"]:.8f} |')
    text+=['', 'The two-site envelope changes per-site sampling, calibrated noise and sequential optimization steps together; '
        'it does not isolate a noise-only effect. Fixed unit weights remain equal patient weights for these equal-size partitions.']
    if 'heterogeneous' in selection['confirmation_arms']:
        text+=['', 'The heterogeneous arm sorts the same training patients by age before partitioning. '
            'This also reorders the pooled input, so its deterministic nonprivate minibatch path differs from the full arm. '
            'The pooled comparator is matched within each cell; the cross-arm comparison does not isolate site heterogeneity alone.']
    primary=next(r for r in summary['groups'] if r['arm']=='full' and r['epsilon']==8)
    c=primary['c_index']['federated_dp'];null=primary['c_index']['null']['mean']
    two_site=next(r for r in summary['groups'] if r['arm']=='two-sites' and r['epsilon']==8)
    text+=['',f'At epsilon 8 the selected three-site route changes mean C by {c["mean"]-.5951121873821673:+.6f} '
        f'relative to historical h06. Its pooled-DP minus federated gap is '
        f'{primary["pooled_dp_minus_federated"]["mean"]:.6f}, compared with the historical 0.025803. '
        f'The selected pooled nonprivate fit reaches {primary["c_index"]["central"]["mean"]:.6f}, '
        'so the historical 0.621839 ceiling should be interpreted as schedule/grid specific. '
        f'The two-site envelope changes federated C by {two_site["c_index"]["federated_dp"]["mean"]-c["mean"]:+.6f} '
        'relative to the selected three-site route. Configuration and runtime changes prevent treating these cross-version differences '
        'as an isolated causal effect of any single lever.']
    small=next((r for r in summary['groups'] if r['arm']=='small600'),None)
    if small:
        text+=['', f'At 200 patients/site, federated C is {small["c_index"]["federated_dp"]["mean"]:.6f} '
            f'versus pooled-DP {small["c_index"]["central_dp"]["mean"]:.6f}; '
            f'the gap widens to {small["pooled_dp_minus_federated"]["mean"]:.6f}. '
            'This is below the 0.60 absolute floor and retains a small-cohort boundary for this configuration. '
            'The same frozen public grid is used; population changes alter both calibrated noise and sequential update counts.']
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
        'Before confirmation, the pooled adaptive-strategy eta_l setting was corrected to the local learning rate to match the R API; '
        'development fits and scores do not use pooled aggregation, and the synthetic pilot used FedAvgM. '
        'All 12 strategy/learning-rate settings were checked against the installed R API; see `strategy_parameter_audit.json`. '
        f'All {summary["expected_matrix_cells"]} confirmation cells were validated and all four model artifacts per cell were independently reloaded/rescored. '
        f'The {audit["verified_development_cells"]} successful development models were also archived and independently reloaded/rescored. '
        'No confirmation fit was repeated. Details are in `artifact_verification.json` and `infrastructure_investigation.json`.', '',
        '## Provenance','',
        '- Packages: dsFlower 0.5.0 and dsFlowerClient 0.5.0. 101 server and 105 client source files match tag v0.5.0; full fingerprints and installed commits are in `provenance/`.',
        '- Runner SHA256: `2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.',
        '- Pod: `x0w6ewmpinpsuk`, Ubuntu 22.04, 32 available CPUs, 64000000000-byte memory limit, CPU only; left running.',
        '- Runtime: R 4.6.1, Python 3.11.10, Torch 2.4.1+cpu, Opacus 1.5.2, Flower 1.31.0. Full Python/R versions are archived.',
        '- SUPPORT2 SHA256: `9da794bbd5c3a6a816e677cc17535e58c122d9ef4cbefd404489330a9f9cd2de`; UCI source and license attribution retained in each cell.',
        '- Development seeds 1101/1102; confirmation 1101/1102/1103. All three outer train/test file hashes reproduce the historical splits exactly.',
        '- `preregistration.json`, `frozen_configurations.json`, `selection.json`, confirmation start/completion records and per-cell hashes bind the execution order and settings.',
        '- `diagnosis_before_training.md` preserves the exact initial summary bytes bound by the preregistration diagnosis hash.',
        '- Historical v2 ran on CUDA A40 with package labels 0.4.5/0.4.4 and runner `ac08384…e4ac8`; current 0.5.0 source and CPU runtime are recorded separately. Cross-version changes are not a controlled device-only comparison.',
        '- Each reported epsilon is a per-fit contract, not a privacy budget for the public development/confirmation campaign as a whole. Private-data selection would compose; data-derived private quantile grids would need their own accounted release.',
        '- Only public evidence and whitelisted released models are archived. No node secrets or staged private manifests are exported.','']
    path.write_text('\n'.join(text))


if __name__=='__main__':main()
