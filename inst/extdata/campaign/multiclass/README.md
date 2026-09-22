# Multiclass utility evidence: CTG and HAR561

Dataset: [UCI Cardiotocography, id 193](https://archive.ics.uci.edu/dataset/193/cardiotocography),
all 2,126 records and 21 standard CTG measurements. `NSP` is the three-class
target; `CLASS` is excluded. The official UCI CSV is used under CC BY 4.0.
Dataset DOI: [10.24432/C51S4N](https://doi.org/10.24432/C51S4N).

Reference: Ayres-de-Campos D, Bernardes J, Garrido A, Marques-de-Sá J,
Pereira-Leite L. (2000). *SisPorto 2.0: A program for automated analysis of
cardiotocograms*. Journal of Maternal-Fetal Medicine 9:311–318.

Raw CSV SHA-256:
`4648b5bf338d18f0c7e030a5d2cfb2018c831695cdb90129912cf00367c7d751`.

Each cell uses the unchanged v0.5.0 `pytorch_multiclass` defaults, three sites,
five rounds, three paired seeds and one local epoch. The split is stratified
80/20 (1,701 train, 425 test), with 566/568/567 training rows per site. The
central comparator is converged unregularized multinomial logistic regression;
the trivial comparator predicts the training majority, with training class
frequencies for AUC and log-loss. Epsilon is per node training, delta is 1e-6,
row clipping is 1. The entire campaign is not one composed epsilon guarantee.

See [reproduction and interpretation](../../../../tools/campaign/multiclass/README.md)
and the pre-scoring [protocol](../../../../tools/campaign/multiclass/protocol.json).
`summary.json` lists CTG and HAR561 separately, with three epsilon entries each;
`environment.json` records the original CTG pod,
provisioning time and dependency versions. The cell JSONs contain the complete
per-replicate metrics, splits, public TRAIN-derived feature bounds, node-reported
privacy settings, model hashes, round histories and wall-clock times.

## CTG: calibration boundary

The epsilon-8 diagnostic **failed**: accuracy equaled the majority rate in
all three replicates. Macro-AUC was above 0.5. No scored cell was rerun, no
settings were changed. The original protocol and scored CTG JSONs remain
unchanged. This is a calibration boundary: ranking retained, argmax lost,
consistent with the MLP cells' calibration observation. The single HAR561
follow-up was separately declared before its scoring under
`FLOWER_CELLS_MULTICLASS_2026-09-22`.

| Epsilon | Federated macro-AUC | Accuracy | Log-loss | Macro-AUC gap |
|---:|---:|---:|---:|---:|
| 1 | 0.8087 ± 0.0908 | 0.7788 ± 0.0000 | 0.7047 ± 0.0435 | -0.1592 ± 0.0931 |
| 4 | 0.8637 ± 0.0216 | 0.7788 ± 0.0000 | 0.6812 ± 0.0093 | -0.1041 ± 0.0211 |
| 8 | 0.8674 ± 0.0470 | 0.7788 ± 0.0000 | 0.6774 ± 0.0259 | -0.1004 ± 0.0457 |

Values are mean ± sample SD across three seeds. The central macro-AUC was
0.9679 ± 0.0025, accuracy 0.8965 ± 0.0085, and log-loss
0.2429 ± 0.0100. Trivial macro-AUC was 0.5, accuracy 0.7788 and
log-loss 0.6744. The gap is federated minus central, paired by split seed.

All 45 global rounds completed with three clients and zero reported failures.
All node policies, published defaults, split/bounds/baseline pairing, source
hashes and summary calculations passed validation. Provisioning took 627 seconds
(10 minutes 27 seconds) on pod `6aq9cxaigfwlby`; dependencies are in
`environment.json`. Both packages and the canonical runner remained unchanged.

## HAR561: subject-private alternative

[UCI Human Activity Recognition Using Smartphones, id 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
provides the 561 engineered features from the same collection used by the sequence
track. All 10,299 windows are retained: the official subject-disjoint split has
7,352 training windows from 21 subjects and 2,947 test windows from nine subjects.
The six public activities are WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,
SITTING, STANDING and LAYING. Public feature bounds are `[-1,1]`, as documented
in the archive README; bounds are not estimated from either split.

Anguita D, Ghio A, Oneto L, Parra X, Reyes-Ortiz JL. (2013). *A Public Domain
Dataset for Human Activity Recognition Using Smartphones*. ESANN 2013.
Dataset DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K).
UCI currently lists CC BY 4.0; the historical archive README separately states
noncommercial use and requires citation.

Official [download](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip) SHA-256:
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
Nested dataset ZIP SHA-256:
`2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0`.

The [HAR protocol](../../../../tools/campaign/multiclass/har561_protocol.json)
fixes three sites of seven training subjects, five rounds, three seeds, epsilon
1/4/8, delta `1e-6`, subject privacy through patient column `subject`, and clipping
norm 1. All optimization parameters retain their registry defaults;
`n_classes=6` supplies the six-class output schema. Seeds 20260820–20260822 control
subject-to-site assignments and central initialization, paired across epsilon.
Federated initialization and node-owned cryptographic DP randomness retain
release behavior. The nine fits are not one composed epsilon guarantee.

The unchanged patient path averages bounded feature vectors within each subject
and chooses the modal label (lowest label breaks ties), then clips each subject's
gradient. Thus there are **21 effective training units, seven per site**, with one
full-unit-batch update per round. Subject-modal training label counts are
`[6, 0, 0, 1, 5, 9]`; activities 2 and 3 disappear from the pooled targets. These
counts were inspected only on training data, before scoring.

The central reference is unregularized multinomial logistic regression on all
7,352 training windows with the same public bounds. It shares the split and model
family but differs in subject pooling. The gap includes that difference as well
as optimization, federation and privacy; it is not a causal privacy-only penalty.
The trivial probabilities use training-window class proportions, with their
majority as the hard prediction. All held-out metrics are window-level.

All three central models and nine federated models are trained before test
tables are opened in one final scoring run; each model is scored once. Central
scores are reused across epsilon. `har561_*.json` records the new evidence,
including `har561_environment.json` for the reused pod environment.

### Measured HAR result

The epsilon-8 diagnostic **failed**: mean accuracy was below the majority
baseline, while macro-AUC exceeded 0.5. One of three epsilon-8 replicates passed
both conditions; all three exceeded 0.5 macro-AUC. All nine planned fits completed
with the frozen settings, all were scored once, and no further alternative
was run. This result characterizes the released subject-pooling path on HAR;
it does not test grouped gradients over the original activity windows.

| Epsilon | Federated macro-AUC | Accuracy | Log-loss | Macro-AUC gap |
|---:|---:|---:|---:|---:|
| 1 | 0.5013 ± 0.1133 | 0.1463 ± 0.0710 | 8.6498 ± 3.4520 | -0.4766 ± 0.1133 |
| 4 | 0.5281 ± 0.0548 | 0.1785 ± 0.0100 | 3.0266 ± 0.8486 | -0.4499 ± 0.0548 |
| 8 | 0.5653 ± 0.0238 | 0.1564 ± 0.0472 | 2.1694 ± 0.1896 | -0.4126 ± 0.0238 |

Values are mean ± sample SD across three seeds on the fixed official split.
Central macro-AUC was 0.9779 ± 0.0000, accuracy
0.8945 ± 0.0000, and log-loss 3.3988 ± 0.0000.
Trivial macro-AUC was 0.5000, accuracy 0.1822, and log-loss
1.7899. The unregularized central fit converged but its high log-loss
also demonstrates that high ranking performance does not establish probability
calibration. No central settings were changed in response.

| Epsilon-8 seed | Macro-AUC | Accuracy | Majority accuracy | Diagnostic |
|---:|---:|---:|---:|:---|
| 20260820 | 0.592470 | 0.190024 | 0.182219 | Pass |
| 20260821 | 0.548123 | 0.176790 | 0.182219 | Fail |
| 20260822 | 0.555306 | 0.102477 | 0.182219 | Fail |

All 45 HAR global rounds completed with three clients and zero reported failures;
all federations cleaned up successfully. Independent evidence validation checked
subject/site separation, public bounds, six-class schema, registry defaults,
node-reported patient policies, release runner hashes, frozen tooling hashes,
scoring order, baseline pairing, metrics and sample SD. CTG evidence JSON hashes
remain unchanged. Unique central-plus-federated training time was
952.0 seconds (central fits counted once);
final scoring and orchestration time are excluded. The pod is left running.
