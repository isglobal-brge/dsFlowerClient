# Cardiotocography multiclass utility evidence

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
`summary.json` aggregates all three cells; `environment.json` records the pod,
provisioning time and dependency versions. The cell JSONs contain the complete
per-replicate metrics, splits, public TRAIN-derived feature bounds, node-reported
privacy settings, model hashes, round histories and wall-clock times.

## Measured result

The epsilon-8 diagnostic **failed**: accuracy equaled the majority rate in
all three replicates. Macro-AUC was above 0.5. No scored cell was rerun, no
settings were changed, and no alternative was selected.

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
