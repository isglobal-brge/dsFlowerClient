# V5 public-pretrained decoder zero-shot baseline

Fresh CPU-only zero-shot inference of the three BUSI epochs60 public-pretraining checkpoints. Each 9,521-parameter narrow decoder was reconstructed with strict state-dict loading and evaluated without any private fine-tuning. The frozen ResNet-18 encoder is represented by the existing frozen public feature tensors, which are the same inference inputs used by `audit.py`.

Foreground Dice is the audit's subject-macro `foreground_positive` Dice at a 0.5 threshold. The strongest trivial value is the audit's fixed-mask `all`-subject Dice; it is included as the established floor reference, not recomputed from predictions.

| Cohort | Seed | Zero-shot foreground Dice | Strongest trivial mask (all-subject Dice) | Released v5 fedDP ε8 Dice (reference) |
|---|---:|---:|---|---:|
| BUS-BRA | 20260919 | 0.413434514 | square_64: 0.306009431 | 0.646 |
| BUS-BRA | 20260920 | 0.362465328 | square_64: 0.309286707 | 0.646 |
| BUS-BRA | 20260921 | 0.491835523 | square_64: 0.271305022 | 0.646 |
| BUS-BRA | mean | 0.422578455 | across-seed mean: 0.295533720 | 0.646 |
| BrEaST | 20260919 | 0.469735707 | disk_r32: 0.270204988 | 0.526 |
| BrEaST | 20260920 | 0.464112629 | disk_r32: 0.235564137 | 0.526 |
| BrEaST | 20260921 | 0.511695513 | disk_r32: 0.320421577 | 0.526 |
| BrEaST | mean | 0.481847950 | across-seed mean: 0.275396900 | 0.526 |

## Verification

- Each checkpoint NPZ SHA-256 and each of its six tensor SHA-256 values matched the epochs60 public-pretraining manifest.
- Each model was rebuilt from the recorded segmentation configuration, had exactly 9,521 parameters, and loaded its checkpoint with `strict=True`.
- Each cohort used its matching full ε8 effective split; the sorted outer-test subject IDs resolved to 212 BUS-BRA or 51 BrEaST frozen feature/mask rows.
- All scored inputs were SHA-256 checked before and after inference; `protected_inputs_unchanged` is `true`.

The released fedDP ε8 figures are comparison references supplied for v5; they are not zero-shot results. No training, optimization step, or evidence-file mutation occurred.
