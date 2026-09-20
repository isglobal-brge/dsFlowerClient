# Segmentation Remedy 2 — protocol v5

Phase: completed

Public BUSI decoder pretraining is NONPRIVATE. BUS-BRA DP fine-tuning is separately accounted.
V3/v4 evidence and the failed per-site boundary are retained unchanged. vetted=FALSE.

```json
{
  "phase": "completed",
  "updated_at": "2026-09-19T23:53:09.900424+00:00",
  "primary_batch": 16,
  "primary_floor": {
    "verdict": "PASS",
    "numerical_floor_pass": true,
    "across_seed_means": {
      "dice": 0.6460808696351549,
      "foreground_dice": 0.6460808696351549,
      "strongest_trivial": 0.29553372019881846,
      "foreground_strongest_trivial": 0.29553372019881846
    },
    "replicates": [
      {
        "seed": 20260919,
        "dice": 0.6617864461182665,
        "foreground_dice": 0.6617864461182665,
        "strongest_trivial": 0.30600943092365357,
        "foreground_strongest_trivial": 0.30600943092365357
      },
      {
        "seed": 20260920,
        "dice": 0.6360230875122249,
        "foreground_dice": 0.6360230875122249,
        "strongest_trivial": 0.30928670744809,
        "foreground_strongest_trivial": 0.30928670744809
      },
      {
        "seed": 20260921,
        "dice": 0.6404330752749733,
        "foreground_dice": 0.6404330752749733,
        "strongest_trivial": 0.27130502222471187,
        "foreground_strongest_trivial": 0.27130502222471187
      }
    ],
    "absolute_floor": 0.5,
    "trivial_margin": 0.1
  },
  "sensitivity_floor": {
    "verdict": "PASS",
    "numerical_floor_pass": true,
    "across_seed_means": {
      "dice": 0.6352267313214685,
      "foreground_dice": 0.6352267313214685,
      "strongest_trivial": 0.29553372019881846,
      "foreground_strongest_trivial": 0.29553372019881846
    },
    "replicates": [
      {
        "seed": 20260919,
        "dice": 0.6473868086626239,
        "foreground_dice": 0.6473868086626239,
        "strongest_trivial": 0.30600943092365357,
        "foreground_strongest_trivial": 0.30600943092365357
      },
      {
        "seed": 20260920,
        "dice": 0.6278844942480368,
        "foreground_dice": 0.6278844942480368,
        "strongest_trivial": 0.30928670744809,
        "foreground_strongest_trivial": 0.30928670744809
      },
      {
        "seed": 20260921,
        "dice": 0.6304088910537449,
        "foreground_dice": 0.6304088910537449,
        "strongest_trivial": 0.27130502222471187,
        "foreground_strongest_trivial": 0.27130502222471187
      }
    ],
    "absolute_floor": 0.5,
    "trivial_margin": 0.1
  },
  "confirmation_exit_codes": {
    "16": 0,
    "64": 0
  },
  "all_66_cells_validated": true,
  "development_failures": 0,
  "summary": "/workspace/segmentation/v5/summary-v5.json"
}
```

Primary confirmation floor: **PASS**.
