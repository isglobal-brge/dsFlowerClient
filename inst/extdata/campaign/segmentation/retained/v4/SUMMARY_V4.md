# Segmentation v4 public-development study

V3 floors remain FAILED. Registration remains vetted=FALSE.
Public benchmark confirmation uses previously inspected outer test sets.

Selected configuration: {"decoder": "narrow", "batch_size": 16, "rounds": 20, "parameters": 9521}
Inner-validation floor: FAILED

See summary-v4.json and evidence/batch{16,64} for every cell, stratum, twin and envelope.
Selected batch is primary; the other batch is sensitivity. No batch pooling.
Development failures: 0
Confirmation process exit codes: {"16": 0, "64": 0}

| Version | Batch | ε8 BUS-BRA Dice | Foreground Dice | Floor |
|---|---:|---:|---:|---|
| v3 | 16 | 0.0 | 0.0 | FAILED |
| v3 | 64 | 0.0 | 0.0 | FAILED |
| v4 | 16 | 0.21153949338428624 | 0.21153949338428624 | FAILED |
| v4 | 64 | 0.0 | 0.0 | FAILED |
