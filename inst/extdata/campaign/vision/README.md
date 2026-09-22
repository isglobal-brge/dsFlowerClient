# BUS-BRA vision campaign

R5 is declared before training: SGD lr 3, momentum 0.9, two local epochs, full-site batches (284/site; 852 pooled), five rounds. Selected under the real DP contract on training patients only; real inner-validation AUC 0.705486.

See the [complete declaration and comparator definitions](../../../../tools/campaign/vision/README.md) and [frozen protocol](../../../../tools/campaign/vision/r5_impl/protocol.json). Fixed 852/212 patient split, three sites of 284; three seeds; epsilon order 8, 4, 1; delta 1e-6; patient unit; clipping norm 1. All fits precede one held-out scoring pass.

R3: schedule-limited at registry defaults. R4: selection lesson—non-private pruning does not transfer under DP; final scoring stopped before test access. Original records remain unchanged.

BUS-BRA provenance: thesis key `gomezflores_busbra_2024`, paper DOI `10.1002/mp.16812`, dataset DOI `10.5281/zenodo.8231412`.
