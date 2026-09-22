# BUS-BRA vision campaign

The first executed cell is schedule-limited at registry defaults; its finite-schedule central twin AUC is **0.596183 ± 0.041527**. Original cell JSONs and verification files are preserved byte-for-byte; complete historical documentation is in [README-r1.md](README-r1.md) and [summary-r1.json](summary-r1.json).

R4 has been selected on training patients only and declared before corrected training. See the [track declaration](../../../../tools/campaign/vision/README.md) and [frozen protocol](../../../../tools/campaign/vision/r4/protocol.json). No corrected test score exists yet.

Selected Adam .003, batch32, 20 local epochs, five rounds. Converged logistic inner-CV AUC: **0.779405 ± 0.025903**. Actual private inner AUC at epsilon8: **0.500940**, versus **0.478683** for the second candidate. Both private results are weak and retained honestly.

BUS-BRA citation key: `gomezflores_busbra_2024`; [paper DOI](https://doi.org/10.1002/mp.16812), [dataset DOI](https://doi.org/10.5281/zenodo.8231412).
