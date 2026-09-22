# Vision evidence: blocked before execution

No utility scores were obtained. On 2026-09-22 the requested A40 pod was
reachable, but `/workspace`, the installed package path and prepared BUS-BRA
audit each exceeded a ten-second filesystem timeout. See
[runtime_preflight.json](runtime_preflight.json) for the observed commands,
timings, GPU identity and mount identity.

`pilot_busbra_pytorch_resnet18_eps{1,4,8}.json` records every planned replicate
as unexecuted, with null metrics and unobserved node privacy configuration.
[summary.json](summary.json) records zero completed replicates and the
artifact hashes. This does not establish a package failure or an epsilon-8
utility boundary.

Dataset: [BUS-BRA v1.0](https://zenodo.org/records/8231412), 1,875 images and
1,064 patients in the previously archived audit. Citation: Gómez-Flores W,
Gregorio-Calas MJ, Pereira WCA (2024), *BUS-BRA: A Breast Ultrasound Dataset
for Assessing Computer-aided Diagnosis Systems*, Medical Physics 51,
3110–3123, [doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812).
Zenodo declares CC BY 4.0; the ZIP includes an additional permissive
attribution licence preserved under `../segmentation/provenance/`.

See [the tooling README](../../../../tools/campaign/vision/README.md) for
the planned protocol, archive SHA-256 and reproduction of the blocker.
