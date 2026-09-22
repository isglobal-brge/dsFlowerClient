# Vision diagnosis R4

Token: `FLOWER_CELLS_VISION_R4_2026-09-22`.

The converged C=1 logistic head achieves **0.779405 ± 0.025903 patient-level inner-CV AUC** across five folds. Its out-of-fold accuracy is 0.751174, Brier 0.172757, and log-loss 0.537158. This uses only the 852 training patients (1,501 images) in the archived 20260919 split.

The original three seeds had different outer splits. R4 fixes the 20260919 outer split and varies only initialization/training seeds, preventing schedule-selection leakage into another seed's held-out cohort. No original split JSON, held-out metadata, held-out images or previous predictions were read during diagnosis. Only the prepared training-site collections and the old public initial model configuration were opened.

Exact released image extraction, image size 224, first-appearance patient ordering, feature averaging and deterministic modal labels were retained. Images are min/max scaled by the runner, replicated to three channels, and bilinearly resized; no extra ImageNet channel normalization or fitted feature scaling was added. The head is the same 512-to-2 affine model class. The converged central uses fixed C=1 L2 logistic regression with an unpenalized intercept, L-BFGS-B, and an equivalent symmetric two-logit parameterization.

The five folds reached AUCs 0.779154, 0.738401, 0.800447, 0.802842, 0.776181; each optimizer reported convergence. All training and inner split identifier hashes are retained in `vision-client/tools/campaign/vision/r4/diagnosis/audit.json`. Feature arrays and raw patient identifiers remain on the pod.

The schedule proxy uses a stratified 681/171 patient split, five rounds, and the same update count and Poisson rate as a 227-patient inner site. It fits centrally to prune candidates; it is not the federated confirmation. Optimizer state resets every round.

| Optimizer | LR | Local epochs | Batch | Steps | Inner AUC | Accuracy | Brier | Log-loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sgd | 0.001 | 1 | 32 | 40 | 0.518182 | 0.678363 | 0.221128 | 0.634108 |
| sgd | 0.01 | 5 | 32 | 200 | 0.700627 | 0.684211 | 0.198493 | 0.583996 |
| sgd | 0.01 | 5 | 128 | 50 | 0.652038 | 0.678363 | 0.210127 | 0.610593 |
| sgd | 0.01 | 20 | 32 | 800 | 0.731975 | 0.701754 | 0.186742 | 0.553768 |
| sgd | 0.01 | 20 | 128 | 200 | 0.691223 | 0.707602 | 0.199493 | 0.587714 |
| sgd | 0.03 | 5 | 32 | 200 | 0.725235 | 0.684211 | 0.249655 | 0.795500 |
| sgd | 0.03 | 5 | 128 | 50 | 0.650313 | 0.485380 | 0.261640 | 0.718543 |
| sgd | 0.03 | 20 | 32 | 800 | 0.745925 | 0.614035 | 0.239928 | 0.689982 |
| sgd | 0.03 | 20 | 128 | 200 | 0.692947 | 0.397661 | 0.377777 | 1.031088 |
| sgd | 0.1 | 5 | 32 | 200 | 0.710188 | 0.707602 | 0.232975 | 0.965488 |
| sgd | 0.1 | 5 | 128 | 50 | 0.623981 | 0.333333 | 0.613148 | 3.399798 |
| sgd | 0.1 | 20 | 32 | 800 | 0.756113 | 0.719298 | 0.230675 | 1.055384 |
| sgd | 0.1 | 20 | 128 | 200 | 0.677194 | 0.345029 | 0.622989 | 4.758069 |
| adam | 0.001 | 5 | 32 | 200 | 0.746082 | 0.713450 | 0.183592 | 0.546971 |
| adam | 0.001 | 5 | 128 | 50 | 0.695298 | 0.707602 | 0.201669 | 0.592637 |
| adam | 0.001 | 20 | 32 | 800 | 0.768182 | 0.736842 | 0.177674 | 0.526846 |
| adam | 0.001 | 20 | 128 | 200 | 0.744828 | 0.707602 | 0.185170 | 0.549994 |
| adam | 0.003 | 5 | 32 | 200 | 0.759404 | 0.695906 | 0.197110 | 0.578135 |
| adam | 0.003 | 5 | 128 | 50 | 0.716928 | 0.672515 | 0.195779 | 0.576877 |
| adam | 0.003 | 20 | 32 | 800 | 0.776332 | 0.754386 | 0.176842 | 0.530117 |
| adam | 0.003 | 20 | 128 | 200 | 0.757210 | 0.730994 | 0.183592 | 0.544782 |
| adam | 0.01 | 5 | 32 | 200 | 0.768339 | 0.684211 | 0.194042 | 0.572580 |
| adam | 0.01 | 5 | 128 | 50 | 0.721787 | 0.684211 | 0.209401 | 0.615842 |
| adam | 0.01 | 20 | 32 | 800 | 0.772727 | 0.742690 | 0.180829 | 0.560787 |
| adam | 0.01 | 20 | 128 | 200 | 0.761755 | 0.725146 | 0.185258 | 0.549162 |

## Actual private confirmation and selection

| Adam LR (20 epochs, batch32) | DP epsilon8 inner AUC | Accuracy | Brier | Log-loss |
|---:|---:|---:|---:|---:|
| .003 (selected) | .500940439 | .678362573 | .290158148 | 1.502839654 |
| .01 | .478683386 | .660818713 | .313499904 | 2.215140676 |

The declared rule selects **Adam .003, batch32, 20 local epochs per round, five rounds**, with no weight decay, L1 penalty or scheduler. The representation carries signal, but both actual private confirmations are weak. The 25-candidate nonprivate sweep and two private confirmations do not prove that every allowed schedule is untrainable. LR is allowed up to 10 and local epochs up to 1000; no package parameter cap blocked this search.

Each inner site held 227 fitting patients; 171 training patients were reserved for validation. Actual node feature/target hashes matched fresh extraction exactly; 15 captures per candidate each showed 160 updates per round, 800 total/site, Poisson q=1/8, expected divisor 28 and noise multiplier 4.98046875. Delta 1e-6, patient privacy and clipping norm 1 stayed unchanged.

The first .003 launch failed before head initialization/training on the SuperLink 15-second readiness timeout. Its complete failed record and executed source are retained. A driver-only extension calls the existing readiness helper with an additional 90 seconds for that exact error. The untrained retry and the .01 candidate completed with clean teardown; no scored model was repeated. No package code changed.

The corrected cell is declared before execution in `vision-client/tools/campaign/vision/README.md` and `r4/protocol.json`. All final seeds reuse the fixed 852/212 split and original 284-patient sites. Final training re-extracts features fresh; the diagnostic feature cache is not reused. Central is the converged C=1 logistic head; other comparators are the selected finite-schedule nonprivate FedAvg twin (both clipping and noise removed), pooled-DP, and training-majority predictor. Primary metrics remain per image; patient metrics are secondary annotations. No post-score alternative is permitted.

BUS-BRA: Gómez-Flores, Gregorio-Calas and Pereira (2024), Medical Physics 51:3110–3123, doi:10.1002/mp.16812; dataset doi:10.5281/zenodo.8231412. Thesis key `gomezflores_busbra_2024`.
