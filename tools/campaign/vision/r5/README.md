# Vision DP-aware schedule diagnosis

This is a training-only selection exercise for the released `pytorch_resnet18` contract. It does not implement or run a final cell. Only the fixed R4 852-patient training cohort and its unchanged 681/171 inner split are read. The 212 test patients are excluded.

The runner is dsFlower 0.5.1 / dsFlowerClient 0.5.0, canonical SHA256 `2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`. Package code is unchanged. Scripts operate on `/workspace/cells-vision` on the vision pod only.

Execution order after sourcing `../environment.sh`, using the pod's `venvs/pytorch-gpu/bin/python`:

1. `pod_state.py --clean --output /workspace/cells-vision/r5/pod-initial.json` and the existing `../verify_runtime.py` verify the pod and installed releases.
2. `prepare_inner.py --root /workspace/cells-vision` reproduces the actual R4 inner-site feature tensors and verifies captured hashes. Validation tensors remain the original cached R4 tensors.
3. `saturation_check.py` and `validate.py`, each with `--root /workspace/cells-vision`, validate the analytical emulator against unchanged `_dp_fit`, with shared fresh secure streams, including full five-round federations at R3/R4 schedules. Historical comparisons use independent streams, never recovered node keys.
4. `emulate.py --root /workspace/cells-vision` declares and ranks all 736 schedules using three fixed noise/sampling seeds. It requires passing numerical validation. SGD learning rates are 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10; Adam rates are 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1. SGD momentum is 0 or 0.9; local epochs are 1, 2, 4, 8; batches are 32, 64, 128, full site; weight decay is 0 or 0.0001. Five rounds, epsilon 8, delta 1e-6, clipping norm 1 remain fixed.
5. `validate_shortlist.py` and `validate_aggregation.py`, each with `--root /workspace/cells-vision`, check the shortlisted schedules across all three search seeds and five rounds. They compare prediction ranks as well as parameter/probability errors, and test installed Flower aggregation and final-round reply order.
6. `confirm.py --root /workspace/cells-vision` runs the top three through actual image admission, frozen-backbone extraction and private federation, one each. It selects maximum real inner-validation AUC. If that AUC is below 0.60, the same schedule also runs at epsilon 4 and 1. It never launches a final cell.
7. Record final pod state and runner hashes, then use `report.py --results /workspace/cells-vision/r5` and `package_evidence.py --root /workspace/cells-vision` to produce the public records.

The final runtime check retained a 45-second Torch import timeout. `verify_finish.py --root /workspace/cells-vision` repeats only that import probe with a 120-second allowance, after asserting unchanged package versions and runner hashes; no training is repeated.

All commands run in the foreground. Confirmation drivers retain prior attempts and refuse overwriting existing run directories. Feature arrays and patient identifiers stay on the pod. No saved model is selected using test data.

For full site, batch 227 during selection denotes a semantic full-site batch: batch 284 at each outer training site and batch 852 for its pooled twin. Fixed batches retain their numeric values. This rule is declared before ranking. Accountants must be recalibrated on the implementation cohort.

`PROCEED: yes` requires a selected real inner-validation AUC at least 0.60. A negative result describes this tested grid, not a proof over every continuous optimizer setting. The three-seed sweep and three real confirmations reuse one validation cohort; the winning validation score is subject to selection optimism and is not a test estimate.
