"""Hash-bound PUBLIC decoder initialization, outside the trusted DP runner."""
import hashlib
import io
import json
from pathlib import Path


def load_arrays(binding):
    import numpy as np
    data = Path(binding['checkpoint']).read_bytes()
    if hashlib.sha256(data).hexdigest() != binding['checkpoint_sha256']:
        raise ValueError('public checkpoint digest differs')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != {str(i) for i in range(6)}:
            raise ValueError('public narrow decoder must contain six tensors')
        arrays = [archive[str(i)].copy() for i in range(6)]
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
    if hashes != binding['tensor_sha256']:
        raise ValueError('public checkpoint tensor digest differs')
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('public checkpoint requires finite float32 tensors')
    return arrays


def apply(model, cfg, binding, seed):
    import base64
    from dsflower_runner import params, segmentation
    if (binding['seed'] != seed or binding['decoder'] != 'narrow'
            or binding['dataset'] != 'BUSI' or binding['privacy'] != 'public_nonprivate'
            or binding['encoder_sha256'] != segmentation.CHECKPOINT_SHA256
            or json.loads(base64.b64decode(cfg['model-spec-b64'])) != segmentation.decoder_spec('narrow')):
        raise ValueError('public pretraining binding differs from requested model/seed')
    arrays = load_arrays(binding)
    if [a.shape for a in arrays] != [tuple(p.shape) for p in model.parameters()]:
        raise ValueError('public checkpoint tensor shapes differ')
    params.set_torch_params(model, arrays)
    return arrays


def verify_capture(work, expected_binding):
    import numpy as np
    binding = json.loads((work / 'public-pretraining-binding.json').read_text())
    initial = json.loads((work / 'public-capture/public-initial.json').read_text())
    if binding != expected_binding or initial.get('public_pretraining') != binding:
        raise ValueError('captured public pretraining provenance differs')
    expected = load_arrays(binding)
    with np.load(work / 'public-capture/public-initial-arrays.npz', allow_pickle=False) as archive:
        if set(archive.files) != {str(i) for i in range(6)}:
            raise ValueError('captured initialization tensor count differs')
        if any(not np.array_equal(archive[str(i)], a) for i, a in enumerate(expected)):
            raise ValueError('federation did not start from public checkpoint')
    if initial['tensor_sha256'] != binding['tensor_sha256'] or initial['seed'] != binding['seed']:
        raise ValueError('captured initialization identity differs')
