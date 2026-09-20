"""Copy the selected seed-specific public binding before any federation starts."""
import json
import os
from pathlib import Path
import sys
from benchmark_hooks.public_initialization import load_arrays


def bind(work, seed):
    source = Path(os.environ['F_SEG_V5_BINDINGS']) / f'seed{seed}.json'
    binding = json.loads(source.read_text())
    if binding['seed'] != seed:
        raise ValueError('public pretraining seed mismatch')
    load_arrays(binding)
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'public-pretraining-binding.json').open('x') as stream:
        json.dump(binding, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    bind(Path(sys.argv[1]), int(sys.argv[2]))
