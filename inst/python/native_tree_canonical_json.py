"""Reproduce native-tree container JSON without rounding binary64 leaves."""

import json
import sys


MAX_BYTES = 64 * 1024 * 1024


def _without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def canonical_json(raw):
    if not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("container exceeds its byte bound")
    value = json.loads(raw.decode("ascii"), object_pairs_hook=_without_duplicates)
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("ascii")


if __name__ == "__main__":
    try:
        encoded = canonical_json(sys.stdin.buffer.read(MAX_BYTES + 1))
    except (ValueError, RecursionError, OverflowError):
        sys.stderr.write("native-tree canonical JSON rejected")
        raise SystemExit(2)
    sys.stdout.buffer.write(encoded)
