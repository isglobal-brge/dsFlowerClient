"""Bound simultaneous public federations on the CPU-quota-limited campaign pod."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import subprocess
import sys
import time


@contextmanager
def slot(root):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    handles = [open(root / str(i), "a") for i in range(2)]
    try:
        while True:
            for index, handle in enumerate(handles):
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                yield index
                return
            time.sleep(.25)
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    with slot(Path("/tmp/dsflower-segmentation-public-slots")) as index:
        print("Public federation CPU slot: %d" % index, flush=True)
        raise SystemExit(subprocess.call(sys.argv[1:]))
