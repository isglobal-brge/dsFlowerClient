#!/bin/sh
# Gate outside R: do not invoke the interpreter before provisioning completes.
set -eu
if [ -d /workspace ]; then
    if ! grep -q '^R_STACK_DONE$' /workspace/logs/install_r.log &&
       ! grep -q '^R_STACK_DONE$' /workspace/segmentation/r-stack-ready.log; then
        echo 'Blocked: neither shared nor segmentation-local R stack is verified.' >&2
        exit 2
    fi
    export R_LIBS_USER=/workspace/segmentation/rlib
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${F_SEG_V5_BINDINGS:-}" ]; then
    python "$script_dir/bind_public_initialization.py" "$5" "$4"
fi
if [ -d /workspace/segmentation ]; then
    exec python "$script_dir/public_slots.py" Rscript "$script_dir/run_federated.R" "$@" "$script_dir"
fi
exec Rscript "$script_dir/run_federated.R" "$@" "$script_dir"
