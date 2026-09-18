#!/bin/sh
# Gate outside R: do not invoke the interpreter before provisioning completes.
set -eu
if [ -d /workspace ]; then
    if ! grep -q 'R_STACK_DONE' /workspace/logs/install_r.log; then
        echo 'Blocked: /workspace/logs/install_r.log has no R_STACK_DONE.' >&2
        exit 2
    fi
    export R_LIBS_USER=/workspace/segmentation/rlib
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec Rscript "$script_dir/run_federated.R" "$@" "$script_dir"
