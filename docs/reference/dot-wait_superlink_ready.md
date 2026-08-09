# Wait for SuperLink to be ready

Verifies the process is alive and the fleet port is accepting
connections (via a native socket probe – no subprocess, no TLS handshake
needed).

## Usage

``` r
.wait_superlink_ready(proc, port, log_path, timeout = 15)
```

## Arguments

- proc:

  processx process object.

- port:

  Integer; port to check.

- log_path:

  Character; path to the log file (for error messages).

- timeout:

  Numeric; seconds to wait.
