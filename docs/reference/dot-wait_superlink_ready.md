# Wait for SuperLink to be ready

Verifies the process is alive and the fleet port is listening. Uses
`lsof` (macOS/Linux) to check port binding without needing a TLS
handshake.

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
