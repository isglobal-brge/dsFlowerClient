# Ensure the client Python venv exists and is healthy

Downloads uv if needed, creates venv with Python 3.11, installs flwr.
Idempotent: skips if venv already healthy.

## Usage

``` r
.ensure_client_venv(timeout_secs = 600)
```

## Arguments

- timeout_secs:

  Numeric; max seconds for install (default 600).

## Value

Invisible TRUE.
