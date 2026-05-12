# Attach to a detached SuperLink

Reconnects to a SuperLink started with `detached = TRUE` in a previous R
session. Reads the state file, verifies the process is alive and the
port is listening, and restores the session state.

## Usage

``` r
ds.flower.superlink.attach()
```

## Value

Invisible list with SuperLink info.
