# Check if a local PID is alive (cross-platform, never kills the process)

Uses the ps package (`ps_handle` + `ps_is_running`), which also detects
PID reuse via the process create-time. We deliberately avoid
`tools::pskill(pid, 0L)`: on Windows `pskill` always calls
`TerminateProcess`, so the Unix "signal 0" liveness trick would KILL the
very process it is meant to probe.

## Usage

``` r
.pid_is_alive_local(pid)
```
