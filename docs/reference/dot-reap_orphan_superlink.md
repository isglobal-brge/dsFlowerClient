# Reap an orphaned SuperLink left by a crashed or abandoned session

Collects candidate PIDs two fork-free ways – the PID file (fast, exact)
and a ps scan for any `flower-super*` still holding one of our ports
(catches pidfile-less orphans and the superexec child) – then signals
only those it can re-confirm are Flower processes. Neither path shells
out, so neither can trip the macOS fork-after-threads segfault the old
lsof/ps scan could; the identity re-check means a recycled PID is never
killed.

## Usage

``` r
.reap_orphan_superlink(fleet_port, ports = fleet_port)
```

## Arguments

- fleet_port:

  Integer; the SuperLink's fleet port (PID-file key).

- ports:

  Integer vector; all SuperLink ports to free (default: just
  `fleet_port`).

## Value

Invisible TRUE if anything was reaped, FALSE otherwise.
