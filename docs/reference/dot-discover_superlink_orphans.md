# Discover orphaned Flower SuperLink/SuperExec PIDs squatting given ports

Fork-free process scan via the ps package (libproc on macOS, /proc on
Linux) – the client analogue of the server's /proc SuperNode scan.
Matches only `flower-super*` processes whose command line references one
of OUR ports, so it reaps a stale SuperLink (or its superexec child)
that has no PID file – e.g. one started before PID-file tracking, or
whose file was lost – and never touches an unrelated service.

## Usage

``` r
.discover_superlink_orphans(ports)
```

## Arguments

- ports:

  Integer vector; the SuperLink ports we are about to bind.

## Value

Integer vector of matching PIDs.
