# Start a Flower SuperLink

Spawns a `flower-superlink` process. In detached mode, the process
survives R session exit.

## Usage

``` r
ds.flower.superlink.start(
  fleet_port = 9092L,
  control_port = 9093L,
  serverappio_port = 9091L,
  detached = FALSE,
  insecure = getOption("dsflower.superlink_insecure", FALSE)
)
```

## Arguments

- fleet_port:

  Integer; port for the Fleet API (default 9092).

- control_port:

  Integer; port for the Control API (default 9093).

- serverappio_port:

  Integer; port for the ServerAppIO API (default 9091).

- detached:

  Logical; if TRUE, SuperLink runs as daemon (survives R session exit).
  Default FALSE for interactive use.

- insecure:

  Logical; explicitly allow plaintext local SuperLink transport.

## Value

Invisible list with process info.
