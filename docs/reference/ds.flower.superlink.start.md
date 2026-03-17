# Start a Flower SuperLink

Spawns a `flower-superlink` process using processx with a private
FLWR_HOME directory. Writes a `config.toml` so that `flwr run` can
connect to this SuperLink.

## Usage

``` r
ds.flower.superlink.start(
  insecure = TRUE,
  fleet_port = 9092L,
  control_port = 9093L,
  serverappio_port = 9091L
)
```

## Arguments

- insecure:

  Logical; use insecure mode (default TRUE).

- fleet_port:

  Integer; port for the Fleet API (default 9092). SuperNodes connect
  here.

- control_port:

  Integer; port for the Control API (default 9093). `flwr run` connects
  here.

- serverappio_port:

  Integer; port for the ServerAppIO API (default 9091).

## Value

Invisible list with process info.
