# Initialize Flower handles on all servers

Creates a Flower handle on each server from a symbol already assigned in
the DataSHIELD session (data.frame, matrix, or any object loaded via
`datashield.assign.table` or DataSHIELD operations). Imaging resources
are first admitted by `imagingInitDS` and only its opaque handle is
passed to dsFlower.

## Usage

``` r
ds.flower.nodes.init(
  conns,
  data = NULL,
  resource = NULL,
  symbol = "flower",
  resource_kind = "imaging"
)
```

## Arguments

- conns:

  DSI connections object.

- data:

  Character or named list; symbol name(s) of data already assigned in
  the DataSHIELD session. Mutually exclusive with `resource`.

- resource:

  Character or NULL; name of an Opal resource to assign before init.

- symbol:

  Character; symbol name for the Flower handle (default `"flower"`).

- resource_kind:

  Character; exactly `"imaging"` or `"tabular"`. This public routing
  choice is not inferred from a failed server call.

## Value

A `dsflower_result` with per-site init results.

## Details

Accepts a single string (same symbol on all servers) or a named list
(one entry per server):


    # Same symbol on all servers
    ds.flower.nodes.init(conns, data = "D")

    # Different symbol per server
    ds.flower.nodes.init(conns, data = list(
      hospital_a = "D_filtered",
      hospital_b = "D_merged",
      hospital_c = "D"
    ))

    # From an Opal resource (e.g. imaging+dataset://)
    ds.flower.nodes.init(conns, resource = "chest_xray")
