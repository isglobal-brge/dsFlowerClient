# Connect to a data source for federated learning

Single entry point that handles the full init chain: detects data type,
assigns resources, initializes dsImaging and dsFlower handles, and
returns a connection handle with metadata.

## Usage

``` r
ds.flower.connect(conns, data = NULL, resource = NULL, symbol = NULL)
```

## Arguments

- conns:

  DSI connections object.

- data:

  Character; auto-detected data source. Use explicit params if
  ambiguous.

- resource:

  Character; explicit Opal resource name (e.g. "RSRC.brain_mri").

- symbol:

  Character; explicit DS symbol already assigned (e.g. "D").

## Value

A `dsflower_connection` object.

## Details

Uses unique hidden symbols per connection to avoid collisions when
multiple connections are active.
