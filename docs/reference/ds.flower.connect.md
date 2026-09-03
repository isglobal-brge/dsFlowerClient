# Connect to a data source for federated learning

Single entry point that handles the full init chain: detects data type,
admits imaging resources through dsImaging, initializes dsFlower
handles, and returns a connection handle with metadata.

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

  Character; explicit Opal dsImaging resource name (e.g.
  "RSRC.brain_mri"). It is admitted with `imagingInitDS` before dsFlower
  initialization. For an assigned tabular object, use `symbol`.

- symbol:

  Character; explicit DS symbol already assigned (e.g. "D"), including
  an imaging handle created by
  [`dsImagingClient::ds.imaging.init()`](https://isglobal-brge.github.io/dsImagingClient/reference/ds.imaging.init.html).

## Value

A `dsflower_connection` object.

## Details

Uses unique capability-named symbols per connection to avoid collisions
when multiple connections are active. The names remain visible to the
DSI symbol API so exact per-node teardown can distinguish absent from
retained handles.
