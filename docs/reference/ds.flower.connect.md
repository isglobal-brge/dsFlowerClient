# Connect to a data source for federated learning

Single entry point that handles the full init chain. Resource type is an
explicit public choice: imaging resources are admitted through
dsImaging, while tabular resources are resolved with
`as.resource.client()`.

## Usage

``` r
ds.flower.connect(
  conns,
  data = NULL,
  resource = NULL,
  symbol = NULL,
  resource_kind = "imaging"
)
```

## Arguments

- conns:

  DSI connections object.

- data:

  Character; auto-detected data source. Use explicit params if
  ambiguous.

- resource:

  Character; explicit Opal resource name (e.g. "RSRC.brain_mri"). For an
  assigned object, use `symbol`.

- symbol:

  Character; explicit DS symbol already assigned (e.g. "D"), including
  an imaging handle created by
  [`dsImagingClient::ds.imaging.init()`](https://isglobal-brge.github.io/dsImagingClient/reference/ds.imaging.init.html).

- resource_kind:

  Character; exactly `"imaging"` or `"tabular"`. This is never inferred
  by retrying a failed admission.

## Value

A `dsflower_connection` object.

## Details

Uses unique capability-named symbols per connection to avoid collisions
when multiple connections are active. The names remain visible to the
DSI symbol API so exact per-node teardown can distinguish absent from
retained handles.
