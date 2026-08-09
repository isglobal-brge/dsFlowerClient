# Start a Flower run

Invokes `flwr run` for a pre-built Flower App against the running
SuperLink. Model weights and training history are automatically saved to
`output_dir` after training completes.

## Usage

``` r
ds.flower.run.start(
  recipe,
  conns = NULL,
  app_dir = NULL,
  run_config = list(),
  output_dir = NULL,
  output_name = NULL,
  results_dir = NULL,
  symbol = "flower",
  verbose = FALSE,
  silent = FALSE
)
```

## Arguments

- recipe:

  A `dsflower_recipe` object.

- conns:

  DSI connections object used to determine the required site count. If
  NULL, uses the connections stored during `ds.flower.nodes.init`.

- app_dir:

  Required character path to a pre-built app directory. The high-level
  [`ds.flower.submit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.submit.md)
  and
  [`ds.flower.fit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.fit.md)
  pipelines build and supply it automatically.

- run_config:

  Named list; additional run config overrides.

- output_dir:

  Character; persistent directory for model output. Defaults to
  `"dsflower_output/<timestamp>"` in the working directory.

- output_name:

  Optional name for the persisted model artifact.

- results_dir:

  Character; temporary directory where the Flower ServerApp writes model
  artefacts. Usually generated automatically.

- symbol:

  Character; server-side Flower handle symbol. Low-level callers that
  initialise the default handle can leave this as `"flower"`.

- verbose:

  Logical; print flwr output (default FALSE).

- silent:

  Logical; suppress progress feedback.

## Value

A `dsflower_run` object with run status and identifiers, model metadata,
weights, history, output paths, and captured CLI output.
