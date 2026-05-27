# Start a Flower run

Fetches the model template from the server, builds a Flower App from the
recipe, then invokes `flwr run` against the running SuperLink. Model
weights and training history are automatically saved to `output_dir`
after training completes.

## Usage

``` r
ds.flower.run.start(
  recipe,
  conns = NULL,
  app_dir = NULL,
  run_config = list(),
  output_dir = NULL,
  results_dir = NULL,
  symbol = "flower",
  verbose = TRUE
)
```

## Arguments

- recipe:

  A `dsflower_recipe` object.

- conns:

  DSI connections object. Used to fetch the model template from the
  server. If NULL, uses the connections stored during
  `ds.flower.nodes.init`.

- app_dir:

  Character; path to a pre-built app directory (optional).

- run_config:

  Named list; additional run config overrides.

- output_dir:

  Character; persistent directory for model output. Defaults to
  `"dsflower_output/<timestamp>"` in the working directory.

- results_dir:

  Character; temporary directory where the Flower ServerApp writes model
  artefacts. Usually generated automatically.

- symbol:

  Character; server-side Flower handle symbol. Low-level callers that
  initialise the default handle can leave this as `"flower"`.

- verbose:

  Logical; print flwr output (default TRUE).

## Value

A `dsflower_run` object with weights, history, and predictions.
