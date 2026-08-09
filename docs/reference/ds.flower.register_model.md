# Register a dsFlower model generator

Intended for dsFlowerClient extension packages: call this from your
package's
[`.onLoad()`](https://isglobal-brge.github.io/dsFlowerClient/reference/dot-onLoad.md)
to add models to the registry. The model becomes usable via
`ds.flower.model("<name>")` / `ds.flower.fit(..., model = "<name>")`.

## Usage

``` r
ds.flower.register_model(
  name,
  track,
  generate,
  loss = NULL,
  defaults = list(),
  description = NULL,
  vetted = FALSE,
  overwrite = FALSE,
  parameter_types = NULL,
  parameter_aliases = character(),
  required_parameters = character(),
  parameter_choices = list(),
  data_kinds = "tabular"
)
```

## Arguments

- name:

  Character; the model name (e.g. "pytorch_logreg").

- track:

  Character; the enforced-DP track. The public model registry currently
  accepts only "neural" (nn.Module + DP-SGD). Native tree engines are
  registered by their trusted server adapters, not extension generators.

- generate:

  Function of one argument `params` (a named list). It MUST return a
  named model SPEC (DATA, never code): model SPEC
  `list(kind = "sequential", layers = list(...))` the node builds with
  stock torch layers (end with a linear onto `"@out"`). The node owns
  the loop, loss, and optimizer.

- loss:

  Character; required for the neural track. The per-sample neural loss
  must come from the node allowlist: stock losses `bce_logits`,
  `cross_entropy`, `mse`, `poisson_nll`, `multilabel_bce`, `hinge`
  (linear SVM), `ordinal` (CORN); plus vetted custom per-sample losses
  `negbin_nll` (overdispersed counts), `gamma_nll` (positive
  continuous), `huber` (bounded robust regression), and `quantile`
  (bounded conditional-quantile regression). The node pins the actual
  loss; this is the client's request.

- defaults:

  Named list of default params merged under user-supplied params.

- description:

  Character or NULL; a one-line human description.

- vetted:

  Logical; informational only. Every model is node-built from the
  allowlisted spec vocabulary (no researcher code runs), so first- and
  third-party generators take the same validated path; this flag just
  marks first-party collections in
  [`ds.flower.list_models()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.list_models.md).

- overwrite:

  Logical; allow replacing an existing registration.

- parameter_types:

  Required named character vector defining every accepted parameter and
  its basic type. Use
  [`character()`](https://rdrr.io/r/base/character.html) for a model
  with no parameters. This makes misspelled or unsupported parameters
  fail early.

- parameter_aliases:

  Optional named character vector mapping accepted aliases to canonical
  parameter names (for example `c(eta = "learning_rate")`).

- required_parameters:

  Character vector of canonical parameters which must be present after
  defaults and user values are merged.

- parameter_choices:

  Optional named list of finite allowlists for declared parameters.
  Values outside the corresponding allowlist fail before any DSI
  operation.

- data_kinds:

  Character vector of supported inputs: `"tabular"`, `"image"`, or both.

## Value

Invisibly, the model name.
