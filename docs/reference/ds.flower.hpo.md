# Optimize an explicit objective locally with Optuna

Runs one sequential Optuna TPE study entirely on the researcher's
machine. Optuna uses ephemeral in-memory storage. For each suggestion
this function calls \`objective(params)\` in the current R process; the
function itself is never serialized, uploaded, or sent to a data node.
It must return one finite numeric value.

## Usage

``` r
ds.flower.hpo(
  objective,
  space,
  n_trials = 20L,
  direction = c("minimize", "maximize"),
  seed = 0L
)
```

## Arguments

- objective:

  Local R function accepting one named parameter list and returning one
  finite numeric scalar.

- space:

  Non-empty named list of dimensions created with
  \`ds.flower.hpo.float()\`, \`ds.flower.hpo.integer()\`, or
  \`ds.flower.hpo.categorical()\`. Parameter names are unrestricted by a
  model catalog.

- n_trials:

  Positive integer number of trials in this local study, at most one
  million. This per-process memory ceiling is not a historical quota or
  a limit on the number of HPO calls.

- direction:

  Either \`"minimize"\` or \`"maximize"\`.

- seed:

  Integer in \`\[0, 2^32 - 1\]\` controlling only local Optuna
  suggestions. It is unrelated to node-owned privacy randomness.

## Value

A \`dsflower_hpo\` object with the best value and parameters, complete
local trial history, seed, direction, and Optuna version.

## Details

The objective may explicitly call \`ds.flower.fit()\` and a suitable
private validation API. Each such call remains an ordinary independent
server-enforced training or validation release. HPO does not weaken or
alter node privacy policy. \`n_trials\` is only the geometry of this
local call: it is not a rate limit, quota, catalog permission, or
persistent budget.

Repeating a study with Optuna 4.8.0, the same seed, search space, and
sequence of objective values reproduces its suggestions. Reproducibility
of the returned values additionally depends on the objective itself.
Static typed spaces are intentional: conditional search logic would
require exposing Python Trial semantics to R or serializing code,
neither of which is part of this data-only local bridge.
