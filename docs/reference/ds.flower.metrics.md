# Inspect and select private validation metrics

These helpers operate only on an already released pooled metric object.
They do not contact data nodes, spend privacy budget, or create another
release. The catalog contains the scalar metrics suitable for model
selection for the result's task. Diagnostic mass (\`n\`) and structured
outputs such as curves, calibration bins, label details, and confusion
matrices remain available in \`result\$metrics\`, but cannot be selected
as an HPO score.

## Usage

``` r
ds.flower.metrics(result)

ds.flower.score(result, metric)

ds.flower.metric_direction(metric, task)
```

## Arguments

- result:

  A \`dsflower_validation\`, \`dsflower_cv\`, or \`dsflower_run\`
  containing an atomic holdout release.

- metric:

  One scalar metric name from \`ds.flower.metrics(result)\`.

- task:

  Validation task: \`"binary"\`, \`"multiclass"\`, \`"ordinal"\`,
  \`"multilabel"\`, \`"regression"\`, or \`"count"\`.

## Value

\`ds.flower.metrics()\` returns a data frame with \`task\`, \`metric\`,
\`direction\`, \`available\`, and \`value\`. \`ds.flower.score()\`
returns one finite numeric scalar. \`ds.flower.metric_direction()\`
returns \`"minimize"\` or \`"maximize"\`.

## Details

A metric can be unavailable when the released sufficient statistics do
not support a finite denominator. \`ds.flower.score()\` fails explicitly
in that case instead of inventing or coercing a value.
