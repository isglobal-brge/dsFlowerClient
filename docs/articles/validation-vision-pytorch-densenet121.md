# Vision Validation: pytorch_densenet121

This vignette records the catalogue coverage check for a dsFlower vision
template. Synthetic PNG images were split across three Opal/Rock nodes,
trained federatively through DataSHIELD, and compared with a centralized
PyTorch baseline trained on the pooled synthetic cohort. The purpose is
method-path validation: image staging, template execution, result
persistence, and cleanup. The current biomedical direct-image benchmark
is the PathMNIST vignette, and the dsImaging handoff evidence is
reported in the LUNG1 vignettes.

It is laid out in two parts: first the commands you would run to
reproduce the validation, then the recorded results of that run loaded
from the committed evidence artifact. No live Opal servers are required
to render this page; the command chunks are shown for reference and are
not executed.

## Running This Validation

The federated run trains this vision template through dsFlower and
compares it with a centralized PyTorch baseline. Each node holds an
image metadata table (`id`, `relative_path`, `mask_path`, `label`) while
the PNG files stay on the matching Rock server; only aggregate training
history and model parameters return to the researcher.

**1. Connect to the DataSHIELD nodes.** Open a single session across the
three Opal nodes that serve the image metadata table.

``` r

library(DSI)
library(DSOpal)
library(dsFlowerClient)

builder <- DSI::newDSLoginBuilder()
builder$append(server = 'node1', url = 'https://opal-node-1.example.org',
               user = 'researcher', password = '********',
               table = 'dsflower_demo.vision_cohort')
builder$append(server = 'node2', url = 'https://opal-node-2.example.org',
               user = 'researcher', password = '********',
               table = 'dsflower_demo.vision_cohort')
builder$append(server = 'node3', url = 'https://opal-node-3.example.org',
               user = 'researcher', password = '********',
               table = 'dsflower_demo.vision_cohort')
conns <- DSI::datashield.login(builder$build(), assign = TRUE, symbol = 'D')
```

**2. Run the federated fit.** Build the model and recipe, stage the
nodes, run FedAvg over the server-side image tables, then clean up and
log out. Only aggregate training history and model parameters are
returned to the researcher.

``` r

model <- ds.flower.model.pytorch_densenet121(
  n_classes = 2L, learning_rate = 0.001,
  batch_size = 2L, local_epochs = 1L, image_size = 64L
)

recipe <- ds.flower.recipe(
  model      = model,
  strategy   = ds.flower.strategy.fedavg(),
  privacy    = ds.flower.privacy.sandbox_open(),
  target     = 'label',
  num_rounds = 1
)

ds.flower.nodes.init(conns, data = 'D', symbol = 'flower')
ds.flower.nodes.prepare(
  conns, symbol = 'flower',
  target_column = 'label',
  run_config = c(recipe$model$params, recipe$strategy$params,
                 list(data_type = 'image')),
  privacy = recipe$privacy, template_name = recipe$model$template
)
ds.flower.nodes.ensure(conns, symbol = 'flower', template_name = recipe$model$template)
fit <- ds.flower.run.start(recipe, conns = conns, verbose = TRUE)
ds.flower.nodes.cleanup(conns, symbol = 'flower')

DSI::datashield.logout(conns)
```

## Recorded Results

The values below come from the committed evidence artifact for the run
described above; no live servers are contacted when the article renders.

This run trained the pytorch_densenet121 template on the classification
task (target label) for 1 federated round(s) under the sandbox_open
profile, across 3 Opal/Rock nodes that staged 12 synthetic 64x64 PNG
samples.

The centralized baseline reached a cross_entropy of 0.6033; the
federated run reached 0.9373 (delta 0.3339) with 0 client failures,
leaving this template **pass**. The comparison is shown below, followed
by the centralized vs. federated loss.

``` r

comparison <- data.frame(
  centralized_loss = row$centralized_loss,
  centralized_accuracy = row$centralized_accuracy,
  federated_loss = row$federated_loss,
  delta_loss = row$delta_loss,
  federated_n_failures = row$federated_n_failures,
  validation_status = row$validation_status
)
knitr::kable(comparison, digits = 4)
```

| centralized_loss | centralized_accuracy | federated_loss | delta_loss | federated_n_failures | validation_status |
|---:|---:|---:|---:|---:|:---|
| 0.6033 | 0.5 | 0.9373 | 0.3339 | 0 | pass |

``` r

loss_df <- data.frame(
  mode = c('Centralized', 'Federated'),
  loss = c(row$centralized_loss, row$federated_loss)
)
loss_df <- loss_df[is.finite(loss_df$loss), , drop = FALSE]
if (nrow(loss_df) > 0 && requireNamespace('ggplot2', quietly = TRUE)) {
  ggplot2::ggplot(loss_df, ggplot2::aes(x = mode, y = loss, fill = mode)) +
    ggplot2::geom_col(width = 0.65, show.legend = FALSE) +
    ggplot2::labs(x = NULL, y = row$centralized_metric[[1]], title = method_id) +
    ggplot2::theme_minimal(base_size = 11)
} else {
  plot.new()
  text(0.5, 0.5, 'No federated loss available for this method')
}
```

![Bar chart comparing centralized and federated validation loss for
pytorch_densenet121.](validation-vision-pytorch-densenet121_files/figure-html/unnamed-chunk-2-1.png)
