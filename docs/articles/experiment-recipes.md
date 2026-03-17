# Composing Experiments with Recipes

## The recipe system

A recipe in dsFlowerClient is a composition of four building blocks:

- **Task**: classification or regression
- **Model**: the learning algorithm and its hyperparameters
- **Strategy**: how the server aggregates updates from the nodes
- **Privacy**: whether to apply differential privacy

Each block is an independent S3 object. You can mix and match them
freely. The recipe itself is just a specification. Nothing runs until
you call [`ds.flower.run.start()`](../reference/ds.flower.run.start.md).

``` r

library(dsFlowerClient)
```

## Available models

### Logistic Regression

A solid default for binary classification. Converges fast in federated
settings because the model is small (one weight per feature).

``` r

model_lr <- ds.flower.model.sklearn_logreg(
  penalty  = "l2",
  C        = 1.0,
  max_iter = 200L
)
model_lr
#> dsflower_model: sklearn_logreg ( sklearn )
#>    penalty = "l2" 
#>    C = 1 
#>    max_iter = 200L
```

### SGD Classifier

Supports several loss functions and learning rate schedules. The
`"log_loss"` loss function gives logistic regression with SGD
optimization.

``` r

model_sgd <- ds.flower.model.sklearn_sgd(
  loss        = "log_loss",
  alpha       = 0.0001,
  lr_schedule = "optimal"
)
model_sgd
#> dsflower_model: sklearn_sgd ( sklearn )
#>    loss = "log_loss" 
#>    alpha = 1e-04 
#>    lr_schedule = "optimal"
```

### Ridge Classifier

L2-regularized linear classifier. Good for high-dimensional data.

``` r

model_ridge <- ds.flower.model.sklearn_ridge(alpha = 1.0)
model_ridge
#> dsflower_model: sklearn_ridge ( sklearn )
#>    alpha = 1
```

### PyTorch MLP

For problems where a linear model is not expressive enough:

``` r

model_mlp <- ds.flower.model.pytorch_mlp(
  hidden_layers = c(128L, 64L),
  learning_rate = 0.001,
  batch_size    = 32L,
  local_epochs  = 2L
)
model_mlp
#> dsflower_model: pytorch_mlp ( pytorch )
#>    hidden_layers = c(128, 64) 
#>    learning_rate = 0.001 
#>    batch_size = 32L 
#>    local_epochs = 2L
```

`local_epochs` controls how many passes over the local data each node
makes per round. More epochs can speed up convergence but risk local
model drift (especially with non-IID data).

## Aggregation strategies

### FedAvg

The standard algorithm. Each node trains, sends updated weights, and the
server computes a weighted average proportional to dataset size.

``` r

strat_avg <- ds.flower.strategy.fedavg(
  fraction_fit          = 1.0,
  fraction_evaluate     = 1.0,
  min_fit_clients       = 2L,
  min_available_clients = 2L
)
strat_avg
#> dsflower_strategy: FedAvg 
#>    fraction_fit = 1 
#>    fraction_evaluate = 1 
#>    min_fit_clients = 2L 
#>    min_available_clients = 2L
```

### FedProx

Adds a proximal term to keep local models closer to the global model.
Helps with heterogeneous (non-IID) data.

``` r

strat_prox <- ds.flower.strategy.fedprox(
  proximal_mu           = 0.1,
  min_fit_clients       = 2L,
  min_available_clients = 2L
)
strat_prox
#> dsflower_strategy: FedProx 
#>    proximal_mu = 0.1 
#>    fraction_fit = 1 
#>    min_fit_clients = 2L 
#>    min_available_clients = 2L
```

## Privacy settings

### Research mode

No privacy enhancements. Suitable for trusted environments.

``` r

priv <- ds.flower.privacy.research()
priv
#> dsflower_privacy: research
```

### Differential privacy

Clips gradient norms and adds calibrated noise. Smaller epsilon =
stronger privacy but noisier model.

``` r

priv_dp <- ds.flower.privacy.dp(
  epsilon       = 1.0,
  delta         = 1e-5,
  clipping_norm = 1.0
)
priv_dp
#> dsflower_privacy: dp 
#>    epsilon = 1 
#>    delta = 1e-05 
#>    clipping_norm = 1
```

## Putting it together

``` r

recipe_cls <- ds.flower.recipe(
  task            = ds.flower.task.classification(),
  model           = model_lr,
  strategy        = strat_avg,
  privacy         = ds.flower.privacy.research(),
  num_rounds      = 10L,
  target_column   = "diagnosis",
  feature_columns = c("age", "bmi", "glucose", "insulin")
)
recipe_cls
#> dsflower_recipe
#>   Task:      classification 
#>   Model:     sklearn_logreg ( sklearn )
#>   Strategy:  FedAvg 
#>   Privacy:   research 
#>   Rounds:    10 
#>   Target:    diagnosis 
#>   Features:  age, bmi, glucose, insulin
```

Regression works the same way:

``` r

recipe_reg <- ds.flower.recipe(
  task            = ds.flower.task.regression(),
  model           = ds.flower.model.sklearn_sgd(loss = "squared_error"),
  strategy        = strat_avg,
  num_rounds      = 10L,
  target_column   = "price",
  feature_columns = c("sqft", "bedrooms", "bathrooms")
)
recipe_reg
#> dsflower_recipe
#>   Task:      regression 
#>   Model:     sklearn_sgd ( sklearn )
#>   Strategy:  FedAvg 
#>   Privacy:   research 
#>   Rounds:    10 
#>   Target:    price 
#>   Features:  sqft, bedrooms, bathrooms
```

## Live demo: training different models across 3 sites

Let’s run real federated training with different model configurations
against three Opal servers.

``` r

library(DSI)
#> Loading required package: progress
#> Loading required package: R6
library(DSOpal)
#> Loading required package: opalr
#> Loading required package: httr

builder <- DSI::newDSLoginBuilder()
builder$append(server = "site_a", url = "https://localhost:8443",
               user = "administrator", password = "admin123",
               driver = "OpalDriver",
               options = "list(ssl_verifyhost=0, ssl_verifypeer=0)")
builder$append(server = "site_b", url = "https://localhost:8444",
               user = "administrator", password = "admin123",
               driver = "OpalDriver",
               options = "list(ssl_verifyhost=0, ssl_verifypeer=0)")
builder$append(server = "site_c", url = "https://localhost:8445",
               user = "administrator", password = "admin123",
               driver = "OpalDriver",
               options = "list(ssl_verifyhost=0, ssl_verifypeer=0)")

conns <- DSI::datashield.login(logins = builder$build(), assign = FALSE)
#> 
#> Logging into the collaborating servers
cat("Connected to:", paste(names(conns), collapse = ", "), "\n")
#> Connected to: site_a, site_b, site_c
```

### Experiment 1: Logistic Regression with FedAvg

``` r

quiet(ds.flower.nodes.init(conns, resource = "dsflower_test.flower_node"))
quiet(ds.flower.nodes.prepare(conns, target_column = "target",
                               feature_columns = c("f1", "f2", "f3", "f4", "f5")))
quiet(ds.flower.superlink.start(insecure = TRUE))
Sys.sleep(2)
quiet(ds.flower.nodes.ensure(conns, symbol = "flower"))
#> Warning: site_c cannot reach SuperLink at host.docker.internal:9092
#> (There are some DataSHIELD errors, list them with datashield.errors())
#> Warning: Some nodes failed connectivity check: site_c. Consider providing
#> per-node superlink_address for those nodes.
Sys.sleep(10)

recipe_exp1 <- ds.flower.recipe(
  task     = ds.flower.task.classification(),
  model    = ds.flower.model.sklearn_logreg(C = 1.0, max_iter = 200L),
  strategy = ds.flower.strategy.fedavg(
    min_fit_clients = 3L, min_available_clients = 3L),
  num_rounds      = 5L,
  target_column   = "target",
  feature_columns = c("f1", "f2", "f3", "f4", "f5")
)

cat("=== Logistic Regression + FedAvg (5 rounds) ===\n\n")
#> === Logistic Regression + FedAvg (5 rounds) ===
run_exp1 <- ds.flower.run.start(recipe_exp1, verbose = TRUE)
#> Flower App configuration warnings in '/private/var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/Rtmp0Cks8s/dsflower_app/sklearn_logreg/pyproject.toml':
#> - Recommended property "description" missing in [project]
#> - Recommended property "license" missing in [project]
#> 🎊 Successfully started run 5468004023582724476
#> INFO :      Start `flwr-serverapp` process
#> 🎊 Successfully installed sklearn_logreg to /var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/Rtmp0Cks8s/dsflower_superlink/apps/dsflower.sklearn_logreg.0.1.0.4173ea32.
#> INFO :      Starting Flower ServerApp, config: num_rounds=5, no round_timeout
#> INFO :      
#> INFO :      [INIT]
#> INFO :      Requesting initial parameters from one random client
#> INFO :      Received initial parameters from one random client
#> INFO :      Starting evaluation of initial global parameters
#> INFO :      Evaluation returned no results (`None`)
#> INFO :      
#> INFO :      [ROUND 1]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> WARNING :   No fit_metrics_aggregation_fn provided
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> WARNING :   No evaluate_metrics_aggregation_fn provided
#> INFO :      
#> INFO :      [ROUND 2]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 3]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 4]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 5]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [SUMMARY]
#> INFO :      Run finished 5 round(s) in 75.34s
#> INFO :          History (loss, distributed):
#> INFO :              round 1: 0.14106420993565436
#> INFO :              round 2: 0.14103147463155055
#> INFO :              round 3: 0.14103127348483877
#> INFO :              round 4: 0.1410312711155861
#> INFO :              round 5: 0.14103127108541957
#> INFO :      
#> INFO :
#> INFO :      Starting logstream for run_id `5468004023582724476`
cat(sprintf("\nExit status: %s\n", run_exp1$status))
#> 
#> Exit status: 0

quiet(ds.flower.nodes.cleanup(conns, symbol = "flower"))
quiet(ds.flower.superlink.stop())
```

### Experiment 2: SGD Classifier with FedAvg

``` r

quiet(ds.flower.nodes.init(conns, resource = "dsflower_test.flower_node"))
quiet(ds.flower.nodes.prepare(conns, target_column = "target",
                               feature_columns = c("f1", "f2", "f3", "f4", "f5")))
quiet(ds.flower.superlink.start(insecure = TRUE))
Sys.sleep(2)
quiet(ds.flower.nodes.ensure(conns, symbol = "flower"))
#> Warning: site_c cannot reach SuperLink at host.docker.internal:9092
#> (There are some DataSHIELD errors, list them with datashield.errors())
#> Warning: Some nodes failed connectivity check: site_c. Consider providing
#> per-node superlink_address for those nodes.
Sys.sleep(10)

recipe_exp2 <- ds.flower.recipe(
  task     = ds.flower.task.classification(),
  model    = ds.flower.model.sklearn_sgd(loss = "log_loss", alpha = 0.0001),
  strategy = ds.flower.strategy.fedavg(
    min_fit_clients = 3L, min_available_clients = 3L),
  num_rounds      = 5L,
  target_column   = "target",
  feature_columns = c("f1", "f2", "f3", "f4", "f5")
)

cat("=== SGD Classifier + FedAvg (5 rounds) ===\n\n")
#> === SGD Classifier + FedAvg (5 rounds) ===
run_exp2 <- ds.flower.run.start(recipe_exp2, verbose = TRUE)
#> Flower App configuration warnings in '/private/var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/Rtmp0Cks8s/dsflower_app/sklearn_sgd/pyproject.toml':
#> - Recommended property "description" missing in [project]
#> - Recommended property "license" missing in [project]
#> 🎊 Successfully started run 18013291716223821744
#> INFO :      Start `flwr-serverapp` process
#> 🎊 Successfully installed sklearn_sgd to /var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/Rtmp0Cks8s/dsflower_superlink/apps/dsflower.sklearn_sgd.0.1.0.343f9f24.
#> INFO :      Starting Flower ServerApp, config: num_rounds=5, no round_timeout
#> INFO :      
#> INFO :      [INIT]
#> INFO :      Requesting initial parameters from one random client
#> INFO :      Received initial parameters from one random client
#> INFO :      Starting evaluation of initial global parameters
#> INFO :      Evaluation returned no results (`None`)
#> INFO :      
#> INFO :      [ROUND 1]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> WARNING :   No fit_metrics_aggregation_fn provided
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> WARNING :   No evaluate_metrics_aggregation_fn provided
#> INFO :      
#> INFO :      [ROUND 2]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 3]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 4]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 5]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [SUMMARY]
#> INFO :      Run finished 5 round(s) in 78.37s
#> INFO :          History (loss, distributed):
#> INFO :              round 1: 0.17969965733189516
#> INFO :              round 2: 0.34973387417776713
#> INFO :              round 3: 0.18041587508312013
#> INFO :              round 4: 0.2481234142508982
#> INFO :              round 5: 0.2707563995408066
#> INFO :      
#> INFO :
#> INFO :      Starting logstream for run_id `18013291716223821744`
cat(sprintf("\nExit status: %s\n", run_exp2$status))
#> 
#> Exit status: 0

quiet(ds.flower.nodes.cleanup(conns, symbol = "flower"))
quiet(ds.flower.superlink.stop())
```

### Experiment 3: Ridge Classifier with FedAvg

``` r

quiet(ds.flower.nodes.init(conns, resource = "dsflower_test.flower_node"))
quiet(ds.flower.nodes.prepare(conns, target_column = "target",
                               feature_columns = c("f1", "f2", "f3", "f4", "f5")))
quiet(ds.flower.superlink.start(insecure = TRUE))
Sys.sleep(2)
quiet(ds.flower.nodes.ensure(conns, symbol = "flower"))
#> Warning: site_c cannot reach SuperLink at host.docker.internal:9092
#> (There are some DataSHIELD errors, list them with datashield.errors())
#> Warning: Some nodes failed connectivity check: site_c. Consider providing
#> per-node superlink_address for those nodes.
Sys.sleep(10)

recipe_exp3 <- ds.flower.recipe(
  task     = ds.flower.task.classification(),
  model    = ds.flower.model.sklearn_ridge(alpha = 1.0),
  strategy = ds.flower.strategy.fedavg(
    min_fit_clients = 3L, min_available_clients = 3L),
  num_rounds      = 5L,
  target_column   = "target",
  feature_columns = c("f1", "f2", "f3", "f4", "f5")
)

cat("=== Ridge Classifier + FedAvg (5 rounds) ===\n\n")
#> === Ridge Classifier + FedAvg (5 rounds) ===
run_exp3 <- ds.flower.run.start(recipe_exp3, verbose = TRUE)
#> Flower App configuration warnings in '/private/var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/Rtmp0Cks8s/dsflower_app/sklearn_ridge/pyproject.toml':
#> - Recommended property "description" missing in [project]
#> - Recommended property "license" missing in [project]
#> 🎊 Successfully started run 12759576435245565353
#> INFO :      Start `flwr-serverapp` process
#> 🎊 Successfully installed sklearn_ridge to /var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/Rtmp0Cks8s/dsflower_superlink/apps/dsflower.sklearn_ridge.0.1.0.f7f5d691.
#> INFO :      Starting Flower ServerApp, config: num_rounds=5, no round_timeout
#> INFO :      
#> INFO :      [INIT]
#> INFO :      Requesting initial parameters from one random client
#> INFO :      Received initial parameters from one random client
#> INFO :      Starting evaluation of initial global parameters
#> INFO :      Evaluation returned no results (`None`)
#> INFO :      
#> INFO :      [ROUND 1]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> WARNING :   No fit_metrics_aggregation_fn provided
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> WARNING :   No evaluate_metrics_aggregation_fn provided
#> INFO :      
#> INFO :      [ROUND 2]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 3]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 4]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 5]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [SUMMARY]
#> INFO :      Run finished 5 round(s) in 79.48s
#> INFO :          History (loss, distributed):
#> INFO :              round 1: 0.42592063546180725
#> INFO :              round 2: 0.42592063546180725
#> INFO :              round 3: 0.42592064539591473
#> INFO :              round 4: 0.42592063546180725
#> INFO :              round 5: 0.42592064539591473
#> INFO :      
#> INFO :
#> INFO :      Starting logstream for run_id `12759576435245565353`
cat(sprintf("\nExit status: %s\n", run_exp3$status))
#> 
#> Exit status: 0

quiet(ds.flower.nodes.cleanup(conns, symbol = "flower"))
quiet(ds.flower.superlink.stop())
```

## Under the hood: from recipe to Flower App

When you call `ds.flower.run.start(recipe)`, the recipe is converted
into a Flower App:

1.  [`.build_flower_app()`](../reference/dot-build_flower_app.md) copies
    a Python template from `inst/flower_templates/<model_template>/`
    containing `client_app.py` (local training) and `server_app.py`
    (aggregation strategy).

2.  [`.write_pyproject_toml()`](../reference/dot-write_pyproject_toml.md)
    serializes all recipe parameters into TOML:

        [tool.flwr.app.config]
        num-server-rounds = 5
        task-type = "classification"
        C = 1.0
        strategy = "FedAvg"

3.  `flwr run` installs the app, starts a ServerApp, and orchestrates
    training through the SuperLink.

## Server-side constraints

| Option                           | Default | Effect                  |
|:---------------------------------|:--------|:------------------------|
| `nfilter.subset`                 | 3       | Minimum rows per node   |
| `dsflower.max_rounds`            | 500     | Maximum training rounds |
| `dsflower.allow_custom_config`   | FALSE   | Custom run configs?     |
| `dsflower.allow_supernode_spawn` | TRUE    | Can SuperNodes start?   |
| `dsflower.max_concurrent_runs`   | 5       | Simultaneous runs       |
