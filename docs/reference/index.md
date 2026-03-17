# Package index

## Lifecycle: Node Management

Connect, prepare, and manage federated nodes on Opal servers.

- [`ds.flower.nodes.init()`](ds.flower.nodes.init.md) : Initialize
  Flower handles on all servers
- [`ds.flower.nodes.prepare()`](ds.flower.nodes.prepare.md) : Prepare a
  training run on all servers
- [`ds.flower.nodes.ensure()`](ds.flower.nodes.ensure.md) : Ensure
  SuperNodes are running on all servers
- [`ds.flower.nodes.cleanup()`](ds.flower.nodes.cleanup.md) : Clean up
  training run on all servers

## Lifecycle: SuperLink

Start, monitor, and stop the local Flower SuperLink process.

- [`ds.flower.superlink.start()`](ds.flower.superlink.start.md) : Start
  a Flower SuperLink
- [`ds.flower.superlink.status()`](ds.flower.superlink.status.md) : Get
  SuperLink status
- [`ds.flower.superlink.stop()`](ds.flower.superlink.stop.md) : Stop the
  Flower SuperLink

## Lifecycle: Training Runs

Launch and manage federated training runs.

- [`ds.flower.run.start()`](ds.flower.run.start.md) : Start a Flower run
- [`ds.flower.run.stop()`](ds.flower.run.stop.md) : Stop a Flower run
- [`ds.flower.run.list()`](ds.flower.run.list.md) : List Flower runs
- [`ds.flower.run.logs()`](ds.flower.run.logs.md) : Get Flower run logs

## Specification: Models

Model specification objects for federated learning.

- [`ds.flower.model.sklearn_logreg()`](ds.flower.model.sklearn_logreg.md)
  : Create a scikit-learn Logistic Regression model spec
- [`ds.flower.model.sklearn_ridge()`](ds.flower.model.sklearn_ridge.md)
  : Create a scikit-learn Ridge Classifier model spec
- [`ds.flower.model.sklearn_sgd()`](ds.flower.model.sklearn_sgd.md) :
  Create a scikit-learn SGD Classifier model spec
- [`ds.flower.model.pytorch_mlp()`](ds.flower.model.pytorch_mlp.md) :
  Create a PyTorch MLP model spec

## Specification: Strategies

Federated aggregation strategy specifications.

- [`ds.flower.strategy.fedavg()`](ds.flower.strategy.fedavg.md) : Create
  a FedAvg strategy spec
- [`ds.flower.strategy.fedprox()`](ds.flower.strategy.fedprox.md) :
  Create a FedProx strategy spec

## Specification: Tasks & Privacy

Task types and privacy enhancement specifications.

- [`ds.flower.task.classification()`](ds.flower.task.classification.md)
  : Create a classification task specification
- [`ds.flower.task.regression()`](ds.flower.task.regression.md) : Create
  a regression task specification
- [`ds.flower.privacy.research()`](ds.flower.privacy.research.md) :
  Create a research-mode privacy spec (no enhancements)
- [`ds.flower.privacy.dp()`](ds.flower.privacy.dp.md) : Create a
  differential privacy spec

## Specification: Recipe

Combine task, model, strategy, and privacy into a recipe.

- [`ds.flower.recipe()`](ds.flower.recipe.md) : Create a Flower
  federated learning recipe

## Results & Metrics

Collect, compare, and visualize training results.

- [`ds.flower.metrics()`](ds.flower.metrics.md) : Get training metrics
  from all servers
- [`ds.flower.compare()`](ds.flower.compare.md) : Compare metrics across
  multiple training runs
- [`ds.flower.plot()`](ds.flower.plot.md) : Plot training curves
- [`ds.flower.log()`](ds.flower.log.md) : Get log output from all
  servers
- [`dsflower_result()`](dsflower_result.md) : Create a dsflower_result
  object
- [`ds.flower.code()`](ds.flower.code.md) : Get the R code that produced
  a result
- [`ds.flower.copy_code()`](ds.flower.copy_code.md) : Copy reproducible
  R code to clipboard

## Print Methods

- [`print(`*`<dsflower_model>`*`)`](print.dsflower_model.md) : Print a
  dsflower_model
- [`print(`*`<dsflower_strategy>`*`)`](print.dsflower_strategy.md) :
  Print a dsflower_strategy
- [`print(`*`<dsflower_task>`*`)`](print.dsflower_task.md) : Print a
  dsflower_task
- [`print(`*`<dsflower_privacy>`*`)`](print.dsflower_privacy.md) : Print
  a dsflower_privacy
- [`print(`*`<dsflower_recipe>`*`)`](print.dsflower_recipe.md) : Print a
  dsflower_recipe
- [`print(`*`<dsflower_result>`*`)`](print.dsflower_result.md) : Print a
  dsflower_result
- [`as.data.frame(`*`<dsflower_result>`*`)`](as.data.frame.dsflower_result.md)
  : Convert dsflower_result to data.frame
- [`` `$`( ``*`<dsflower_result>`*`)`](cash-.dsflower_result.md) :
  Access dsflower_result elements
