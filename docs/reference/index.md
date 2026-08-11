# Package index

## High-level submission

Declarative DP training and the separately gated HookApp contract.

- [`ds.flower.fit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.fit.md)
  : Fit a federated model in one call
- [`ds.flower.submit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.submit.md)
  : Submit + run a federated DP job from a model spec (the "pack" API)
- [`ds.flower.hook.run()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.hook.run.md)
  : Request a HookApp run through the node-side egress policy
- [`ds.flower.app.upload()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.app.upload.md)
  : Upload and hash-verify a HookApp candidate archive

## Lifecycle: Node Management

Connect, prepare, and manage federated nodes on Opal servers.

- [`ds.flower.nodes.init()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.init.md)
  : Initialize Flower handles on all servers
- [`ds.flower.nodes.prepare()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.prepare.md)
  : Prepare a training run on all servers
- [`ds.flower.nodes.ensure()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.ensure.md)
  : Ensure SuperNodes are running on all servers
- [`ds.flower.nodes.cleanup()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.cleanup.md)
  : Clean up training run on all servers

## Lifecycle: SuperLink

Start, monitor, and stop the local Flower SuperLink process.

- [`ds.flower.superlink.start()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.start.md)
  : Start a Flower SuperLink
- [`ds.flower.superlink.status()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.status.md)
  : Get SuperLink status
- [`ds.flower.superlink.stop()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.stop.md)
  : Stop the Flower SuperLink

## Lifecycle: Training Runs

Launch and manage federated training runs.

- [`ds.flower.run.start()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.start.md)
  : Start a Flower run
- [`ds.flower.run.stop()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.stop.md)
  : Stop a Flower run
- [`ds.flower.run.list()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.list.md)
  : List Flower runs
- [`ds.flower.run.logs()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.logs.md)
  : Get Flower run logs

## Specification: Models

Model specification objects for federated learning.

- [`ds.flower.model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.md)
  : Create a model spec by name
- [`ds.flower.model.pytorch_linear_regression()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_linear_regression.md)
  : Create a PyTorch Linear Regression model spec
- [`ds.flower.model.pytorch_logreg()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_logreg.md)
  : Create a PyTorch Logistic Regression model spec
- [`ds.flower.model.pytorch_densenet121()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_densenet121.md)
  : Create a PyTorch DenseNet-121 model spec
- [`ds.flower.model.pytorch_lstm()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_lstm.md)
  : Create a PyTorch LSTM model spec
- [`ds.flower.model.pytorch_mlp()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_mlp.md)
  : Create a PyTorch MLP model spec
- [`ds.flower.model.pytorch_multiclass()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_multiclass.md)
  : Create a PyTorch Multi-Class Classifier model spec
- [`ds.flower.model.pytorch_multilabel()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_multilabel.md)
  : Create a Multi-Label Classification model spec
- [`ds.flower.model.pytorch_poisson()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_poisson.md)
  : Create a Poisson Regression model spec
- [`ds.flower.model.pytorch_resnet18()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_resnet18.md)
  : Create a PyTorch ResNet-18 model spec
- [`ds.flower.model.pytorch_tcn()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_tcn.md)
  : Create a PyTorch TCN model spec
- [`ds.flower.model.xgboost()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.xgboost.md)
  : Create a native-tight XGBoost request spec
- [`ds.flower.model.extra_trees()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.extra_trees.md)
  : Create a private ExtraTrees request spec
- [`ds.flower.model.random_forest()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.random_forest.md)
  : Create an adaptive private Random Forest request spec
- [`ds.flower.model.lightgbm()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.lightgbm.md)
  : Create a dsFlower LightGBM-style private boosting request
- [`ds.flower.model.catboost()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.catboost.md)
  : Create a dsFlower CatBoost-style private boosting request

## Specification: Strategies

Federated aggregation strategy specifications.

- [`ds.flower.strategy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.md)
  : Create a strategy spec by name
- [`ds.flower.strategy.fedavg()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedavg.md)
  : Create a FedAvg strategy spec
- [`ds.flower.strategy.fedadam()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedadam.md)
  : Create a FedAdam strategy spec
- [`ds.flower.strategy.fedadagrad()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedadagrad.md)
  : Create a FedAdagrad strategy spec
- [`ds.flower.strategy.fedyogi()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedyogi.md)
  : Create a FedYogi strategy spec
- [`ds.flower.strategy.fedavgm()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedavgm.md)
  : Create a FedAvgM strategy spec

## Specification: Tasks

Task types. Differential privacy policy is node-owned, not a client
specification.

- [`ds.flower.task()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.md)
  : Create a task spec by name
- [`ds.flower.task.classification()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.classification.md)
  : Create a classification task specification
- [`ds.flower.task.regression()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.regression.md)
  : Create a regression task specification

## Specification: Recipe

Combine task, model, and strategy into a recipe; privacy remains
node-owned.

- [`ds.flower.recipe()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.recipe.md)
  : Create a Flower federated learning recipe

## Results

Work with model results and local coordinator diagnostics.

- [`ds.flower.compare()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.compare.md)
  : Compare metrics across multiple training runs
- [`ds.flower.plot()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.plot.md)
  : Plot training curves
- [`dsflower_result()`](https://isglobal-brge.github.io/dsFlowerClient/reference/dsflower_result.md)
  : Create a dsflower_result object
- [`ds.flower.code()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.code.md)
  : Get the R code that produced a result
- [`ds.flower.copy_code()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.copy_code.md)
  : Copy reproducible R code to clipboard

## Print Methods

- [`print(`*`<dsflower_model>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_model.md)
  : Print a dsflower_model
- [`print(`*`<dsflower_strategy>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_strategy.md)
  : Print a dsflower_strategy
- [`print(`*`<dsflower_task>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_task.md)
  : Print a dsflower_task
- [`print(`*`<dsflower_recipe>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_recipe.md)
  : Print a dsflower_recipe
- [`print(`*`<dsflower_result>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_result.md)
  : Print a dsflower_result
- [`as.data.frame(`*`<dsflower_result>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/as.data.frame.dsflower_result.md)
  : Convert dsflower_result to data.frame
- [`` `$`( ``*`<dsflower_result>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/cash-.dsflower_result.md)
  : Access dsflower_result elements

## Full Exported API

Catch-all index for exported helpers not listed above.

- [`ds.flower.app.upload()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.app.upload.md)
  : Upload and hash-verify a HookApp candidate archive
- [`ds.flower.associate()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.associate.md)
  : Differentially-private pooled binary association
- [`ds.flower.code()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.code.md)
  : Get the R code that produced a result
- [`ds.flower.compare()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.compare.md)
  : Compare metrics across multiple training runs
- [`ds.flower.connect()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.connect.md)
  : Connect to a data source for federated learning
- [`ds.flower.copy_code()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.copy_code.md)
  : Copy reproducible R code to clipboard
- [`ds.flower.cross_validate()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.cross_validate.md)
  : Cross-validate a tabular neural or native-tree model across
  federated data
- [`ds.flower.delete_model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.delete_model.md)
  : Delete a saved model
- [`ds.flower.describe()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.describe.md)
  : Describe the connected dataset
- [`ds.flower.disconnect()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.disconnect.md)
  : Disconnect and clean up a flower connection
- [`ds.flower.features()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.features.md)
  : List available feature assets
- [`ds.flower.fit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.fit.md)
  : Fit a federated model in one call
- [`ds.flower.hook.run()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.hook.run.md)
  : Request a HookApp run through the node-side egress policy
- [`ds.flower.hpo()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.hpo.md)
  : Optimize an explicit objective locally with Optuna
- [`ds.flower.hpo.float()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.hpo.dimensions.md)
  [`ds.flower.hpo.integer()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.hpo.dimensions.md)
  [`ds.flower.hpo.categorical()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.hpo.dimensions.md)
  : Define a bounded floating-point HPO dimension
- [`ds.flower.import_xgboost()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.import_xgboost.md)
  : Import a Bounded External XGBoost JSON Model
- [`ds.flower.labels()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.labels.md)
  : List available label sets for an imaging dataset
- [`ds.flower.link.down()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.link.down.md)
  : Close the federation link
- [`ds.flower.link.up()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.link.up.md)
  : Open the federation link
- [`ds.flower.list_models()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.list_models.md)
  : List registered dsFlower models
- [`ds.flower.load_model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.load_model.md)
  : Load a saved model
- [`ds.flower.masks()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.masks.md)
  : List available segmentation masks
- [`ds.flower.metrics()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.metrics.md)
  [`ds.flower.score()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.metrics.md)
  [`ds.flower.metric_direction()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.metrics.md)
  : Inspect and select private validation metrics
- [`ds.flower.model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.md)
  : Create a model spec by name
- [`ds.flower.model.catboost()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.catboost.md)
  : Create a dsFlower CatBoost-style private boosting request
- [`ds.flower.model.extra_trees()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.extra_trees.md)
  : Create a private ExtraTrees request spec
- [`ds.flower.model.lightgbm()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.lightgbm.md)
  : Create a dsFlower LightGBM-style private boosting request
- [`ds.flower.model.pytorch_densenet121()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_densenet121.md)
  : Create a PyTorch DenseNet-121 model spec
- [`ds.flower.model.pytorch_linear_regression()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_linear_regression.md)
  : Create a PyTorch Linear Regression model spec
- [`ds.flower.model.pytorch_logreg()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_logreg.md)
  : Create a PyTorch Logistic Regression model spec
- [`ds.flower.model.pytorch_lstm()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_lstm.md)
  : Create a PyTorch LSTM model spec
- [`ds.flower.model.pytorch_mlp()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_mlp.md)
  : Create a PyTorch MLP model spec
- [`ds.flower.model.pytorch_multiclass()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_multiclass.md)
  : Create a PyTorch Multi-Class Classifier model spec
- [`ds.flower.model.pytorch_multilabel()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_multilabel.md)
  : Create a Multi-Label Classification model spec
- [`ds.flower.model.pytorch_poisson()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_poisson.md)
  : Create a Poisson Regression model spec
- [`ds.flower.model.pytorch_resnet18()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_resnet18.md)
  : Create a PyTorch ResNet-18 model spec
- [`ds.flower.model.pytorch_tcn()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_tcn.md)
  : Create a PyTorch TCN model spec
- [`ds.flower.model.random_forest()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.random_forest.md)
  : Create an adaptive private Random Forest request spec
- [`ds.flower.model.xgboost()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.xgboost.md)
  : Create a native-tight XGBoost request spec
- [`ds.flower.model_parameters()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model_parameters.md)
  : Inspect the public parameter contract for a registered model
- [`ds.flower.models()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.models.md)
  : List saved models
- [`ds.flower.nodes.cleanup()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.cleanup.md)
  : Clean up training run on all servers
- [`ds.flower.nodes.ensure()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.ensure.md)
  : Ensure SuperNodes are running on all servers
- [`ds.flower.nodes.init()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.init.md)
  : Initialize Flower handles on all servers
- [`ds.flower.nodes.prepare()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.prepare.md)
  : Prepare a training run on all servers
- [`ds.flower.plot()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.plot.md)
  : Plot training curves
- [`ds.flower.predict()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.predict.md)
  : Predict with a federated model
- [`ds.flower.recipe()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.recipe.md)
  : Create a Flower federated learning recipe
- [`ds.flower.register_model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.register_model.md)
  : Register a dsFlower model generator
- [`ds.flower.run.list()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.list.md)
  : List Flower runs
- [`ds.flower.run.logs()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.logs.md)
  : Get Flower run logs
- [`ds.flower.run.start()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.start.md)
  : Start a Flower run
- [`ds.flower.run.stop()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.stop.md)
  : Stop a Flower run
- [`ds.flower.save_model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.save_model.md)
  : Save the global model from a training run
- [`ds.flower.strategy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.md)
  : Create a strategy spec by name
- [`ds.flower.strategy.fedadagrad()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedadagrad.md)
  : Create a FedAdagrad strategy spec
- [`ds.flower.strategy.fedadam()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedadam.md)
  : Create a FedAdam strategy spec
- [`ds.flower.strategy.fedavg()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedavg.md)
  : Create a FedAvg strategy spec
- [`ds.flower.strategy.fedavgm()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedavgm.md)
  : Create a FedAvgM strategy spec
- [`ds.flower.strategy.fedyogi()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedyogi.md)
  : Create a FedYogi strategy spec
- [`ds.flower.submit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.submit.md)
  : Submit + run a federated DP job from a model spec (the "pack" API)
- [`ds.flower.superlink.start()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.start.md)
  : Start a Flower SuperLink
- [`ds.flower.superlink.status()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.status.md)
  : Get SuperLink status
- [`ds.flower.superlink.stop()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.stop.md)
  : Stop the Flower SuperLink
- [`ds.flower.task()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.md)
  : Create a task spec by name
- [`ds.flower.task.classification()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.classification.md)
  : Create a classification task specification
- [`ds.flower.task.count()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.count.md)
  : Create a count-outcome task specification
- [`ds.flower.task.regression()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.regression.md)
  : Create a regression task specification
- [`ds.flower.validate()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.validate.md)
  : Differentially-private federated model validation
- [`print(`*`<dsflower_app>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_app.md)
  : Print a dsflower_app
- [`print(`*`<dsflower_association>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_association.md)
  : Print a pooled private association
- [`print(`*`<dsflower_cv>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_cv.md)
  : Print a federated cross-validation result
- [`print(`*`<dsflower_hpo>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_hpo.md)
  : Print a local HPO result
- [`print(`*`<dsflower_model>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_model.md)
  : Print a dsflower_model
- [`print(`*`<dsflower_recipe>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_recipe.md)
  : Print a dsflower_recipe
- [`print(`*`<dsflower_result>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_result.md)
  : Print a dsflower_result
- [`print(`*`<dsflower_run>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_run.md)
  : Print a dsflower_run
- [`print(`*`<dsflower_strategy>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_strategy.md)
  : Print a dsflower_strategy
- [`print(`*`<dsflower_task>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_task.md)
  : Print a dsflower_task
- [`print(`*`<dsflower_validation>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_validation.md)
  : Print a private dsFlower validation result
