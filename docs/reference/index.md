# Package index

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

- [`ds.flower.model.sklearn_logreg()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.sklearn_logreg.md)
  : Create a scikit-learn Logistic Regression model spec
- [`ds.flower.model.sklearn_ridge()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.sklearn_ridge.md)
  : Create a scikit-learn Ridge Classifier model spec
- [`ds.flower.model.sklearn_sgd()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.sklearn_sgd.md)
  : Create a scikit-learn SGD Classifier model spec
- [`ds.flower.model.pytorch_mlp()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_mlp.md)
  : Create a PyTorch MLP model spec
- [`ds.flower.model.pytorch_resnet18()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_resnet18.md)
  : Create a PyTorch ResNet-18 model spec
- [`ds.flower.model.pytorch_densenet121()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_densenet121.md)
  : Create a PyTorch DenseNet-121 model spec
- [`ds.flower.model.pytorch_unet2d()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_unet2d.md)
  : Create a PyTorch U-Net 2D model spec

## Specification: Strategies

Federated aggregation strategy specifications.

- [`ds.flower.strategy.fedavg()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedavg.md)
  : Create a FedAvg strategy spec
- [`ds.flower.strategy.fedprox()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedprox.md)
  : Create a FedProx strategy spec

## Specification: Tasks & Privacy

Task types and privacy enhancement specifications.

- [`ds.flower.task.classification()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.classification.md)
  : Create a classification task specification
- [`ds.flower.task.regression()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.regression.md)
  : Create a regression task specification
- [`ds.flower.task.segmentation()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.segmentation.md)
  : Create a segmentation task specification
- [`ds.flower.task.survival()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.survival.md)
  : Create a survival task specification
- [`ds.flower.privacy.sandbox_open()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.sandbox_open.md)
  : Create a sandbox_open privacy spec
- [`ds.flower.privacy.trusted_internal()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.trusted_internal.md)
  : Create a trusted_internal privacy spec
- [`ds.flower.privacy.consortium_internal()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.consortium_internal.md)
  : Create a consortium_internal privacy spec
- [`ds.flower.privacy.clinical_default()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.clinical_default.md)
  : Create a clinical_default privacy spec (recommended)
- [`ds.flower.privacy.clinical_hardened()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.clinical_hardened.md)
  : Create a clinical_hardened privacy spec
- [`ds.flower.privacy.clinical_update_noise()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.clinical_update_noise.md)
  : Create a clinical_update_noise privacy spec
- [`ds.flower.privacy.high_sensitivity_dp()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.high_sensitivity_dp.md)
  : Create a high_sensitivity_dp privacy spec
- [`ds.flower.privacy.evaluation_only()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.evaluation_only.md)
  : Apply evaluation_only modifier to a privacy spec
- [`ds.flower.privacy.budget()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.budget.md)
  : Query remaining privacy budget on all servers

## Specification: Recipe

Combine task, model, strategy, and privacy into a recipe.

- [`ds.flower.recipe()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.recipe.md)
  : Create a Flower federated learning recipe

## Results & Metrics

Collect, compare, and visualize training results.

- [`ds.flower.metrics()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.metrics.md)
  : Get training metrics from all servers
- [`ds.flower.compare()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.compare.md)
  : Compare metrics across multiple training runs
- [`ds.flower.plot()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.plot.md)
  : Plot training curves
- [`ds.flower.log()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.log.md)
  : Get log output from all servers
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
- [`print(`*`<dsflower_privacy>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_privacy.md)
  : Print a dsflower_privacy
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

- [`ds.flower.code()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.code.md)
  : Get the R code that produced a result
- [`ds.flower.compare()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.compare.md)
  : Compare metrics across multiple training runs
- [`ds.flower.connect()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.connect.md)
  : Connect to a data source for federated learning
- [`ds.flower.copy_code()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.copy_code.md)
  : Copy reproducible R code to clipboard
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
- [`ds.flower.labels()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.labels.md)
  : List available label sets for an imaging dataset
- [`ds.flower.load_model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.load_model.md)
  : Load a saved model
- [`ds.flower.log()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.log.md)
  : Get log output from all servers
- [`ds.flower.masks()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.masks.md)
  : List available segmentation masks
- [`ds.flower.metrics()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.metrics.md)
  : Get training metrics from all servers
- [`ds.flower.model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.md)
  : Create a model spec by name
- [`ds.flower.model.pytorch_cause_specific_cox()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_cause_specific_cox.md)
  : Create a Cause-Specific Cox model spec
- [`ds.flower.model.pytorch_coxph()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_coxph.md)
  : Create a PyTorch Cox Proportional Hazards model spec
- [`ds.flower.model.pytorch_densenet121()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_densenet121.md)
  : Create a PyTorch DenseNet-121 model spec
- [`ds.flower.model.pytorch_linear_regression()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_linear_regression.md)
  : Create a PyTorch Linear Regression model spec
- [`ds.flower.model.pytorch_lognormal_aft()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_lognormal_aft.md)
  : Create a Log-Normal AFT survival model spec
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
- [`ds.flower.model.pytorch_unet2d()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.pytorch_unet2d.md)
  : Create a PyTorch U-Net 2D model spec
- [`ds.flower.model.sklearn_elastic_net()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.sklearn_elastic_net.md)
  : Create an Elastic Net model spec
- [`ds.flower.model.sklearn_logreg()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.sklearn_logreg.md)
  : Create a scikit-learn Logistic Regression model spec
- [`ds.flower.model.sklearn_ridge()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.sklearn_ridge.md)
  : Create a scikit-learn Ridge Classifier model spec
- [`ds.flower.model.sklearn_sgd()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.sklearn_sgd.md)
  : Create a scikit-learn SGD Classifier model spec
- [`ds.flower.model.sklearn_svm()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.sklearn_svm.md)
  : Create a Linear SVM model spec
- [`ds.flower.model.xgboost()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.xgboost.md)
  : Create an XGBoost model spec
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
- [`ds.flower.privacy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.md)
  : Create a privacy spec by name
- [`ds.flower.privacy.auto()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.auto.md)
  : Create an automatic privacy spec
- [`ds.flower.privacy.budget()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.budget.md)
  : Query remaining privacy budget on all servers
- [`ds.flower.privacy.clinical_default()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.clinical_default.md)
  : Create a clinical_default privacy spec (recommended)
- [`ds.flower.privacy.clinical_hardened()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.clinical_hardened.md)
  : Create a clinical_hardened privacy spec
- [`ds.flower.privacy.clinical_update_noise()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.clinical_update_noise.md)
  : Create a clinical_update_noise privacy spec
- [`ds.flower.privacy.consortium_internal()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.consortium_internal.md)
  : Create a consortium_internal privacy spec
- [`ds.flower.privacy.evaluation_only()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.evaluation_only.md)
  : Apply evaluation_only modifier to a privacy spec
- [`ds.flower.privacy.high_sensitivity_dp()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.high_sensitivity_dp.md)
  : Create a high_sensitivity_dp privacy spec
- [`ds.flower.privacy.sandbox_open()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.sandbox_open.md)
  : Create a sandbox_open privacy spec
- [`ds.flower.privacy.trusted_internal()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.trusted_internal.md)
  : Create a trusted_internal privacy spec
- [`ds.flower.recipe()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.recipe.md)
  : Create a Flower federated learning recipe
- [`ds.flower.run()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.md)
  : Run federated learning (auto-managed)
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
- [`ds.flower.strategy.fedbn()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedbn.md)
  : Create a FedBN strategy spec
- [`ds.flower.strategy.fedprox()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.fedprox.md)
  : Create a FedProx strategy spec
- [`ds.flower.superlink.attach()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.attach.md)
  : Attach to a detached SuperLink
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
- [`ds.flower.task.regression()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.regression.md)
  : Create a regression task specification
- [`ds.flower.task.segmentation()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.segmentation.md)
  : Create a segmentation task specification
- [`ds.flower.task.survival()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.survival.md)
  : Create a survival task specification
- [`ds.flower.templates()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.templates.md)
  : List templates available on the servers
- [`ds.flower.train()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.train.md)
  : One-shot federated learning (simplest path)
- [`print(`*`<dsflower_model>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_model.md)
  : Print a dsflower_model
- [`print(`*`<dsflower_privacy>`*`)`](https://isglobal-brge.github.io/dsFlowerClient/reference/print.dsflower_privacy.md)
  : Print a dsflower_privacy
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
