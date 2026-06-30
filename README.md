# dsFlowerClient

`dsFlowerClient` is the researcher-side R package for running [Flower](https://flower.ai/)
federated learning through [DataSHIELD](https://www.datashield.org/), with **always-on,
server-enforced differential privacy**. It pairs with the node-side
[`dsFlower`](https://github.com/isglobal-brge/dsFlower) package installed on each Opal/Rock
server.

The client starts a local Flower SuperLink, asks each DataSHIELD server to stage data and
start a Flower SuperNode, runs the model, and returns only the privatised outputs. **Raw
rows, images and staging files never leave the data-owning servers, and the analyst never
sets the privacy level — each data node decides and enforces it.**

## Installation

```r
remotes::install_github("isglobal-brge/dsFlowerClient")
```

The client needs the Flower CLI:

```sh
python -m pip install "flwr>=1.13.0"
```

Each Opal/Rock server must have `dsFlower` installed.

## Quick start

One call does the whole lifecycle (capability checks → server-side staging → run → cleanup):

```r
library(dsFlowerClient)
library(DSI); library(DSOpal)

builder <- DSI::newDSLoginBuilder()
builder$append(server = "site1", url = "https://opal1.example.org",
               user = "researcher", password = "...",
               table = "PROJECT.training_data", driver = "OpalDriver")
builder$append(server = "site2", url = "https://opal2.example.org",
               user = "researcher", password = "...",
               table = "PROJECT.training_data", driver = "OpalDriver")
conns <- DSI::datashield.login(builder$build(), assign = TRUE, symbol = "D")

fit <- ds.flower.fit(conns, symbol = "D", target = "outcome", model = "pytorch_logreg")

ds.flower.metrics(fit)
DSI::datashield.logout(conns)
```

That is the whole minimal call: `conns`, the data `symbol`, the `target`, and a `model`.
Everything else has sensible defaults — `features` defaults to every column except the
target, `strategy = "fedavg"`, `rounds = 5`, and **differential privacy is always applied by
the node**. `ds.flower.train()` is an alias of `ds.flower.fit()`.

## Privacy is server-authoritative (you don't set it)

Differential privacy is always on and is decided **entirely by each data node**: the node
sets the (epsilon, delta, clipping) from its own DataSHIELD options (with hard ceilings),
chooses the DP mechanism from what you submit, and debits a Rényi-DP budget ledger. The
client has no privacy knob — it cannot weaken DP, nor even request a different value. You can
only **read** the node's real remaining budget:

```r
ds.flower.privacy.budget(conns, symbol = "D")
```

## Models

20 model families, all PyTorch or XGBoost (sent to the node as declarative **specs**, never
code):

| Family | Models |
|---|---|
| Linear / GLM | `pytorch_logreg`, `pytorch_linear_regression`, `pytorch_multiclass`, `pytorch_multilabel`, `pytorch_poisson`, `pytorch_negbin`, `pytorch_gamma`, `pytorch_ordinal` |
| Penalised linear / SVM | `pytorch_ridge`, `pytorch_lasso`, `pytorch_elasticnet`, `pytorch_svm` |
| Deep nets | `pytorch_mlp`, `pytorch_cnn`, `pytorch_tcn`, `pytorch_resnet`, `pytorch_transformer`, `pytorch_lstm`, `pytorch_gru` |
| Gradient boosting | `xgboost` (DP-GBDT) |

Set hyperparameters via `model_params`:

```r
fit <- ds.flower.fit(conns, symbol = "D", target = "y",
                     model = "pytorch_mlp", model_params = list(hidden_layers = c(64, 32)),
                     rounds = 10L)
```

## How the privacy guarantee is enforced (node side)

The node routes every submission to the tightest sound DP mechanism **by construction**:

- **Declarative neural specs** → Opacus **DP-SGD** (per-sample clip + noise). The declarative
  graph language covers MLP/CNN/TCN/ResNet/DenseNet/Inception/Transformer/U-Net/LSTM/GRU with
  no researcher code on the node.
- **XGBoost spec** → **DP-GBDT**.
- **Arbitrary uploaded code** (`ds.flower.tier2.run`) → an **output-perturbation floor** run
  **out-of-process**: the untrusted code executes in an isolated interpreter, and the trusted
  node applies all DP itself, so the upload can never disable the noise. Budget is reserved
  before any result is released.

See the [`dsFlower` architecture notes](https://github.com/isglobal-brge/dsFlower/blob/main/ARCHITECTURE.md)
for the full trust model.

## Lower-level API (power users)

`ds.flower.fit()` is enough for most analyses. For explicit control, build a recipe:

```r
flower <- ds.flower.connect(conns, symbol = "D")
recipe <- ds.flower.recipe(
  model    = ds.flower.model("pytorch_mlp", hidden_layers = c(64, 32)),
  strategy = ds.flower.strategy("fedprox", proximal_mu = 0.1),
  target   = "outcome",
  num_rounds = 10L
)
result <- ds.flower.run(flower, recipe)
ds.flower.disconnect(flower)
```

## Authors

- **David Sarrat González** — david.sarrat@isglobal.org
- **Juan R González** — juanr.gonzalez@isglobal.org

[Barcelona Institute for Global Health (ISGlobal)](https://www.isglobal.org/)
