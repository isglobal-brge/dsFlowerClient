# dsFlowerClient

Client-side R package for [dsFlower](https://github.com/isglobal-brge/dsFlower) — federated learning on [DataSHIELD](https://www.datashield.org/) powered by [Flower](https://flower.ai/).

## Installation

```r
remotes::install_github("isglobal-brge/dsFlowerClient")
```

Requires Python with [Flower](https://flower.ai/): `pip install flwr>=1.13.0`

## Usage

```r
library(dsFlowerClient)
library(DSI)
library(DSOpal)

# Connect to Opal nodes
builder <- DSI::newDSLoginBuilder()
builder$append(server = "site1", url = "https://opal1.example.org",
               user = "researcher", password = "...",
               table = "PROJECT.data", driver = "OpalDriver")
builder$append(server = "site2", url = "https://opal2.example.org",
               user = "researcher", password = "...",
               table = "PROJECT.data", driver = "OpalDriver")
conns <- DSI::datashield.login(logins = builder$build(),
                               assign = TRUE, symbol = "D")
flower <- ds.flower.connect(conns, symbol = "D")

# Train
result <- ds.flower.run(flower, ds.flower.recipe(
  model         = ds.flower.model.pytorch_mlp(hidden_layers = "64,32"),
  strategy      = ds.flower.strategy.fedprox(proximal_mu = 0.1),
  target_column = "outcome",
  num_rounds    = 10L
))

# Cleanup
ds.flower.disconnect(flower)
DSI::datashield.logout(conns)
```

## Models

20 models across scikit-learn, PyTorch, and XGBoost. See the [dsFlower README](https://github.com/isglobal-brge/dsFlower) for the full list.

## Authors

- **David Sarrat González** — david.sarrat@isglobal.org
- **Juan R González** — juanr.gonzalez@isglobal.org

[Barcelona Institute for Global Health (ISGlobal)](https://www.isglobal.org/)
