"""Flower ServerApp for Federated SGD Classifier."""

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def server_fn(context: Context) -> ServerAppComponents:
    """Configure the server."""
    cfg = context.run_config

    num_rounds = int(cfg.get("num-server-rounds", 5))
    fraction_fit = float(cfg.get("strategy-fraction_fit", 1.0))
    fraction_evaluate = float(cfg.get("strategy-fraction_evaluate", 1.0))
    min_fit_clients = int(cfg.get("strategy-min_fit_clients", 2))
    min_available_clients = int(cfg.get("strategy-min_available_clients", 2))

    # No initial_parameters — clients provide initial parameters
    # from their local data, which ensures correct feature dimensions.
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
