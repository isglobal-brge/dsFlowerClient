"""In-memory Optuna ask/tell bridge for local R objectives."""

from __future__ import annotations

import json
import math
import sys
from typing import Any, NoReturn

import optuna


_PROTOCOL = "dsflower-local-hpo-v1"
_OPTUNA_VERSION = "4.8.0"


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _parse_line(line: str) -> Any:
    if not line:
        _fail("protocol input ended unexpectedly")

    def reject_constant(value: str) -> NoReturn:
        _fail(f"non-finite JSON number: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                _fail(f"duplicate JSON member: {key}")
            value[key] = item
        return value

    return json.loads(
        line, parse_constant=reject_constant, object_pairs_hook=unique_object
    )


def _exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(f"{label} must contain exactly: {', '.join(sorted(keys))}")
    return value


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    if value < minimum or value > maximum:
        _fail(f"{label} is outside its supported integer range")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} must be finite")
    return number


def _step_fits(lower: float, upper: float, step: float) -> bool:
    ratio = (upper - lower) / step
    return math.isfinite(ratio) and abs(ratio - round(ratio)) <= (
        1.0e-10 * max(1.0, abs(ratio))
    )


def _validate_space(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        _fail("space must be a non-empty object")
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(value):
        if not isinstance(name, str) or not name:
            _fail("parameter names must be non-empty strings")
        dimension = value[name]
        if not isinstance(dimension, dict):
            _fail(f"space.{name} must be an object")
        kind = dimension.get("type")
        if kind == "float":
            dimension = _exact_keys(
                dimension, {"type", "lower", "upper", "log", "step"},
                f"space.{name}",
            )
            lower = _number(dimension["lower"], f"space.{name}.lower")
            upper = _number(dimension["upper"], f"space.{name}.upper")
            log = dimension["log"]
            step = dimension["step"]
            if not isinstance(log, bool) or lower >= upper:
                _fail(f"space.{name} has invalid float bounds or log flag")
            if log and lower <= 0.0:
                _fail(f"space.{name} log bounds must be positive")
            if step is not None:
                step = _number(step, f"space.{name}.step")
                if log or step <= 0.0 or not _step_fits(lower, upper, step):
                    _fail(f"space.{name} has an invalid float step")
            result[name] = {
                "type": kind, "lower": lower, "upper": upper,
                "log": log, "step": step,
            }
        elif kind == "integer":
            dimension = _exact_keys(
                dimension, {"type", "lower", "upper", "log", "step"},
                f"space.{name}",
            )
            lower = _integer(
                dimension["lower"], f"space.{name}.lower",
                minimum=-2_147_483_647, maximum=2_147_483_647,
            )
            upper = _integer(
                dimension["upper"], f"space.{name}.upper",
                minimum=-2_147_483_647, maximum=2_147_483_647,
            )
            step = _integer(
                dimension["step"], f"space.{name}.step",
                minimum=1, maximum=2_147_483_647,
            )
            log = dimension["log"]
            if not isinstance(log, bool) or lower >= upper:
                _fail(f"space.{name} has invalid integer bounds or log flag")
            if log and (lower < 1 or step != 1):
                _fail(f"space.{name} log integers require lower >= 1 and step 1")
            if (upper - lower) % step != 0:
                _fail(f"space.{name} step must divide the integer range")
            result[name] = {
                "type": kind, "lower": lower, "upper": upper,
                "log": log, "step": step,
            }
        elif kind == "categorical":
            dimension = _exact_keys(
                dimension, {"type", "values"}, f"space.{name}"
            )
            choices = dimension["values"]
            if not isinstance(choices, list) or not choices:
                _fail(f"space.{name}.values must be a non-empty array")
            for choice in choices:
                if choice is None or isinstance(choice, (list, dict)):
                    _fail(f"space.{name}.values must contain JSON scalars")
                if isinstance(choice, float) and not math.isfinite(choice):
                    _fail(f"space.{name}.values must be finite")
            encoded = [
                json.dumps(choice, ensure_ascii=True, allow_nan=False,
                           separators=(",", ":"))
                for choice in choices
            ]
            if len(set(encoded)) != len(encoded):
                _fail(f"space.{name}.values must be unique")
            result[name] = {"type": kind, "values": choices}
        else:
            _fail(f"space.{name} has an unknown dimension type")
    return result


def _suggest(trial: optuna.trial.Trial,
             space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, dimension in space.items():
        kind = dimension["type"]
        if kind == "float":
            params[name] = trial.suggest_float(
                name, dimension["lower"], dimension["upper"],
                log=dimension["log"], step=dimension["step"],
            )
        elif kind == "integer":
            params[name] = trial.suggest_int(
                name, dimension["lower"], dimension["upper"],
                log=dimension["log"], step=dimension["step"],
            )
        else:
            params[name] = trial.suggest_categorical(name, dimension["values"])
    return params


def _emit(value: dict[str, Any]) -> None:
    print(
        json.dumps(value, ensure_ascii=True, allow_nan=False,
                   separators=(",", ":")),
        flush=True,
    )


def _run() -> None:
    request = _exact_keys(
        _parse_line(sys.stdin.readline()),
        {"protocol", "direction", "seed", "n_trials", "space"},
        "request",
    )
    if request["protocol"] != _PROTOCOL:
        _fail("unsupported protocol")
    if request["direction"] not in {"minimize", "maximize"}:
        _fail("direction must be minimize or maximize")
    seed = _integer(request["seed"], "seed", minimum=0, maximum=4_294_967_295)
    n_trials = _integer(
        request["n_trials"], "n_trials", minimum=1, maximum=1_000_000
    )
    space = _validate_space(request["space"])

    if optuna.__version__ != _OPTUNA_VERSION:
        _fail(f"this bridge requires Optuna {_OPTUNA_VERSION}")
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(
        direction=request["direction"],
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=None,
    )
    for _ in range(n_trials):
        trial = study.ask()
        params = _suggest(trial, space)
        _emit({"event": "trial", "number": trial.number, "params": params})
        reply = _exact_keys(
            _parse_line(sys.stdin.readline()),
            {"event", "number", "value"}, "trial reply",
        )
        if reply["event"] != "value" or reply["number"] != trial.number:
            _fail("trial reply does not match the outstanding trial")
        value = _number(reply["value"], "objective value")
        study.tell(trial, value)

    _emit({
        "event": "complete",
        "optuna_version": optuna.__version__,
    })


def main() -> int:
    try:
        _run()
    except Exception as exc:  # The R side receives one bounded protocol error.
        message = str(exc).replace("\r", " ").replace("\n", " ")[:512]
        _emit({"event": "error", "message": message or "backend error"})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
