# Function reference

Privacy policy is not a client specification. It is selected and enforced by
each `dsFlower` node.

## High-level submission

- [`ds.flower.fit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.fit.html)
- [`ds.flower.submit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.submit.html)
- [`ds.flower.hook.run()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.hook.run.html)

## Node and SuperLink lifecycle

- [`ds.flower.nodes.init()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.init.html)
- [`ds.flower.nodes.prepare()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.prepare.html)
- [`ds.flower.nodes.ensure()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.ensure.html)
- [`ds.flower.nodes.cleanup()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.nodes.cleanup.html)
- [`ds.flower.superlink.start()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.start.html)
- [`ds.flower.superlink.status()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.status.html)
- [`ds.flower.superlink.stop()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.superlink.stop.html)

## Declarative specifications

- [`ds.flower.model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.html)
- [`ds.flower.strategy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.html)
- [`ds.flower.task()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.task.html)
- [`ds.flower.recipe()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.recipe.html)

## Compatibility diagnostics

- [`ds.flower.metrics()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.metrics.html)
- [`ds.flower.log()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.log.html)

Hardened nodes return empty node metrics and logs. These functions remain for
wire compatibility; the intended output is the DP global-model artifact.
