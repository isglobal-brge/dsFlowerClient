# Registering dsFlower server methods

This walkthrough explains how a data custodian installs and registers
the server-side methods used by `dsFlowerClient`. It intentionally does
not reproduce private statistics or claim results from a particular live
federation.

## DataSHIELD registration is part of the trust boundary

A DataSHIELD analyst can invoke only methods explicitly registered by
the node administrator. Installing an R package does not by itself
expose all of its functions: the administrator also registers the
package’s declared assign and aggregate methods with Opal/Rock.

For `dsFlower`, the trusted node package performs data staging, privacy
contract validation, runner verification and SuperNode lifecycle
management. The client cannot register methods, set the node’s
epsilon/delta or authorize HookApps.

## Install and configure the node package

Install `dsFlower` in the compute profile used for the analysis:

``` r

remotes::install_github("isglobal-brge/dsFlower")
```

Before accepting runs, configure the node-owned per-training policy and
the persistent secret path. The following is an example; production
paths and values are custodian decisions:

``` r

options(
  default.dsflower.dp_per_training_epsilon = 1,
  default.dsflower.dp_per_training_delta = 1e-6,
  default.dsflower.node_secret_path =
    "/run/secrets/dsflower_node_key",
  default.dsflower.hook_enabled = FALSE
)
```

Only the dedicated 256-bit node secret persists across service sessions.
It must remain writable only by the service account and be protected
against cloning. There is no privacy database, historical counter or
admission decision based on earlier trainings. `datashield.seed` is not
used as the DP key.

## Register methods with Opal

With administrator credentials, `opalr` can register every method
declared by the package:

``` r

library(opalr)

opal <- opal.login(
  username = "administrator",
  password = Sys.getenv("OPAL_ADMIN_PW"),
  url = "https://opal.example.org"
)
dsadmin.set_package_methods(opal, "dsFlower")
opal.logout(opal)
```

[`dsadmin.get_methods()`](https://www.obiba.org/opalr/reference/dsadmin.get_methods.html)
can be used to audit the resulting assign and aggregate allowlists.
Registration should be repeated after upgrading the package when its
declared method set changes.

## Verify the installed security contract

After a researcher logs in and assigns a table to `D`, the public
capability endpoint reports the runner ABI/hash and non-secret policy
metadata:

``` r

library(DSI)

caps <- datashield.aggregate(
  conns,
  call("flowerGetCapabilitiesDS")
)

lapply(caps, function(x) x[c(
  "dsflower_version", "runner_abi", "runner_sha256",
  "dp_tracks", "declarative_model_ops", "declarative_losses",
  "aggregation_strategies",
  "privacy_accountant", "privacy_scope",
  "hook_enabled", "hook_sandbox_attested",
  "hook_resource_isolation_attested"
)])
```

`dsFlowerClient` performs the ABI and recursive runner-hash check
automatically before a submission. A mismatch fails early rather than
running a different implementation from the one the client expected.

Models are data-only specifications assembled from the advertised
declarative operations. The advertised operations describe what this
software version can execute; they are not a privacy permission list. No
researcher-supplied Python executes in the neural track.

The privacy-policy endpoint is read-only:

``` r

datashield.aggregate(conns, call("flowerPrivacyPolicyDS"))
```

It exposes the fixed server-owned per-training policy. This never gives
the analyst a privacy configuration knob.

## Preprocessing uses public bounds

Exact counts, sums and sums of squares are not used for normalization
and are not released. Supply bounds from public domain knowledge or the
study protocol instead:

``` r

fit <- ds.flower.fit(
  conns,
  symbol = "D",
  target = "malignant",
  features = c("mean_radius", "mean_texture"),
  model = "pytorch_logreg",
  feature_bounds = list(
    lower = c(0, 0),
    upper = c(50, 50)
  )
)
```

The bounds are pinned in the server-written manifest and the same
clipping and affine transform is stored for prediction. They must not be
estimated by querying the protected node.

## What registration does not authorize

Registering upload methods permits archive transport and validation; it
does not make uploaded code trusted. A HookApp runs only if the
custodian enabled it, attested the required Bubblewrap
filesystem/network sandbox, independently attested inherited cgroup v2
and writable-volume quotas, and enabled the release-global
minimum-duration timing envelope. This is defense in depth, not a formal
constant-time guarantee. If any gate is absent, the code is not executed
and the node returns an unchanged, data-independent model.

Node logs, node metrics, exact cohort weights and exact preprocessing
statistics are disabled. The intended egress is the differentially
private model update produced by the canonical runner.

## Recap

1.  Install `dsFlower` in the compute profile and configure the
    per-training policy and persistent node secret.
2.  Register only the package’s declared DataSHIELD methods as an
    administrator.
3.  Verify capabilities and runner identity after upgrades.
4.  Use public preprocessing bounds; do not reintroduce exact
    feature-statistic releases.
5.  Treat HookApp upload, execution and DP-SGD-level granularity as
    distinct concerns.
