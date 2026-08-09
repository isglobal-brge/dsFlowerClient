# Fail before upload/staging when a node publicly reports that arbitrary Hook execution is administratively disabled. Runtime sandbox self-tests still run node-side and fail closed; this preflight removes the common silent no-op UX.

Fail before upload/staging when a node publicly reports that arbitrary
Hook execution is administratively disabled. Runtime sandbox self-tests
still run node-side and fail closed; this preflight removes the common
silent no-op UX.

## Usage

``` r
.assert_hook_execution_configured(capabilities, conns)
```
