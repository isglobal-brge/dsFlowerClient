# Destroy session handles only after exact, per-node acknowledgement

Nodes where a target is already absent are successful no-ops. This makes
a repeated call retry only the targets retained after an earlier partial
failure. Destroy results are assigned to a separate temporary symbol so
a failed workspace removal cannot replace the opaque handle with `NULL`.
A handle symbol is never removed unless its destroy method has returned
an exact success callback for that same node. The acknowledgement symbol
is deterministic for each target, allowing the same API call to remove
an orphan left by a previously lost workspace-removal response.

## Usage

``` r
.dsi_destroy_session_exact(conns, symbol, imaging_symbol = NULL)
```
