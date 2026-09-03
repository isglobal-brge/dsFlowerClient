# Run a DataSHIELD aggregate without deparsing private transport arguments

DSI's progress display includes the complete call expression. Capability
tokens and tunnel payloads therefore require progress and raw
remote-error printing to be disabled for the duration of the call.

## Usage

``` r
.dsi_private_aggregate(conns, expr)
```
