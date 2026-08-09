# Write once to a non-blocking socket and return the bytes accepted by R

Base R's writeBin() intentionally does not report whether a non-blocking
output write succeeded. The small native shim exposes
R_WriteConnection's byte count so the relay never acknowledges bytes
that remain unwritten. It is compile-time pinned to R's connection ABI
version 1 and fails installation explicitly if a future R release
changes that private ABI.

## Usage

``` r
.tunnel_socket_write_some(socket, payload)
```
