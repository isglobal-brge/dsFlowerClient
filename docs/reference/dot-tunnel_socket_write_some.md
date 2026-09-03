# Write a tunnel payload and return the bytes confirmed by R

The native shim exposes R_WriteConnection's byte count. R's socket
writer may wait even for a connection opened with `blocking = FALSE`,
and its timeout path can report zero after already writing an unknown
prefix. A zero count for a nonempty payload is therefore indeterminate:
this helper closes the socket and aborts so those bytes cannot be
retried on the same TCP stream. The shim is compile-time pinned to R's
connection ABI version 1.

## Usage

``` r
.tunnel_socket_write_some(socket, payload)
```
