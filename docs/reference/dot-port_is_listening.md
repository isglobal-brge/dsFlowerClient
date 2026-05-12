# Check if a port is being listened on

Uses `lsof` on macOS/Linux or `netstat` on Windows to check if any
process is listening on the given port. Does not require a TLS
handshake.

## Usage

``` r
.port_is_listening(port)
```

## Arguments

- port:

  Integer; port number.

## Value

Logical.
