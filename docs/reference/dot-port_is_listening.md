# Check if something is accepting connections on a local TCP port

Probes the port with a native R socket (`socketConnection`) instead of
shelling out to `lsof`/`netstat`. Opening a socket never forks, so this
is safe to call from an R process with live curl (DataSHIELD) threads –
unlike `system2`, which forks and can segfault mid-fork on macOS. Works
uniformly for insecure and TLS SuperLinks (a plain TCP connect succeeds
in both cases).

## Usage

``` r
.port_is_listening(port)
```

## Arguments

- port:

  Integer; port number.

## Value

Logical; TRUE if a server is accepting connections on the port.
