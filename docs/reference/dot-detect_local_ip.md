# Detect the researcher's routable local IP address

Opens a UDP socket toward a public DNS server (no data is sent) to let
the OS routing table choose the correct outgoing interface. Falls back
to `hostname -I` (Linux) and `ipconfig getifaddr` (macOS) if the socket
approach fails.

## Usage

``` r
.detect_local_ip()
```

## Value

Character; an IPv4 address string.
