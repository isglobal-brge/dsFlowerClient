# Detect all routable IPv4 addresses on the researcher's machine

Returns a prioritized list of IPs: OS-routed IP first (from UDP socket
trick), then VPN/tunnel interfaces (tun, utun, wg, tailscale), then
remaining LAN interfaces. Excludes loopback (127.x.x.x) and link-local
(169.254.x.x).

## Usage

``` r
.detect_all_ips()
```

## Value

Character vector of IPv4 address strings, ordered by priority.
