# Parse interface IPs from system commands

Uses `ifconfig` (macOS/BSD) or `ip addr` (Linux) to list all IPv4
addresses. Returns them ordered: VPN/tunnel interfaces first (tun, utun,
wg, tailscale, ts), then physical interfaces.

## Usage

``` r
.parse_interface_ips()
```

## Value

Character vector of IPv4 addresses, VPN-first ordering.
