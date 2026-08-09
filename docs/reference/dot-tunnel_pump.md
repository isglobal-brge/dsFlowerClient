# Carry one batch of tunnel bytes (returns TRUE if the tunnel is active)

One fan-out flowerTunnelExchangeDS per cycle. The relay OWNS the byte
offsets (up_off = acked node-\>SuperLink bytes, down_sent = acked
SuperLink-\>node bytes), so a transient DSI failure loses nothing: the
next cycle re-requests the same ranges and the node applies them
idempotently. SuperLink replies are buffered per node and only dropped
once the node acks them, and the up offset only advances after the bytes
are confirmed written to the SuperLink socket. Negotiated chunks and
bounded per-node buffers apply TCP backpressure instead of accumulating
unbounded payloads in the R process. A single asynchronous fan-out call
(not sequential per-node pushes) lets DSI service the sites
concurrently. Its named expression list carries only the node-local
payload to each DataSHIELD connection.

## Usage

``` r
.tunnel_pump()
```
