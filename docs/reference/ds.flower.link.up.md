# Open the federation link

Starts a local Flower SuperLink and connects each node (SuperNode) to it
over the DataSHIELD channel, so training can run with no public host,
account or key required. Startup is all-or-nothing: if any node does not
publish a live, ready forwarder, every attempted node is torn down and
the call fails. Pass `verbose = TRUE` to see the internal transport
details. `options(dsflower.tunnel_port = ...)` accepts one shared port
or one port per node (positional or named), which permits multiple node
sessions in the same Rock/container without listener collisions.

## Usage

``` r
ds.flower.link.up(
  conns,
  symbol = "flower",
  verbose = getOption("dsflower.verbose", FALSE),
  allow_insecure_http = getOption("dsflower.dsi_allow_insecure_http", character())
)
```

## Arguments

- conns:

  DSI connections object.

- symbol:

  Character; handle symbol (default "flower").

- verbose:

  Logical; show internal transport details (SuperLink PID/ports).
  Defaults to `getOption("dsflower.verbose", FALSE)` – off, because this
  is internal plumbing the user does not need to see.

- allow_insecure_http:

  Character vector of exact connection names allowed to use plaintext
  HTTP. Empty by default. This exception does not provide transport
  security; use it only behind an independently trusted network.

## Value

Invisibly, the tunnel connection id.

## Details

Transport security is checked before any local service or node forwarder
is started. DSOpal requires HTTPS with retained certificate verification
enabled. A recognized DSMolgenisArmadillo connection is accepted
automatically when it exposes a valid HTTPS endpoint; its frontend or
reverse proxy remains responsible for certificate and hostname
validation. DSLite is accepted as an in-process transport. For an
unknown or unidentifiable connector, an operator who has independently
verified transport confidentiality, integrity, and peer authentication
must list the exact names in
`options(dsflower.dsi_tls_attested = c("site1", "site2"))`. Plaintext
HTTP is rejected by default, but an exact site can be allowed
deliberately with `allow_insecure_http` or
`options(dsflower.dsi_allow_insecure_http = "site1")`; this requires an
independent trusted network layer for confidentiality and integrity. The
exception only relaxes this package's preflight; it cannot override a
DSI connector that rejects remote HTTP while creating the connection,
and it cannot protect credentials already exchanged during connection
creation. Plaintext or unverifiable loopback connections are also
available for explicit local development with
`options(dsflower.dsi_allow_insecure_loopback = TRUE)`. This client-side
gate prevents accidental downgrade; production DataSHIELD frontends must
also reject plaintext independently.
