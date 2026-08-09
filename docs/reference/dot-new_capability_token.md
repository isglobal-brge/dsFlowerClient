# Generate a 128-bit capability token with a protocol-specific prefix

Uses Python's operating-system-backed `secrets` generator. These tokens
authorize access to transient node-side resources, so R's statistical
PRNG is not an appropriate source even when collision probability would
be low.

## Usage

``` r
.new_capability_token(prefix)
```

## Arguments

- prefix:

  One of the protocol-owned prefixes `dsf`, `app`, or `usr`.

## Value

Character scalar of the form `prefix_[0-9a-f]{32}`.
