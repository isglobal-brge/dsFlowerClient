# Upload and hash-verify a HookApp candidate archive

Builds `app_dir` into a FAB, pushes it to every node in `conns` in
idempotent base64 chunks over DataSHIELD, then installs it (the node
verifies the SHA-256 and applies the Hook/Tier-2 static scan). This is
the low-level HookApp store, not the path used to submit dsFlower's
canonical declarative runner FAB to the SuperLink. Uploading does not
authorize execution or turn arbitrary code into a per-sample DP
computation. A HookApp is separately hash-pinned and runs only when
every node-side execution gate holds.

## Usage

``` r
ds.flower.app.upload(conns, app_dir, chunk_bytes = 262144L, verbose = TRUE)
```

## Arguments

- conns:

  DSI connections object.

- app_dir:

  Character; path to the Flower app directory to bundle.

- chunk_bytes:

  Integer; bytes per push (default 256 KiB; maximum 512 KiB).

- verbose:

  Logical; print progress (default TRUE).

## Value

A `dsflower_app` object: list(token, sha256, size).
