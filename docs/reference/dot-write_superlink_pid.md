# Record a running SuperLink so a later session can reap it without lsof

Written for BOTH interactive and detached SuperLinks. The old code
persisted state only when detached, so an interactive crash could only
be recovered by an lsof port scan – the very call that segfaulted.

## Usage

``` r
.write_superlink_pid(info)
```

## Arguments

- info:

  List; the SuperLink info (needs pid + the three ports).

## Value

Invisible NULL.
