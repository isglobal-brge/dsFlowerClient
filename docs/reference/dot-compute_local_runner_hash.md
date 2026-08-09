# Compute the canonical hash of the bundled runner

This must remain byte-identical to dsFlower's .compute_harness_hash():
radix-sorted relative path, newline, content, NUL; compiled files
excluded.

## Usage

``` r
.compute_local_runner_hash(pkg_dir = .runner_skeleton_dir())
```
