# Package unload hook

Automatically stops the SuperLink when the package is unloaded or the R
session ends. Prevents orphaned flower-superlink processes.

## Usage

``` r
.onUnload(libpath)
```

## Arguments

- libpath:

  Library path.
