# Package load hook

Checks that the client Python venv is healthy. If not, prints a message
guiding the researcher to reinstall or run ensure manually.

## Usage

``` r
.onLoad(libname, pkgname)
```

## Arguments

- libname:

  Library name.

- pkgname:

  Package name.
