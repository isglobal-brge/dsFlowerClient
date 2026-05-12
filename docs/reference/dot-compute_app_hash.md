# Compute SHA-256 hash of the Python files in the built app

Must match the server-side `.compute_template_hash()` algorithm.

## Usage

``` r
.compute_app_hash(app_dir, template_name)
```

## Arguments

- app_dir:

  Character; path to the built app directory.

- template_name:

  Character; template name (subdirectory name).

## Value

Character; hex-encoded SHA-256 hash.
