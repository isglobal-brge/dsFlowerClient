# Run an openssl command with error checking

Run an openssl command with error checking

## Usage

``` r
.run_openssl(openssl_path, args, stdin = NULL)
```

## Arguments

- openssl_path:

  Character; path to the openssl binary.

- args:

  Character vector; arguments to pass.

- stdin:

  Character or NULL; optional stdin input.

## Value

Character vector of stdout lines (invisible).
