# Generate ephemeral TLS certificates for SuperLink

Creates a CA and server certificate using EC P-256 via the system
openssl CLI. SANs are auto-populated with localhost,
host.docker.internal, 127.0.0.1, and the detected local IP.

## Usage

``` r
.generate_tls_certs(cert_dir, extra_sans = NULL)
```

## Arguments

- cert_dir:

  Character; directory to write certificate files.

- extra_sans:

  Character vector or NULL; additional SANs to include.

## Value

A named list with ca_cert_path, ca_key_path, srv_cert_path,
srv_key_path, and ca_cert_pem.
