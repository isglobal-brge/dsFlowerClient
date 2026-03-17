# Secure Connections: TLS Auto-Certificates

## Why TLS matters

The Flower gRPC channel between the SuperLink (your machine) and the
SuperNodes (Opal/Rock servers) carries model parameters, gradients, and
evaluation metrics. Without encryption, anyone on the network path
could:

- **Eavesdrop** on gradient updates (gradient inversion attacks can leak
  training data)
- **Redirect** a SuperNode to a rogue SuperLink that sends back a
  poisoned model
- **Tamper** with gradient updates in transit, degrading model quality

The DataSHIELD/Opal control channel is already HTTPS, but the Flower
gRPC training channel is separate. By default it runs unencrypted.
Setting `insecure = FALSE` enables TLS.

## How certificate generation works

When `insecure = FALSE`, dsFlowerClient auto-generates ephemeral TLS
certificates using the system `openssl` CLI. Let’s see what it produces:

``` r

library(dsFlowerClient)

cert_dir <- tempfile("dsflower_certs_")
certs <- dsFlowerClient:::.generate_tls_certs(cert_dir)

cat("Generated files:\n")
#> Generated files:
for (f in list.files(cert_dir)) {
  sz <- file.size(file.path(cert_dir, f))
  cat(sprintf("  %-15s  %d bytes\n", f, sz))
}
#>   ca.key           302 bytes
#>   ca.pem           444 bytes
#>   ca.srl           17 bytes
#>   san.cnf          95 bytes
#>   server.csr       367 bytes
#>   server.key       302 bytes
#>   server.pem       542 bytes
```

Seven files. Let’s look at each important one.

### The CA certificate (root of trust)

``` r

ca_lines <- readLines(certs$ca_cert_path)
cat(paste(ca_lines, collapse = "\n"), "\n")
#> -----BEGIN CERTIFICATE-----
#> MIIBGzCBwgIJALkbDCE+kXCOMAoGCCqGSM49BAMCMBYxFDASBgNVBAMMC2RzRmxv
#> d2VyLUNBMB4XDTI2MDMxNzE4MDk0OFoXDTI2MDMxODE4MDk0OFowFjEUMBIGA1UE
#> AwwLZHNGbG93ZXItQ0EwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAT/Q15pML8l
#> uleDbt/57hMveNAvikTArhH1FlqINaQWrS5i1EQrQDsmKzJu46BO+sP37yf1TDny
#> gW8eA/XMGGbNMAoGCCqGSM49BAMCA0gAMEUCIQDrAXGQ2zLIVqYz1v1V3C4b1sS7
#> F2A2KOrHZlknJOPvZwIgDBT8GWJc+/tHOCykVnFbXZmEzGmW/SjNT6xiF+56yBw=
#> -----END CERTIFICATE-----
```

This CA certificate gets distributed to every Opal. The CA private key
(`ca.key`) never leaves your machine:

``` r

cat("CA key permissions:", format(file.info(certs$ca_key_path)$mode), "\n")
#> CA key permissions: 600
cat("(0600 = owner read/write only)\n")
#> (0600 = owner read/write only)
```

### The SAN configuration

Subject Alternative Names determine which hostnames/IPs the server
certificate is valid for:

``` r

cat(readLines(file.path(cert_dir, "san.cnf")), sep = "\n")
#> [v3_req]
#> subjectAltName = DNS:localhost,DNS:host.docker.internal,IP:127.0.0.1,IP:192.168.1.89
```

| SAN                        | Purpose                        |
|:---------------------------|:-------------------------------|
| `DNS:localhost`            | SuperNode on the same machine  |
| `DNS:host.docker.internal` | SuperNode inside Docker        |
| `IP:127.0.0.1`             | Loopback connections           |
| `IP:<LAN IP>`              | SuperNode on the local network |

### The server certificate

``` r

srv_lines <- readLines(certs$srv_cert_path)
cat(paste(srv_lines, collapse = "\n"), "\n")
#> -----BEGIN CERTIFICATE-----
#> MIIBYjCCAQmgAwIBAgIJAL2C5R2bS5YmMAkGByqGSM49BAEwFjEUMBIGA1UEAwwL
#> ZHNGbG93ZXItQ0EwHhcNMjYwMzE3MTgwOTQ4WhcNMjYwMzE4MTgwOTQ4WjAdMRsw
#> GQYDVQQDDBJkc0Zsb3dlci1TdXBlckxpbmswWTATBgcqhkjOPQIBBggqhkjOPQMB
#> BwNCAAQq8d6WzU8uDEPIoe0QtfqBBZhyl1WIG/M/6eJ7uY/jPHJCTKICiGdKW2gt
#> 7JyIGwnveM8p1xHZ8RJNTUh+AM4vozowODA2BgNVHREELzAtgglsb2NhbGhvc3SC
#> FGhvc3QuZG9ja2VyLmludGVybmFshwR/AAABhwTAqAFZMAkGByqGSM49BAEDSAAw
#> RQIgThWlSj/OjNqX+5xfNDfAv/5AjNg0gkg2iLAjKaHtALkCIQDzVNa/TP2ZL0Bg
#> z9ZltK22NjLwUsl9ww7fzKNJOYEY+Q==
#> -----END CERTIFICATE-----
```

This is what the SuperLink presents during the TLS handshake. SuperNodes
verify it against the CA certificate.

### The generation sequence

All seven steps take roughly 5 milliseconds total (EC P-256 is fast):

    1. Sys.which("openssl")          → find the binary
    2. ecparam -name prime256v1      → probe EC support
    3. Write san.cnf                 → SAN configuration
    4. ecparam -genkey + req -x509   → CA key + self-signed cert (1-day expiry)
    5. ecparam -genkey + req + x509  → server key + CSR + sign with CA
    6. Sys.chmod("ca.key", "0600")   → restrict permissions
    7. readLines("ca.pem")           → load PEM for distribution

We use `-extfile san.cnf` instead of `-addext` because `-addext` is not
supported by LibreSSL (the default on macOS).

``` r

cat("OpenSSL version:", system2("openssl", "version", stdout = TRUE), "\n")
#> OpenSSL version: LibreSSL 3.3.6
```

## Live demo: insecure vs TLS

Let’s connect to three real Opal servers and compare both modes.

``` r

library(DSI)
library(DSOpal)

builder <- DSI::newDSLoginBuilder()
builder$append(server = "site_a", url = "https://localhost:8443",
               user = "administrator", password = "admin123",
               driver = "OpalDriver",
               options = "list(ssl_verifyhost=0, ssl_verifypeer=0)")
builder$append(server = "site_b", url = "https://localhost:8444",
               user = "administrator", password = "admin123",
               driver = "OpalDriver",
               options = "list(ssl_verifyhost=0, ssl_verifypeer=0)")
builder$append(server = "site_c", url = "https://localhost:8445",
               user = "administrator", password = "admin123",
               driver = "OpalDriver",
               options = "list(ssl_verifyhost=0, ssl_verifypeer=0)")

conns <- DSI::datashield.login(logins = builder$build(), assign = FALSE)
#> 
#> Logging into the collaborating servers
cat("Connected to:", paste(names(conns), collapse = ", "), "\n")
#> Connected to: site_a, site_b, site_c
```

### Insecure mode

``` r

quiet(ds.flower.nodes.init(conns, resource = "dsflower_test.flower_node"))
quiet(ds.flower.nodes.prepare(conns, target_column = "target",
                               feature_columns = c("f1", "f2", "f3", "f4", "f5")))

quiet(ds.flower.superlink.start(insecure = TRUE))
Sys.sleep(2)
status_insecure <- ds.flower.superlink.status()
cat(sprintf("Insecure mode:\n"))
#> Insecure mode:
cat(sprintf("  Running:       %s\n", status_insecure$running))
#>   Running:       TRUE
cat(sprintf("  Insecure:      %s\n", status_insecure$insecure))
#>   Insecure:      TRUE
cat(sprintf("  TLS cert:      %s\n",
    ifelse(is.null(status_insecure$ca_cert_pem), "none", "present")))
#>   TLS cert:      none
cat(sprintf("  Federation ID: %s\n", status_insecure$federation_id))
#>   Federation ID: fl-ieexombeu7fd

quiet(ds.flower.nodes.ensure(conns, symbol = "flower"))
#> Warning: site_c cannot reach SuperLink at host.docker.internal:9092
#> (There are some DataSHIELD errors, list them with datashield.errors())
#> Warning: Some nodes failed connectivity check: site_c. Consider providing
#> per-node superlink_address for those nodes.
Sys.sleep(10)

recipe <- ds.flower.recipe(
  task     = ds.flower.task.classification(),
  model    = ds.flower.model.sklearn_logreg(C = 1.0),
  strategy = ds.flower.strategy.fedavg(
    min_fit_clients = 3L, min_available_clients = 3L),
  num_rounds      = 3L,
  target_column   = "target",
  feature_columns = c("f1", "f2", "f3", "f4", "f5")
)

cat("\nTraining (insecure, 3 sites, 3 rounds)...\n\n")
#> 
#> Training (insecure, 3 sites, 3 rounds)...
run_insecure <- ds.flower.run.start(recipe, verbose = TRUE)
#> Flower App configuration warnings in '/private/var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/RtmpoIM0we/dsflower_app/sklearn_logreg/pyproject.toml':
#> - Recommended property "description" missing in [project]
#> - Recommended property "license" missing in [project]
#> 🎊 Successfully started run 10648262044936446603
#> INFO :      Start `flwr-serverapp` process
#> 🎊 Successfully installed sklearn_logreg to /var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/RtmpoIM0we/dsflower_superlink/apps/dsflower.sklearn_logreg.0.1.0.9921d28d.
#> INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
#> INFO :      
#> INFO :      [INIT]
#> INFO :      Requesting initial parameters from one random client
#> INFO :      Received initial parameters from one random client
#> INFO :      Starting evaluation of initial global parameters
#> INFO :      Evaluation returned no results (`None`)
#> INFO :      
#> INFO :      [ROUND 1]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> WARNING :   No fit_metrics_aggregation_fn provided
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> WARNING :   No evaluate_metrics_aggregation_fn provided
#> INFO :      
#> INFO :      [ROUND 2]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 3]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [SUMMARY]
#> INFO :      Run finished 3 round(s) in 48.23s
#> INFO :          History (loss, distributed):
#> INFO :              round 1: 0.1410575314912668
#> INFO :              round 2: 0.14103128152699967
#> INFO :              round 3: 0.14103127519907777
#> INFO :      
#> INFO :
#> INFO :      Starting logstream for run_id `10648262044936446603`
cat(sprintf("\nExit status: %s\n", run_insecure$status))
#> 
#> Exit status: 0

quiet(ds.flower.nodes.cleanup(conns, symbol = "flower"))
quiet(ds.flower.superlink.stop())
```

### TLS mode

Now the same thing with encryption. Only one line changes:
`insecure = FALSE`.

``` r

quiet(ds.flower.nodes.init(conns, resource = "dsflower_test.flower_node"))
quiet(ds.flower.nodes.prepare(conns, target_column = "target",
                               feature_columns = c("f1", "f2", "f3", "f4", "f5")))

quiet(ds.flower.superlink.start(insecure = FALSE))
Sys.sleep(2)
status_tls <- ds.flower.superlink.status()
cat(sprintf("TLS mode:\n"))
#> TLS mode:
cat(sprintf("  Running:       %s\n", status_tls$running))
#>   Running:       TRUE
cat(sprintf("  Insecure:      %s\n", status_tls$insecure))
#>   Insecure:      FALSE
cat(sprintf("  TLS cert:      %s\n",
    ifelse(is.null(status_tls$ca_cert_pem), "none",
           paste0(substr(status_tls$ca_cert_pem, 1, 27), "... (",
                  nchar(status_tls$ca_cert_pem), " chars)"))))
#>   TLS cert:      -----BEGIN CERTIFICATE-----... (443 chars)
cat(sprintf("  Federation ID: %s\n", status_tls$federation_id))
#>   Federation ID: fl-g5fivj57qv4a

quiet(ds.flower.nodes.ensure(conns, symbol = "flower"))
#> Warning: site_c cannot reach SuperLink at host.docker.internal:9092
#> (There are some DataSHIELD errors, list them with datashield.errors())
#> Warning: Some nodes failed connectivity check: site_c. Consider providing
#> per-node superlink_address for those nodes.
Sys.sleep(10)

cat("\nTraining (TLS encrypted, 3 sites, 3 rounds)...\n\n")
#> 
#> Training (TLS encrypted, 3 sites, 3 rounds)...
run_tls <- ds.flower.run.start(recipe, verbose = TRUE)
#> Flower App configuration warnings in '/private/var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/RtmpoIM0we/dsflower_app/sklearn_logreg/pyproject.toml':
#> - Recommended property "description" missing in [project]
#> - Recommended property "license" missing in [project]
#> 🎊 Successfully started run 5550878570024827726
#> INFO :      Start `flwr-serverapp` process
#> 🎊 Successfully installed sklearn_logreg to /var/folders/tn/qg45ss_91k375mrb66zqhx_m0000gn/T/RtmpoIM0we/dsflower_superlink/apps/dsflower.sklearn_logreg.0.1.0.9921d28d.
#> INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
#> INFO :      
#> INFO :      [INIT]
#> INFO :      Requesting initial parameters from one random client
#> INFO :      Received initial parameters from one random client
#> INFO :      Starting evaluation of initial global parameters
#> INFO :      Evaluation returned no results (`None`)
#> INFO :      
#> INFO :      [ROUND 1]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> WARNING :   No fit_metrics_aggregation_fn provided
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> WARNING :   No evaluate_metrics_aggregation_fn provided
#> INFO :      
#> INFO :      [ROUND 2]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [ROUND 3]
#> INFO :      configure_fit: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_fit: received 3 results and 0 failures
#> INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
#> INFO :      aggregate_evaluate: received 3 results and 0 failures
#> INFO :      
#> INFO :      [SUMMARY]
#> INFO :      Run finished 3 round(s) in 51.22s
#> INFO :          History (loss, distributed):
#> INFO :              round 1: 0.1410575314912668
#> INFO :              round 2: 0.14103128152699967
#> INFO :              round 3: 0.14103127519907774
#> INFO :      
#> INFO :
#> INFO :      Starting logstream for run_id `5550878570024827726`
cat(sprintf("\nExit status: %s\n", run_tls$status))
#> 
#> Exit status: 0

quiet(ds.flower.nodes.cleanup(conns, symbol = "flower"))
quiet(ds.flower.superlink.stop())
```

The loss values are comparable. TLS is transparent to the training
protocol: the only difference is that all gRPC traffic was encrypted.

## How the CA cert reaches the Opals

The CA certificate travels through the DataSHIELD HTTPS channel (which
is already secure). Here is the encoding chain:

    Your machine                        Opal/Rock
    -----------                         ---------

    1. ca_cert_pem (string, ~500 bytes)
       |
    2. .ds_encode(list(pem = ca_cert_pem))
       → JSON → Base64 → URL-safe → "B64:..." prefix
       |
    3. flowerEnsureSuperNodeDS(         ──[HTTPS]──>  4. .ds_arg() decodes
          symbol, address,                                → Base64 → JSON
          fed_id, "B64:eyJw...")                          → list(pem = "-----BEGIN...")
                                                      |
                                                   5. writeLines(pem, "ca.pem")
                                                      |
                                                   6. flower-supernode
                                                        --root-certificates ca.pem

## Certificate lifecycle

| Event | What happens |
|:---|:---|
| `superlink.start(insecure=FALSE)` | Certs generated in temp dir (~5ms) |
| `nodes.ensure()` | CA public cert sent to each Opal via HTTPS |
| SuperNode connects | gRPC TLS handshake verifies server cert |
| Training runs | All gRPC traffic encrypted |
| `superlink.stop()` | Entire temp dir deleted (certs included) |
| 24 hours | Certs expire (safety net if stop was not called) |

## Known limitations

- **No `extra_sans` in the public API**:
  [`.generate_tls_certs()`](../reference/dot-generate_tls_certs.md)
  accepts extra SANs but `superlink.start()` does not expose it yet.
- **No certificate renewal**: certs expire after 24 hours. Very long
  training runs may need manually generated certificates.
- **Opal method registration**: reinstalling dsFlower on Rock requires
  re-registering methods through Opal’s admin UI.
