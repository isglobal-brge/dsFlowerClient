# Upload the researcher's training package to the nodes (zip + chunked push + verify + exfiltration scan + install). Returns the upload token + package name.

Upload the researcher's training package to the nodes (zip + chunked
push + verify + exfiltration scan + install). Returns the upload token +
package name.

## Usage

``` r
.upload_user_module(conns, user_pkg_dir, chunk_bytes = 262144L)
```
