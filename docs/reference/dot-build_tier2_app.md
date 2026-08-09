# Build the Hook runner app dir (bundled dsflower_runner + the user package + pyproject). The user package is included so the researcher-side ServerApp can call initial_arrays(); on the node it is still hash-verified by the integrity hook against the node-computed upload hash (so it cannot be tampered with).

Build the Hook runner app dir (bundled dsflower_runner + the user
package + pyproject). The user package is included so the
researcher-side ServerApp can call initial_arrays(); on the node it is
still hash-verified by the integrity hook against the node-computed
upload hash (so it cannot be tampered with).

## Usage

``` r
.build_tier2_app(
  user_module,
  user_pkg_dir,
  n_features,
  results_dir,
  n_nodes,
  num_rounds,
  task,
  num_classes,
  app_params_b64,
  app_params_sha256
)
```
