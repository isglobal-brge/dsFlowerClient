# Fetch a template from the server

Calls `flowerGetTemplateDS` on the first available server. Results are
cached per session so repeated runs with the same model don't re-fetch.

## Usage

``` r
.fetch_template(conns, template_name)
```

## Arguments

- conns:

  DSI connections object.

- template_name:

  Character; template name.

## Value

Named list mapping relative file paths to contents.
