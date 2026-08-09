# Serialize a model spec to a base64 JSON string for transport as ONE TOML string value. Base64 is ASCII, so the JSON (which contains quotes) needs no TOML escaping; the node base64-decodes + json.loads it back into the spec it builds.

Serialize a model spec to a base64 JSON string for transport as ONE TOML
string value. Base64 is ASCII, so the JSON (which contains quotes) needs
no TOML escaping; the node base64-decodes + json.loads it back into the
spec it builds.

## Usage

``` r
.spec_to_b64(spec)
```
