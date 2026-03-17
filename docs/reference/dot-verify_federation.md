# Verify all nodes joined the same federation

Compares the `federation_id` reported by each node against the expected
value from the local SuperLink. Warns if any mismatch is found.

## Usage

``` r
.verify_federation(results, expected_fed_id)
```

## Arguments

- results:

  Named list of per-node status results.

- expected_fed_id:

  Character or NULL; expected federation ID.

## Value

Invisible NULL. Emits warnings on mismatches.
