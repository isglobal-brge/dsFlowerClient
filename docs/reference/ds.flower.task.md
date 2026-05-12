# Create a task spec by name

Convenience wrapper around the concrete `ds.flower.task.*` constructors.
Existing `dsflower_task` objects are returned unchanged.

## Usage

``` r
ds.flower.task(name = "classification")
```

## Arguments

- name:

  Character task name or a `dsflower_task` object.

## Value

A `dsflower_task` object.
