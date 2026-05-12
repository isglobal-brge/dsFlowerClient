# Kill any process listening on a TCP port

Finds and kills orphaned flower-superlink processes that hold our ports
from crashed or abandoned R sessions. Only kills `flower-superlink`
processes to avoid accidentally killing unrelated services.

## Usage

``` r
.kill_orphan_on_port(port)
```

## Arguments

- port:

  Integer; the port number.

## Value

Invisible NULL.
