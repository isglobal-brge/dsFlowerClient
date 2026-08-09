# Is this PID a live Flower SuperLink/SuperExec process? (cross-platform)

Verifies identity via the ps command line before we ever signal a PID,
so a recycled PID – some unrelated process now holding a dead
SuperLink's old PID – is never killed. Restores the safety the old lsof
reaper had, which matched `ps -o command=` before killing.

## Usage

``` r
.is_superlink_pid(pid)
```
