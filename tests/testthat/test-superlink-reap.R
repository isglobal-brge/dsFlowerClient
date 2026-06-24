# Tests for R/superlink.R — fork-free orphan reaping.
#
# These exercise the SuperLink reaper that replaced the old system2("lsof"/"ps")
# port scan. The scan forked, and forking an R process with live curl
# (DataSHIELD) threads can segfault mid-fork on macOS — crashing the very
# cleanup meant to prevent orphans. The replacement uses native sockets +
# tools::pskill (the kill(2) syscall), which never fork.

test_that(".port_is_listening detects a listening socket (no subprocess)", {
  port <- NULL; srv <- NULL
  for (p in sample(20000:60000, 30)) {
    srv <- tryCatch(serverSocket(p), error = function(e) NULL)
    if (!is.null(srv)) { port <- p; break }
  }
  skip_if(is.null(port), "could not bind a test port")
  on.exit(close(srv), add = TRUE)

  # A listening server socket completes the TCP handshake in the kernel, so a
  # connect probe succeeds even before the app calls accept().
  expect_true(dsFlowerClient:::.port_is_listening(port))
})

test_that(".port_is_listening is FALSE for a free port", {
  port <- NULL
  for (p in sample(20000:60000, 30)) {
    s <- tryCatch(serverSocket(p), error = function(e) NULL)
    if (!is.null(s)) { close(s); port <- p; break }   # claim then release
  }
  skip_if(is.null(port), "could not find a free test port")
  expect_false(dsFlowerClient:::.port_is_listening(port))
})

test_that("PID file round-trips, and a non-SuperLink PID is never killed", {
  skip_on_os("windows")
  skip_if_not_installed("ps")
  tmp <- withr::local_tempdir()
  testthat::local_mocked_bindings(.client_venv_root = function() tmp)

  # A plain sleep stands in for a recycled PID: its number is in the PID file,
  # but it is NOT a flower-super process, so the reaper must leave it alone.
  proc <- processx::process$new("sleep", "120")
  on.exit(if (proc$is_alive()) proc$kill(), add = TRUE)
  pid <- proc$get_pid()

  dsFlowerClient:::.write_superlink_pid(
    list(pid = pid, fleet_port = 59123L,
         control_port = 59124L, serverappio_port = 59125L))
  pf <- dsFlowerClient:::.superlink_pid_path(59123L)
  expect_true(file.exists(pf))
  expect_equal(as.integer(readLines(pf)[1]), pid)        # write/read round-trip

  reaped <- dsFlowerClient:::.reap_orphan_superlink(
    59123L, ports = c(59123L, 59124L, 59125L))
  expect_false(reaped)               # not a flower-super -> not reaped
  Sys.sleep(0.3)
  expect_true(proc$is_alive())       # SAFETY: the unrelated process survived
  expect_false(file.exists(pf))      # but the stale PID file was cleared
})

test_that(".reap_orphan_superlink kills a confirmed SuperLink PID (kill path)", {
  skip_on_os("windows")
  tmp <- withr::local_tempdir()
  testthat::local_mocked_bindings(.client_venv_root = function() tmp)

  proc <- processx::process$new("sleep", "120")
  on.exit(if (proc$is_alive()) proc$kill(), add = TRUE)
  pid <- proc$get_pid()
  dsFlowerClient:::.write_superlink_pid(
    list(pid = pid, fleet_port = 59133L,
         control_port = 59134L, serverappio_port = 59135L))

  # Treat this PID as a confirmed Flower process so the kill machinery runs
  # (real flower-superlink identity is exercised by the e2e proofs).
  testthat::local_mocked_bindings(.is_superlink_pid = function(p) identical(as.integer(p), pid))
  reaped <- dsFlowerClient:::.reap_orphan_superlink(59133L, ports = 59133L)
  expect_true(reaped)
  Sys.sleep(0.5)
  expect_false(proc$is_alive())      # kill path executed
})

test_that(".pid_is_alive_local is correct and never kills the probed process", {
  skip_on_os("windows")
  expect_true(dsFlowerClient:::.pid_is_alive_local(Sys.getpid()))
  expect_false(dsFlowerClient:::.pid_is_alive_local(999999L))
  expect_false(dsFlowerClient:::.pid_is_alive_local(NA_integer_))

  proc <- processx::process$new("sleep", "60")
  on.exit(if (proc$is_alive()) proc$kill(), add = TRUE)
  # Probe twice. The point is the cross-platform contract: on Windows the old
  # pskill(0) would TerminateProcess here -- the ps-based check must not.
  expect_true(dsFlowerClient:::.pid_is_alive_local(proc$get_pid()))
  expect_true(dsFlowerClient:::.pid_is_alive_local(proc$get_pid()))
  expect_true(proc$is_alive())
})

test_that(".is_superlink_pid is FALSE for non-Flower processes (no false kills)", {
  skip_if_not_installed("ps")
  expect_false(dsFlowerClient:::.is_superlink_pid(Sys.getpid()))   # R itself
  skip_on_os("windows")
  proc <- processx::process$new("sleep", "60")
  on.exit(if (proc$is_alive()) proc$kill(), add = TRUE)
  expect_false(dsFlowerClient:::.is_superlink_pid(proc$get_pid()))
})

test_that(".reap_orphan_superlink is a quiet no-op when there is no live orphan", {
  tmp <- withr::local_tempdir()
  testthat::local_mocked_bindings(.client_venv_root = function() tmp)

  # No PID file at all -> FALSE, no error.
  expect_false(dsFlowerClient:::.reap_orphan_superlink(59222L))

  # PID file pointing at a dead PID -> file removed, returns FALSE, never signals.
  dsFlowerClient:::.write_superlink_pid(
    list(pid = 999999L, fleet_port = 59222L,
         control_port = 0L, serverappio_port = 0L))
  pf <- dsFlowerClient:::.superlink_pid_path(59222L)
  expect_true(file.exists(pf))
  expect_false(dsFlowerClient:::.reap_orphan_superlink(59222L))
  expect_false(file.exists(pf))
})

test_that(".discover_superlink_orphans never matches an unrelated process", {
  skip_on_os("windows")
  skip_if_not_installed("ps")
  # A plain sleep is not a Flower process and is not on a SuperLink port: the
  # scan must never return it (it only ever reaps flower-super* on OUR ports).
  proc <- processx::process$new("sleep", "60")
  on.exit(if (proc$is_alive()) proc$kill(), add = TRUE)
  hits <- dsFlowerClient:::.discover_superlink_orphans(c(59991L))
  expect_false(proc$get_pid() %in% hits)
})

test_that("the reaper never shells out to a forking subprocess", {
  # The fork-after-threads segfault came from system2()/lsof/netstat shell-outs.
  # Fork-free primitives only: native sockets, tools::pskill, and the ps package
  # (libproc / /proc at the C level -- NOT the `ps` shell command).
  fns <- c(".reap_orphan_superlink", ".discover_superlink_orphans",
           ".is_superlink_pid", ".kill_pid_hard", ".pid_is_alive_local",
           ".port_is_listening", ".write_superlink_pid")
  src <- paste(vapply(fns, function(f)
    paste(deparse(body(getFromNamespace(f, "dsFlowerClient"))), collapse = "\n"),
    character(1)), collapse = "\n")
  expect_false(grepl("system2|system\\(|lsof|netstat", src))
})
