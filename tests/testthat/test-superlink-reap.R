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

test_that("SuperLink PID file round-trips and reaps a live orphan", {
  skip_on_os("windows")  # uses `sleep` to stand in for a SuperLink

  tmp <- withr::local_tempdir()
  testthat::local_mocked_bindings(.client_venv_root = function() tmp)

  proc <- processx::process$new("sleep", "300")
  on.exit(if (proc$is_alive()) proc$kill(), add = TRUE)
  pid <- proc$get_pid()

  info <- list(pid = pid, fleet_port = 59123L,
               control_port = 59124L, serverappio_port = 59125L)
  dsFlowerClient:::.write_superlink_pid(info)

  pf <- dsFlowerClient:::.superlink_pid_path(59123L)
  expect_true(file.exists(pf))

  reaped <- dsFlowerClient:::.reap_orphan_superlink(59123L)
  expect_true(reaped)
  Sys.sleep(0.5)
  expect_false(proc$is_alive())   # the orphan SuperLink was killed
  expect_false(file.exists(pf))   # and its PID file removed
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
           ".kill_pid_hard", ".port_is_listening", ".write_superlink_pid")
  src <- paste(vapply(fns, function(f)
    paste(deparse(body(getFromNamespace(f, "dsFlowerClient"))), collapse = "\n"),
    character(1)), collapse = "\n")
  expect_false(grepl("system2|system\\(|lsof|netstat", src))
})
