# Module: SuperLink Lifecycle
# Manages the Flower SuperLink process on the researcher's machine.

# --- Orphan reaping (fork-free) ---
#
# The local SuperLink (and its flower-superexec child) must be reaped when a
# previous R session left them behind: a hard crash, a SIGKILL, or a researcher
# who quit without ds.flower.superlink.stop(). We do this WITHOUT shelling out.
# Every running SuperLink records its PID in a small file, and reaping signals
# that PID directly via tools::pskill() (the kill(2) syscall). The previous
# implementation scanned ports with system2("lsof"/"ps"), which forks; on macOS,
# forking an R process that already has live curl (DataSHIELD) threads can
# segfault mid-fork -- exactly during the cleanup meant to PREVENT orphans, so a
# flaky-network run would crash and leave the orphan it was trying to reap.
# Native sockets + pskill never fork, so the reaper can no longer crash. This
# mirrors the server-side SuperNode reaping in dsFlower (PID files + pskill).

#' Directory holding SuperLink PID files
#' @keywords internal
.superlink_pid_dir <- function() {
  d <- file.path(.client_venv_root(), "superlink")
  if (!dir.exists(d)) {
    tryCatch(dir.create(d, recursive = TRUE, showWarnings = FALSE),
             error = function(e) NULL)
  }
  d
}

#' Path to the PID file for a SuperLink, keyed by its fleet port
#' @keywords internal
.superlink_pid_path <- function(fleet_port) {
  file.path(.superlink_pid_dir(), paste0("superlink_", fleet_port, ".pid"))
}

#' Record a running SuperLink so a later session can reap it without lsof
#'
#' Written for BOTH interactive and detached SuperLinks. The old code persisted
#' state only when detached, so an interactive crash could only be recovered by
#' an lsof port scan -- the very call that segfaulted.
#'
#' @param info List; the SuperLink info (needs pid + the three ports).
#' @return Invisible NULL.
#' @keywords internal
.write_superlink_pid <- function(info) {
  tryCatch({
    pf <- .superlink_pid_path(info$fleet_port)
    # Direct write (no temp+rename): the file is single-owner and port-keyed, so
    # it is overwritten every session. Windows MoveFile cannot replace an
    # existing file, so a temp+rename would fail exactly when a stale PID file is
    # present. A torn read is harmless -- the reader gets NA and the ps scan in
    # .reap_orphan_superlink still finds the orphan.
    writeLines(as.character(c(info$pid, info$fleet_port,
                              info$control_port, info$serverappio_port,
                              format(Sys.time(), "%Y-%m-%dT%H:%M:%S"))), pf)
  }, error = function(e) NULL)
  invisible(NULL)
}

#' Remove a SuperLink PID file
#' @keywords internal
.remove_superlink_pid <- function(fleet_port) {
  if (is.null(fleet_port)) return(invisible(NULL))
  tryCatch({
    pf <- .superlink_pid_path(fleet_port)
    if (file.exists(pf)) unlink(pf)
  }, error = function(e) NULL)
  invisible(NULL)
}

#' Hard-stop a PID: SIGTERM, then SIGKILL if it lingers (fork-free)
#' @keywords internal
.kill_pid_hard <- function(pid) {
  if (is.na(pid) || !.pid_is_alive_local(pid)) return(invisible(FALSE))
  tools::pskill(pid, signal = 15L)               # SIGTERM
  Sys.sleep(1)
  if (.pid_is_alive_local(pid)) {
    tools::pskill(pid, signal = 9L)              # SIGKILL
    Sys.sleep(0.5)
  }
  invisible(TRUE)
}

#' Discover orphaned Flower SuperLink/SuperExec PIDs squatting given ports
#'
#' Fork-free process scan via the \pkg{ps} package (libproc on macOS, /proc on
#' Linux) -- the client analogue of the server's /proc SuperNode scan. Matches
#' only \code{flower-super*} processes whose command line references one of OUR
#' ports, so it reaps a stale SuperLink (or its superexec child) that has no PID
#' file -- e.g. one started before PID-file tracking, or whose file was lost --
#' and never touches an unrelated service.
#'
#' @param ports Integer vector; the SuperLink ports we are about to bind.
#' @return Integer vector of matching PIDs.
#' @keywords internal
.discover_superlink_orphans <- function(ports) {
  ports <- Filter(Negate(is.null), ports)
  if (length(ports) == 0L || !requireNamespace("ps", quietly = TRUE)) {
    return(integer(0))
  }
  port_pat <- paste0(":(", paste(unique(ports), collapse = "|"), ")(\\b|$)")
  pids <- tryCatch(ps::ps_pids(), error = function(e) integer(0))
  hits <- integer(0)
  for (pid in pids) {
    h <- tryCatch(ps::ps_handle(pid), error = function(e) NULL)
    if (is.null(h)) next
    cmd <- tryCatch(paste(ps::ps_cmdline(h), collapse = " "),
                    error = function(e) "")
    if (grepl("flower-super", cmd, fixed = TRUE) && grepl(port_pat, cmd)) {
      hits <- c(hits, pid)
    }
  }
  hits
}

#' Is this PID a live Flower SuperLink/SuperExec process? (cross-platform)
#'
#' Verifies identity via the \pkg{ps} command line before we ever signal a PID,
#' so a recycled PID -- some unrelated process now holding a dead SuperLink's old
#' PID -- is never killed. Restores the safety the old lsof reaper had, which
#' matched \code{ps -o command=} before killing.
#' @keywords internal
.is_superlink_pid <- function(pid) {
  if (is.null(pid) || is.na(pid) || !requireNamespace("ps", quietly = TRUE)) {
    return(FALSE)
  }
  tryCatch({
    h <- ps::ps_handle(as.integer(pid))
    ps::ps_is_running(h) &&
      grepl("flower-super", paste(ps::ps_cmdline(h), collapse = " "), fixed = TRUE)
  }, error = function(e) FALSE)
}

#' Reap an orphaned SuperLink left by a crashed or abandoned session
#'
#' Collects candidate PIDs two fork-free ways -- the PID file (fast, exact) and
#' a \pkg{ps} scan for any \code{flower-super*} still holding one of our ports
#' (catches pidfile-less orphans and the superexec child) -- then signals only
#' those it can re-confirm are Flower processes. Neither path shells out, so
#' neither can trip the macOS fork-after-threads segfault the old lsof/ps scan
#' could; the identity re-check means a recycled PID is never killed.
#'
#' @param fleet_port Integer; the SuperLink's fleet port (PID-file key).
#' @param ports Integer vector; all SuperLink ports to free (default:
#'   just \code{fleet_port}).
#' @return Invisible TRUE if anything was reaped, FALSE otherwise.
#' @keywords internal
.reap_orphan_superlink <- function(fleet_port, ports = fleet_port) {
  candidates <- integer(0)

  # The PID file: the SuperLink this package last started on this fleet port.
  pf <- .superlink_pid_path(fleet_port)
  if (file.exists(pf)) {
    pid <- suppressWarnings(as.integer(readLines(pf, warn = FALSE)[1]))
    if (length(pid) == 1L && !is.na(pid)) candidates <- pid
    unlink(pf)
  }

  # Plus any flower-super* still squatting our ports with no PID file (a
  # pre-tracking orphan, or a superexec child that outlived its parent).
  candidates <- unique(c(candidates, .discover_superlink_orphans(ports)))

  reaped <- FALSE
  for (pid in candidates) {
    # Re-confirm identity before signalling: never kill a recycled PID that some
    # unrelated process now owns.
    if (.is_superlink_pid(pid)) {
      message("  Reaping orphaned SuperLink (PID ", pid,
              ") left by a previous session")
      .kill_pid_hard(pid)
      reaped <- TRUE
    }
  }
  invisible(reaped)
}

# --- TLS certificate generation helpers ---

#' Run an openssl command with error checking
#'
#' @param openssl_path Character; path to the openssl binary.
#' @param args Character vector; arguments to pass.
#' @param stdin Character or NULL; optional stdin input.
#' @return Character vector of stdout lines (invisible).
#' @keywords internal
.run_openssl <- function(openssl_path, args, stdin = NULL) {
  result <- suppressWarnings(
    system2(openssl_path, args,
            stdout = TRUE, stderr = TRUE,
            input = stdin)
  )
  status <- attr(result, "status")
  if (!is.null(status) && status != 0L) {
    stop("openssl command failed (exit ", status, "): ",
         paste(c(openssl_path, args), collapse = " "), "\n",
         paste(result, collapse = "\n"),
         call. = FALSE)
  }
  invisible(result)
}

#' Generate ephemeral TLS certificates for SuperLink
#'
#' Creates a CA and server certificate using EC P-256 via the system openssl
#' CLI. SANs are auto-populated with localhost, host.docker.internal, 127.0.0.1,
#' and the detected local IP.
#'
#' @param cert_dir Character; directory to write certificate files.
#' @param extra_sans Character vector or NULL; additional SANs to include.
#' @return A named list with ca_cert_path, ca_key_path, srv_cert_path,
#'   srv_key_path, and ca_cert_pem.
#' @keywords internal
.generate_tls_certs <- function(cert_dir, extra_sans = NULL, cert_days = 1L) {
  openssl_path <- Sys.which("openssl")
  if (!nzchar(openssl_path)) {
    stop("openssl CLI not found on PATH. ",
         "Install OpenSSL to use TLS.",
         call. = FALSE)
  }

  # Probe EC support
  tryCatch(
    .run_openssl(openssl_path, c("ecparam", "-name", "prime256v1", "-check")),
    error = function(e) {
      stop("openssl does not support EC prime256v1: ", conditionMessage(e),
           call. = FALSE)
    }
  )

  dir.create(cert_dir, recursive = TRUE, showWarnings = FALSE)

  # Build SANs
  sans <- c("DNS:localhost", "DNS:host.docker.internal", "IP:127.0.0.1")
  local_ip <- tryCatch(.detect_local_ip(), error = function(e) NULL)
  if (!is.null(local_ip) && !local_ip %in% c("127.0.0.1")) {
    sans <- c(sans, paste0("IP:", local_ip))
  }
  if (!is.null(extra_sans)) {
    sans <- c(sans, extra_sans)
  }

  # Write SAN config file (LibreSSL compatible — no -addext)
  san_cnf_path <- file.path(cert_dir, "san.cnf")
  san_cnf <- paste0(
    "[v3_req]\n",
    "subjectAltName = ", paste(sans, collapse = ","), "\n"
  )
  writeLines(san_cnf, san_cnf_path)

  # File paths
  ca_key_path  <- file.path(cert_dir, "ca.key")
  ca_cert_path <- file.path(cert_dir, "ca.pem")
  srv_key_path <- file.path(cert_dir, "server.key")
  srv_csr_path <- file.path(cert_dir, "server.csr")
  srv_cert_path <- file.path(cert_dir, "server.pem")

  # 1. Generate CA key
  .run_openssl(openssl_path, c(
    "ecparam", "-genkey", "-name", "prime256v1",
    "-out", ca_key_path
  ))

  # 2. Generate CA certificate (self-signed, 1 day)
  .run_openssl(openssl_path, c(
    "req", "-new", "-x509", "-sha256",
    "-key", ca_key_path,
    "-out", ca_cert_path,
    "-days", as.character(cert_days),
    "-subj", "/CN=dsFlower-CA"
  ))

  # 3. Generate server key
  .run_openssl(openssl_path, c(
    "ecparam", "-genkey", "-name", "prime256v1",
    "-out", srv_key_path
  ))

  # 4. Generate server CSR
  .run_openssl(openssl_path, c(
    "req", "-new",
    "-key", srv_key_path,
    "-out", srv_csr_path,
    "-subj", "/CN=dsFlower-SuperLink"
  ))

  # 5. Sign server cert with CA, applying SANs
  .run_openssl(openssl_path, c(
    "x509", "-req", "-sha256",
    "-in", srv_csr_path,
    "-CA", ca_cert_path,
    "-CAkey", ca_key_path,
    "-CAcreateserial",
    "-out", srv_cert_path,
    "-days", as.character(cert_days),
    "-extfile", san_cnf_path,
    "-extensions", "v3_req"
  ))

  # 6. Restrict CA key permissions
  Sys.chmod(ca_key_path, "0600")

  # 7. Read CA cert PEM for distribution
  ca_cert_pem <- paste(readLines(ca_cert_path, warn = FALSE), collapse = "\n")

  list(
    ca_cert_path  = ca_cert_path,
    ca_key_path   = ca_key_path,
    srv_cert_path = srv_cert_path,
    srv_key_path  = srv_key_path,
    ca_cert_pem   = ca_cert_pem
  )
}

#' Start a Flower SuperLink
#'
#' Spawns a \code{flower-superlink} process. In detached mode, the process
#' survives R session exit.
#'
#' @param fleet_port Integer; port for the Fleet API (default 9092).
#' @param control_port Integer; port for the Control API (default 9093).
#' @param serverappio_port Integer; port for the ServerAppIO API (default 9091).
#' @param detached Logical; if TRUE, SuperLink runs as daemon (survives
#'   R session exit). Default FALSE for interactive use.
#' @param insecure Logical; explicitly allow plaintext local SuperLink transport.
#' @return Invisible list with process info.
#' @export
ds.flower.superlink.start <- function(fleet_port = 9092L,
                                       control_port = 9093L,
                                       serverappio_port = 9091L,
                                       detached = FALSE,
                                       insecure = getOption("dsflower.superlink_insecure", FALSE)) {
  insecure <- isTRUE(insecure)
  .require_flwr_cli()

  # Check if already running (in-session process)
  existing <- .dsflower_client_env$.superlink
  if (!is.null(existing)) {
    alive <- if (!is.null(existing$process)) {
      existing$process$is_alive()
    } else {
      .pid_is_alive_local(existing$pid)
    }
    if (alive) {
      message("SuperLink is already running (PID: ", existing$pid, ")")
      return(invisible(existing))
    }
  }

  # Check for existing detached SuperLink
  if (detached) {
    state <- .load_superlink_state()
    if (!is.null(state) && .pid_is_alive_local(state$pid) &&
        .port_is_listening(state$fleet_port)) {
      message("Attaching to existing detached SuperLink (PID: ", state$pid, ")")
      .dsflower_client_env$.superlink <- state
      return(invisible(state))
    }
    .clear_superlink_state()
  }

  # Reap any SuperLink a previous session left behind, fork-free: by PID file,
  # then by a ps scan for anything still squatting these exact ports.
  .reap_orphan_superlink(fleet_port,
                         ports = c(fleet_port, control_port, serverappio_port))

  # Persistent dir for detached, tempdir for interactive
  if (detached) {
    base_dir <- file.path(.client_venv_root(), "superlink")
    flwr_home <- file.path(base_dir, "flwr_home")
    cert_dir <- file.path(base_dir, "certs")
    log_path <- file.path(base_dir, "superlink.log")
    cert_days <- 30L
  } else {
    flwr_home <- file.path(tempdir(), "dsflower_superlink")
    cert_dir <- file.path(flwr_home, "certs")
    log_path <- file.path(flwr_home, "superlink.log")
    cert_days <- 1L
  }
  dir.create(flwr_home, recursive = TRUE, showWarnings = FALSE)

  # TLS certificates (skipped for the DSI tunnel, which runs insecure because the
  # bytes already travel inside the TLS DataSHIELD channel).
  tls_info <- if (insecure) NULL else .generate_tls_certs(cert_dir, cert_days = cert_days)

  fleet_type <- getOption("dsflower.fleet_api_type", "grpc-rere")

  # The DSI tunnel relay connects to the Fleet API on loopback, so bind there and
  # run insecure; otherwise expose TLS on all interfaces for a remote transport.
  bind <- if (insecure) "127.0.0.1:" else "0.0.0.0:"
  tls_args <- if (insecure) "--insecure"
              else c("--ssl-certfile",    tls_info$srv_cert_path,
                     "--ssl-keyfile",     tls_info$srv_key_path,
                     "--ssl-ca-certfile", tls_info$ca_cert_path)
  args <- c(
    tls_args,
    "--fleet-api-type", fleet_type,
    "--fleet-api-address", paste0(bind, fleet_port),
    "--control-api-address", paste0(bind, control_port),
    "--serverappio-api-address", paste0(bind, serverappio_port)
  )

  # Spawn -- detached processes survive R exit
  superlink_cmd <- .client_superlink_cmd()
  proc <- processx::process$new(
    command = superlink_cmd,
    args = args,
    stdout = log_path,
    stderr = "2>&1",
    cleanup = !detached,
    cleanup_tree = !detached,
    env = .client_venv_env(extra = c(FLWR_HOME = flwr_home))
  )

  # Write config.toml for flwr run
  config_toml <- paste0(
    "[superlink]\n",
    'default = "dsflower"\n\n',
    "[superlink.dsflower]\n",
    'address = "127.0.0.1:', control_port, '"\n',
    if (insecure) "insecure = true\n"
    else paste0('root-certificates = "', tls_info$ca_cert_path, '"\n')
  )
  writeLines(config_toml, file.path(flwr_home, "config.toml"))

  fleet_address   <- paste0("127.0.0.1:", fleet_port)
  control_address <- paste0("127.0.0.1:", control_port)

  federation_id <- paste0("fl-",
    paste(sample(c(letters, 0:9), 12, replace = TRUE), collapse = ""))

  info <- list(
    process          = proc,
    pid              = proc$get_pid(),
    fleet_address    = fleet_address,
    control_address  = control_address,
    fleet_port       = fleet_port,
    control_port     = control_port,
    serverappio_port = serverappio_port,
    flwr_home        = flwr_home,
    log_path         = log_path,
    federation_id    = federation_id,
    ca_cert_pem      = tls_info$ca_cert_pem,
    ca_cert_path     = tls_info$ca_cert_path,
    detached         = detached,
    started_at       = Sys.time()
  )

  .dsflower_client_env$.superlink <- info

  # Wait for ready
  .wait_superlink_ready(proc, fleet_port, log_path, timeout = 15)

  # Record this SuperLink's PID so a later session can reap it without lsof,
  # even after a hard crash (interactive runs persist nothing else).
  .write_superlink_pid(info)

  # Save state for detached reconnection
  if (detached) .save_superlink_state(info)

  message("SuperLink started",
          if (detached) " (detached)" else "", " (PID: ", info$pid, ")")
  message("  Fleet API (SuperNodes): ", fleet_address)
  message("  Control API (flwr run): ", control_address)
  if (detached) {
    message("  Mode: detached -- survives R session exit")
  }
  invisible(info)
}

#' Wait for SuperLink to be ready
#'
#' Verifies the process is alive and the fleet port is accepting connections
#' (via a native socket probe -- no subprocess, no TLS handshake needed).
#'
#' @param proc processx process object.
#' @param port Integer; port to check.
#' @param log_path Character; path to the log file (for error messages).
#' @param timeout Numeric; seconds to wait.
#' @keywords internal
.wait_superlink_ready <- function(proc, port, log_path, timeout = 15) {
  deadline <- Sys.time() + timeout

  while (Sys.time() < deadline) {
    if (!proc$is_alive()) {
      log_tail <- tryCatch(
        paste(utils::tail(readLines(log_path, warn = FALSE), 10), collapse = "\n"),
        error = function(e) "(no log)")
      stop("SuperLink process died during startup.\nLog:\n", log_tail,
           call. = FALSE)
    }

    # Check if the process is listening on the port
    if (.port_is_listening(port)) return(invisible(TRUE))
    Sys.sleep(0.5)
  }

  log_tail <- tryCatch(
    paste(utils::tail(readLines(log_path, warn = FALSE), 10), collapse = "\n"),
    error = function(e) "(no log)")
  stop("SuperLink did not become ready within ", timeout, " seconds.\nLog:\n",
       log_tail, call. = FALSE)
}

#' Check if something is accepting connections on a local TCP port
#'
#' Probes the port with a native R socket (\code{socketConnection}) instead of
#' shelling out to \code{lsof}/\code{netstat}. Opening a socket never forks, so
#' this is safe to call from an R process with live curl (DataSHIELD) threads --
#' unlike \code{system2}, which forks and can segfault mid-fork on macOS. Works
#' uniformly for insecure and TLS SuperLinks (a plain TCP connect succeeds in
#' both cases).
#'
#' @param port Integer; port number.
#' @return Logical; TRUE if a server is accepting connections on the port.
#' @keywords internal
.port_is_listening <- function(port) {
  con <- tryCatch(
    socketConnection(host = "127.0.0.1", port = port, server = FALSE,
                     blocking = TRUE, open = "r+", timeout = 1),
    error = function(e) NULL, warning = function(w) NULL)
  if (is.null(con)) return(FALSE)
  close(con)
  TRUE
}

#' Stop the Flower SuperLink
#'
#' Sends SIGTERM, waits, then SIGKILL if needed. Works for both
#' interactive and detached SuperLinks.
#'
#' @return Invisible TRUE.
#' @export
ds.flower.superlink.stop <- function() {
  info <- .dsflower_client_env$.superlink
  if (is.null(info)) {
    # Check for detached state file
    info <- .load_superlink_state()
    if (is.null(info)) {
      message("No SuperLink is running.")
      return(invisible(TRUE))
    }
  }

  # Kill via processx object if available, otherwise via PID. Use kill_tree so
  # the SuperLink's child flower-superexec (and any ServerApp) die too; killing
  # only the SuperLink would orphan the SuperExec, which keeps holding the
  # Fleet/Control ports and makes the next start fail.
  if (!is.null(info$process)) {
    proc <- info$process
    if (proc$is_alive()) {
      proc$signal(15L)
      proc$wait(timeout = 5000)
      if (proc$is_alive()) proc$kill_tree()
    }
  } else if (!is.null(info$pid) && .pid_is_alive_local(info$pid)) {
    tools::pskill(info$pid, signal = 15L)
    Sys.sleep(2)
    if (.pid_is_alive_local(info$pid)) {
      tools::pskill(info$pid, signal = 9L)
    }
  }

  # The in-session kill_tree() above already reaps the flower-superexec child;
  # drop this SuperLink's PID-file record so it is no longer seen as an orphan.
  # (A cross-session stop has no processx tree to walk, but a superexec whose
  # SuperLink just died loses its endpoint and exits on its own.)
  .remove_superlink_pid(info$fleet_port)

  # Cleanup directories
  if (!is.null(info$flwr_home) && dir.exists(info$flwr_home)) {
    unlink(info$flwr_home, recursive = TRUE)
  }
  # Cleanup detached superlink base dir (certs, logs, state)
  if (isTRUE(info$detached)) {
    base_dir <- file.path(.client_venv_root(), "superlink")
    if (dir.exists(base_dir)) unlink(base_dir, recursive = TRUE)
  }

  # Clear state file if detached
  .clear_superlink_state()

  .dsflower_client_env$.superlink <- NULL
  .dsf_msg("SuperLink stopped.")
  invisible(TRUE)
}

#' Get SuperLink status
#'
#' @return A named list with running, pid, fleet_address, control_address,
#'   ports, detached, started_at.
#' @export
ds.flower.superlink.status <- function() {
  info <- .dsflower_client_env$.superlink

  # Check detached state if no in-session info
  if (is.null(info)) {
    info <- .load_superlink_state()
  }

  if (is.null(info)) {
    return(list(
      running         = FALSE,
      pid             = NULL,
      fleet_address   = NULL,
      control_address = NULL,
      ports           = NULL,
      ca_cert_pem     = NULL,
      detached        = FALSE,
      started_at      = NULL
    ))
  }

  # A remote coordinator SuperLink has no local process/port to probe; we
  # trust it is up (the per-node egress preflight catches unreachability).
  running <- if (isTRUE(info$remote)) {
    TRUE
  } else if (!is.null(info$process)) {
    info$process$is_alive()
  } else {
    .pid_is_alive_local(info$pid) && .port_is_listening(info$fleet_port)
  }

  list(
    running         = running,
    remote          = isTRUE(info$remote),
    pid             = info$pid,
    fleet_address   = info$fleet_address,
    control_address = info$control_address,
    ports           = list(
      fleet       = info$fleet_port,
      control     = info$control_port,
      serverappio = info$serverappio_port
    ),
    federation_id   = info$federation_id,
    ca_cert_pem     = info$ca_cert_pem,
    detached        = isTRUE(info$detached),
    started_at      = info$started_at
  )
}

# --- Auto-discovery helpers ---

#' Detect all routable IPv4 addresses on the researcher's machine
#'
#' Returns a prioritized list of IPs: OS-routed IP first (from UDP socket
#' trick), then VPN/tunnel interfaces (tun, utun, wg, tailscale), then
#' remaining LAN interfaces. Excludes loopback (127.x.x.x) and link-local
#' (169.254.x.x).
#'
#' @return Character vector of IPv4 address strings, ordered by priority.
#' @keywords internal
.detect_all_ips <- function() {
  ips <- character(0)
  ipv4_re <- "^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$"

  # 1. OS-routed IP (respects routing table, best for most cases)
  routed_ip <- tryCatch({
    python_bin <- tryCatch(.client_python_cmd(), error = function(e) "python3")
    out <- system2(python_bin, c("-c",
      shQuote("import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0]); s.close()")),
      stdout = TRUE, stderr = TRUE)
    addr <- trimws(out[1])
    if (grepl(ipv4_re, addr)) addr else NULL
  }, error = function(e) NULL, warning = function(w) NULL)

  if (!is.null(routed_ip)) ips <- c(ips, routed_ip)

  # 2. Enumerate all interfaces via ifconfig/ip (cross-platform)
  iface_ips <- tryCatch({
    .parse_interface_ips()
  }, error = function(e) character(0))

  ips <- c(ips, iface_ips)

  # Deduplicate, exclude loopback and link-local
  ips <- unique(ips)
  ips <- ips[!grepl("^127\\.", ips)]
  ips <- ips[!grepl("^169\\.254\\.", ips)]

  if (length(ips) == 0L) {
    stop("Could not detect any routable IP. ",
         "Please provide superlink_address explicitly.",
         call. = FALSE)
  }

  ips
}

#' Detect the researcher's primary routable local IP address
#'
#' Convenience wrapper that returns only the first (highest priority) IP.
#'
#' @return Character; an IPv4 address string.
#' @keywords internal
.detect_local_ip <- function() {
  .detect_all_ips()[1]
}

#' Parse interface IPs from system commands
#'
#' Uses \code{ifconfig} (macOS/BSD) or \code{ip addr} (Linux) to list all
#' IPv4 addresses. Returns them ordered: VPN/tunnel interfaces first
#' (tun, utun, wg, tailscale, ts), then physical interfaces.
#'
#' @return Character vector of IPv4 addresses, VPN-first ordering.
#' @keywords internal
.parse_interface_ips <- function() {
  ipv4_re <- "^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$"
  vpn_iface_re <- "^(tun|utun|wg|tailscale|ts|nordlynx|proton)"
  vpn_ips <- character(0)
  lan_ips <- character(0)

  if (.Platform$OS.type == "unix") {
    # Try `ip addr` first (Linux), fall back to `ifconfig`
    out <- tryCatch(
      system2("ip", c("-4", "-o", "addr", "show"), stdout = TRUE, stderr = TRUE),
      error = function(e) NULL, warning = function(w) NULL
    )

    if (!is.null(out) && length(out) > 0) {
      # `ip -4 -o addr show` output:
      # 2: eth0    inet 192.168.1.5/24 brd 192.168.1.255 scope global eth0
      for (line in out) {
        parts <- strsplit(trimws(line), "\\s+")[[1]]
        iface_idx <- which(parts == "inet")
        if (length(iface_idx) == 0) next
        addr_cidr <- parts[iface_idx + 1]
        addr <- sub("/.*", "", addr_cidr)
        if (!grepl(ipv4_re, addr)) next

        # Interface name is the 2nd field (strip trailing colon)
        iface <- gsub(":$", "", parts[2])
        if (grepl(vpn_iface_re, iface)) {
          vpn_ips <- c(vpn_ips, addr)
        } else {
          lan_ips <- c(lan_ips, addr)
        }
      }
    } else {
      # macOS / BSD: ifconfig
      out <- tryCatch(
        system2("ifconfig", stdout = TRUE, stderr = TRUE),
        error = function(e) character(0), warning = function(w) character(0)
      )
      current_iface <- ""
      for (line in out) {
        # Interface header: "en0: flags=..."
        if (grepl("^[a-zA-Z]", line) && grepl(":", line)) {
          current_iface <- sub(":.*", "", line)
        }
        # IPv4 line: "  inet 192.168.1.5 netmask ..."
        m <- regmatches(line, regexpr("inet ([0-9.]+)", line))
        if (length(m) > 0) {
          addr <- sub("^inet\\s+", "", m)
          if (!grepl(ipv4_re, addr)) next
          if (grepl(vpn_iface_re, current_iface)) {
            vpn_ips <- c(vpn_ips, addr)
          } else {
            lan_ips <- c(lan_ips, addr)
          }
        }
      }
    }
  }

  # VPN IPs first (more likely to be the right route for remote nodes)
  c(vpn_ips, lan_ips)
}

# --- Detached SuperLink state management ---

#' Path to the SuperLink state file
#' @keywords internal
.superlink_state_path <- function() {
  file.path(.client_venv_root(), "superlink", "state.json")
}

#' Save SuperLink state for cross-session reconnection
#' @keywords internal
.save_superlink_state <- function(info) {
  state <- list(
    pid              = info$pid,
    fleet_address    = info$fleet_address,
    control_address  = info$control_address,
    fleet_port       = info$fleet_port,
    control_port     = info$control_port,
    serverappio_port = info$serverappio_port,
    flwr_home        = info$flwr_home,
    log_path         = info$log_path,
    federation_id    = info$federation_id,
    ca_cert_pem      = info$ca_cert_pem,
    ca_cert_path     = info$ca_cert_path,
    detached         = TRUE,
    started_at       = format(info$started_at, "%Y-%m-%dT%H:%M:%S")
  )
  path <- .superlink_state_path()
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(state, path, auto_unbox = TRUE, pretty = TRUE)
}

#' Load SuperLink state from file
#' @return Named list or NULL.
#' @keywords internal
.load_superlink_state <- function() {
  path <- .superlink_state_path()
  if (!file.exists(path)) return(NULL)
  tryCatch({
    state <- jsonlite::fromJSON(path, simplifyVector = TRUE)
    state$process <- NULL  # No processx object in detached mode
    state$pid <- as.integer(state$pid)
    state$fleet_port <- as.integer(state$fleet_port)
    state$control_port <- as.integer(state$control_port)
    state$serverappio_port <- as.integer(state$serverappio_port)
    state
  }, error = function(e) NULL)
}

#' Clear SuperLink state file
#' @keywords internal
.clear_superlink_state <- function() {
  path <- .superlink_state_path()
  if (file.exists(path)) unlink(path)
}

#' Check if a local PID is alive (cross-platform, never kills the process)
#'
#' Uses the \pkg{ps} package (\code{ps_handle} + \code{ps_is_running}), which
#' also detects PID reuse via the process create-time. We deliberately avoid
#' \code{tools::pskill(pid, 0L)}: on Windows \code{pskill} always calls
#' \code{TerminateProcess}, so the Unix "signal 0" liveness trick would KILL the
#' very process it is meant to probe.
#' @keywords internal
.pid_is_alive_local <- function(pid) {
  if (is.null(pid) || is.na(pid)) return(FALSE)
  tryCatch(ps::ps_is_running(ps::ps_handle(as.integer(pid))),
           error = function(e) FALSE)
}
