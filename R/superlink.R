# Module: SuperLink Lifecycle
# Manages the Flower SuperLink process on the researcher's machine.

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
.generate_tls_certs <- function(cert_dir, extra_sans = NULL) {
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
    "req", "-new", "-x509",
    "-key", ca_key_path,
    "-out", ca_cert_path,
    "-days", "1",
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
    "x509", "-req",
    "-in", srv_csr_path,
    "-CA", ca_cert_path,
    "-CAkey", ca_key_path,
    "-CAcreateserial",
    "-out", srv_cert_path,
    "-days", "1",
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
#' Spawns a \code{flower-superlink} process using processx with a private
#' FLWR_HOME directory. Writes a \code{config.toml} so that \code{flwr run}
#' can connect to this SuperLink.
#'
#' @param fleet_port Integer; port for the Fleet API (default 9092).
#'   SuperNodes connect here.
#' @param control_port Integer; port for the Control API (default 9093).
#'   \code{flwr run} connects here.
#' @param serverappio_port Integer; port for the ServerAppIO API (default 9091).
#' @return Invisible list with process info.
#' @export
ds.flower.superlink.start <- function(fleet_port = 9092L,
                                       control_port = 9093L,
                                       serverappio_port = 9091L) {
  .require_flwr_cli()

  # Check if already running
  existing <- .dsflower_client_env$.superlink
  if (!is.null(existing) && !is.null(existing$process) &&
      existing$process$is_alive()) {
    message("SuperLink is already running (PID: ", existing$process$get_pid(), ")")
    return(invisible(existing))
  }

  # Private FLWR_HOME
  flwr_home <- file.path(tempdir(), "dsflower_superlink")
  dir.create(flwr_home, recursive = TRUE, showWarnings = FALSE)

  # TLS certificate generation (always enabled)
  cert_dir <- file.path(flwr_home, "certs")
  tls_info <- .generate_tls_certs(cert_dir)

  # Build args for flower-superlink
  args <- c(
    "--ssl-certfile", tls_info$srv_cert_path,
    "--ssl-keyfile", tls_info$srv_key_path,
    "--ssl-ca-certfile", tls_info$ca_cert_path
  )
  args <- c(args,
    "--fleet-api-address", paste0("0.0.0.0:", fleet_port),
    "--control-api-address", paste0("0.0.0.0:", control_port),
    "--serverappio-api-address", paste0("0.0.0.0:", serverappio_port)
  )

  # Log path
  log_path <- file.path(flwr_home, "superlink.log")

  # Spawn
  proc <- processx::process$new(
    command = "flower-superlink",
    args = args,
    stdout = log_path,
    stderr = "2>&1",
    cleanup = TRUE,
    cleanup_tree = TRUE,
    env = c("current", FLWR_HOME = flwr_home)
  )

  # Write config.toml so flwr run can find this SuperLink
  config_toml <- paste0(
    "[superlink]\n",
    'default = "dsflower"\n\n',
    "[superlink.dsflower]\n",
    'address = "127.0.0.1:', control_port, '"\n',
    'root-certificates = "', tls_info$ca_cert_path, '"\n'
  )
  writeLines(config_toml, file.path(flwr_home, "config.toml"))

  fleet_address   <- paste0("127.0.0.1:", fleet_port)
  control_address <- paste0("127.0.0.1:", control_port)

  # Unique federation ID for this SuperLink instance — used to verify

  # all nodes connected to the same SuperLink after ensure.
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
    started_at       = Sys.time()
  )

  .dsflower_client_env$.superlink <- info

  # Wait for the SuperLink to be ready (listening on fleet port)
  .wait_superlink_ready(proc, fleet_port, log_path, timeout = 15)

  message("SuperLink started (PID: ", info$pid, ")")
  message("  Fleet API (SuperNodes): ", fleet_address)
  message("  Control API (flwr run): ", control_address)
  invisible(info)
}

#' Wait for SuperLink to be ready
#'
#' Verifies the process is alive and the fleet port is listening.
#' Uses \code{lsof} (macOS/Linux) to check port binding without needing
#' a TLS handshake.
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
        paste(tail(readLines(log_path, warn = FALSE), 10), collapse = "\n"),
        error = function(e) "(no log)")
      stop("SuperLink process died during startup.\nLog:\n", log_tail,
           call. = FALSE)
    }

    # Check if the process is listening on the port
    if (.port_is_listening(port)) return(invisible(TRUE))
    Sys.sleep(0.5)
  }

  log_tail <- tryCatch(
    paste(tail(readLines(log_path, warn = FALSE), 10), collapse = "\n"),
    error = function(e) "(no log)")
  stop("SuperLink did not become ready within ", timeout, " seconds.\nLog:\n",
       log_tail, call. = FALSE)
}

#' Check if a port is being listened on
#'
#' Uses \code{lsof} on macOS/Linux or \code{netstat} on Windows to check
#' if any process is listening on the given port. Does not require a
#' TLS handshake.
#'
#' @param port Integer; port number.
#' @return Logical.
#' @keywords internal
.port_is_listening <- function(port) {
  tryCatch({
    if (.Platform$OS.type == "unix") {
      out <- suppressWarnings(
        system2("lsof", c("-i", paste0(":", port), "-sTCP:LISTEN"),
                stdout = TRUE, stderr = TRUE)
      )
      length(out) > 0
    } else {
      out <- suppressWarnings(
        system2("netstat", c("-an"), stdout = TRUE, stderr = TRUE)
      )
      any(grepl(paste0(":", port, "\\s"), out) & grepl("LISTEN", out))
    }
  }, error = function(e) FALSE, warning = function(w) FALSE)
}

#' Stop the Flower SuperLink
#'
#' Sends SIGTERM, waits, then SIGKILL if needed. Cleans up temp files.
#'
#' @return Invisible TRUE.
#' @export
ds.flower.superlink.stop <- function() {
  info <- .dsflower_client_env$.superlink
  if (is.null(info) || is.null(info$process)) {
    message("No SuperLink is running.")
    return(invisible(TRUE))
  }

  proc <- info$process
  if (proc$is_alive()) {
    proc$signal(15L)  # SIGTERM
    proc$wait(timeout = 5000)
    if (proc$is_alive()) {
      proc$kill()
    }
  }

  # Cleanup temp directory
  if (!is.null(info$flwr_home) && dir.exists(info$flwr_home)) {
    unlink(info$flwr_home, recursive = TRUE)
  }

  .dsflower_client_env$.superlink <- NULL
  message("SuperLink stopped.")
  invisible(TRUE)
}

#' Get SuperLink status
#'
#' @return A named list with running, pid, fleet_address, control_address,
#'   ports, started_at.
#' @export
ds.flower.superlink.status <- function() {
  info <- .dsflower_client_env$.superlink
  if (is.null(info) || is.null(info$process)) {
    return(list(
      running         = FALSE,
      pid             = NULL,
      fleet_address   = NULL,
      control_address = NULL,
      ports           = NULL,
      ca_cert_pem     = NULL,
      started_at      = NULL
    ))
  }

  list(
    running         = info$process$is_alive(),
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
    started_at      = info$started_at
  )
}

# --- Auto-discovery helpers ---

#' Auto-resolve SuperLink address for each Opal node
#'
#' For each node, builds a prioritized list of candidate addresses and tests
#' connectivity until one succeeds. Candidates include host.docker.internal
#' (for containerized nodes), the OS-routed IP, VPN/tunnel IPs, and LAN IPs.
#'
#' @param conns DSI connections object.
#' @param symbol Character; handle symbol name.
#' @return A single address string (if all nodes need the same) or a named
#'   list of per-node addresses.
#' @keywords internal
.auto_resolve_superlink <- function(conns, symbol) {
  status <- ds.flower.superlink.status()
  if (!status$running) {
    stop("No SuperLink running. Start one with ds.flower.superlink.start() ",
         "or provide superlink_address explicitly.", call. = FALSE)
  }
  fleet_port <- status$ports$fleet

  # Query each Opal's environment
  caps <- .ds_safe_aggregate(
    conns, expr = call("flowerGetCapabilitiesDS", symbol)
  )

  # Collect all local IPs once (expensive, do it lazily)
  all_ips <- NULL

  addresses <- list()
  failed <- character(0)

  for (srv in names(caps)) {
    # Build candidate list in priority order
    candidates <- character(0)

    if (isTRUE(caps[[srv]]$is_docker)) {
      # Docker nodes: try host.docker.internal first, then all local IPs
      candidates <- c(candidates, paste0("host.docker.internal:", fleet_port))
    }

    # Add all local IPs as candidates (LAN, VPN, etc.)
    if (is.null(all_ips)) all_ips <- .detect_all_ips()
    for (ip in all_ips) {
      candidates <- c(candidates, paste0(ip, ":", fleet_port))
    }

    # Also try localhost for nodes on the same machine
    candidates <- c(candidates, paste0("127.0.0.1:", fleet_port))
    candidates <- unique(candidates)

    # Test each candidate until one works
    resolved <- NULL
    tried <- character(0)
    for (candidate in candidates) {
      check <- .check_node_connectivity(conns, srv, candidate)
      if (isTRUE(check$reachable)) {
        resolved <- candidate
        message("  ", srv, ": SuperLink reachable at ", candidate)
        break
      }
      tried <- c(tried, candidate)
    }

    if (is.null(resolved)) {
      failed <- c(failed, srv)
      warning(
        srv, " cannot reach SuperLink. Tried: ",
        paste(tried, collapse = ", "), ". ",
        "Provide superlink_address explicitly for this node.",
        call. = FALSE
      )
    } else {
      addresses[[srv]] <- resolved
    }
  }

  if (length(failed) == length(caps)) {
    stop(
      "No Opal node can reach the SuperLink. ",
      "Tried: ", paste(unique(unlist(
        lapply(caps, function(x) {
          if (isTRUE(x$is_docker)) "host.docker.internal" else "local IPs"
        })
      )), collapse = ", "), ". ",
      "Provide superlink_address explicitly (see ?ds.flower.nodes.ensure).",
      call. = FALSE
    )
  }

  # If all same -> return single string; otherwise named list
  unique_addrs <- unique(unlist(addresses))
  if (length(unique_addrs) == 1L) return(unique_addrs)
  addresses
}

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
    out <- system2("python3", c("-c",
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

#' Check connectivity from a single Opal node to a candidate address
#'
#' @param conns DSI connections object.
#' @param srv Character; server name.
#' @param address Character; "host:port" to test.
#' @return Named list with \code{reachable} and \code{error}.
#' @keywords internal
.check_node_connectivity <- function(conns, srv, address) {
  tryCatch({
    res <- DSI::datashield.aggregate(
      conns[srv],
      expr = call("flowerCheckConnectivityDS", address)
    )
    res[[srv]]
  }, error = function(e) {
    list(reachable = FALSE, error = conditionMessage(e))
  })
}
