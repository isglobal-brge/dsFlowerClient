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
         "Install OpenSSL to use TLS (insecure = FALSE).",
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
#' @param insecure Logical; use insecure mode (default TRUE).
#' @param fleet_port Integer; port for the Fleet API (default 9092).
#'   SuperNodes connect here.
#' @param control_port Integer; port for the Control API (default 9093).
#'   \code{flwr run} connects here.
#' @param serverappio_port Integer; port for the ServerAppIO API (default 9091).
#' @return Invisible list with process info.
#' @export
ds.flower.superlink.start <- function(insecure = TRUE,
                                       fleet_port = 9092L,
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

  # TLS certificate generation when insecure = FALSE
  tls_info <- NULL
  if (!insecure) {
    cert_dir <- file.path(flwr_home, "certs")
    tls_info <- .generate_tls_certs(cert_dir)
  }

  # Build args for flower-superlink
  args <- character(0)
  if (insecure) {
    args <- c(args, "--insecure")
  } else {
    args <- c(args,
      "--ssl-certfile", tls_info$srv_cert_path,
      "--ssl-keyfile", tls_info$srv_key_path,
      "--ssl-ca-certfile", tls_info$ca_cert_path
    )
  }
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
  if (insecure) {
    config_toml <- paste0(
      "[superlink]\n",
      'default = "dsflower"\n\n',
      "[superlink.dsflower]\n",
      'address = "127.0.0.1:', control_port, '"\n',
      "insecure = true\n"
    )
  } else {
    config_toml <- paste0(
      "[superlink]\n",
      'default = "dsflower"\n\n',
      "[superlink.dsflower]\n",
      'address = "127.0.0.1:', control_port, '"\n',
      'root-certificates = "', tls_info$ca_cert_path, '"\n'
    )
  }
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
    insecure         = insecure,
    ca_cert_pem      = if (!is.null(tls_info)) tls_info$ca_cert_pem else NULL,
    started_at       = Sys.time()
  )

  .dsflower_client_env$.superlink <- info
  message("SuperLink started (PID: ", info$pid, ")")
  message("  Fleet API (SuperNodes): ", fleet_address)
  message("  Control API (flwr run): ", control_address)
  invisible(info)
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
      insecure        = NULL,
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
    insecure        = info$insecure,
    ca_cert_pem     = info$ca_cert_pem,
    started_at      = info$started_at
  )
}

# --- Auto-discovery helpers ---

#' Auto-resolve SuperLink address for each Opal node
#'
#' Resolution strategy per node:
#' \enumerate{
#'   \item If the node is in a container, try \code{host.docker.internal}.
#'   \item If bare-metal, detect the researcher's routable IP via UDP socket.
#'   \item Verify connectivity from the Opal to the candidate address.
#'   \item If verification fails, error with guidance.
#' }
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

  addresses <- list()
  local_ip <- NULL  # lazily detected

  for (srv in names(caps)) {
    if (isTRUE(caps[[srv]]$is_docker)) {
      candidate <- paste0("host.docker.internal:", fleet_port)
    } else {
      if (is.null(local_ip)) local_ip <- .detect_local_ip()
      candidate <- paste0(local_ip, ":", fleet_port)
    }
    addresses[[srv]] <- candidate
  }

  # Verify connectivity from each Opal to its candidate address
  failed <- character(0)
  for (srv in names(addresses)) {
    check <- .check_node_connectivity(conns, srv, addresses[[srv]])

    if (isTRUE(check$reachable)) {
      message("  ", srv, ": SuperLink reachable at ", addresses[[srv]])
    } else {
      failed <- c(failed, srv)
      warning(
        srv, " cannot reach SuperLink at ", addresses[[srv]],
        if (!is.null(check$error)) paste0(" (", check$error, ")"),
        call. = FALSE
      )
    }
  }

  if (length(failed) > 0 && length(failed) == length(addresses)) {
    stop(
      "No Opal node can reach the SuperLink. ",
      "Tried: ", paste(unique(unlist(addresses)), collapse = ", "), ". ",
      "Provide superlink_address explicitly (see ?ds.flower.nodes.ensure).",
      call. = FALSE
    )
  }

  if (length(failed) > 0) {
    warning(
      "Some nodes failed connectivity check: ",
      paste(failed, collapse = ", "), ". ",
      "Consider providing per-node superlink_address for those nodes.",
      call. = FALSE
    )
  }

  # If all same -> return single string; otherwise named list
  unique_addrs <- unique(unlist(addresses))
  if (length(unique_addrs) == 1L) return(unique_addrs)
  addresses
}

#' Detect the researcher's routable local IP address
#'
#' Opens a UDP socket toward a public DNS server (no data is sent) to let the
#' OS routing table choose the correct outgoing interface. Falls back to
#' \code{hostname -I} (Linux) and \code{ipconfig getifaddr} (macOS) if the
#' socket approach fails.
#'
#' @return Character; an IPv4 address string.
#' @keywords internal
.detect_local_ip <- function() {
  ip <- NULL

  # Strategy 1: Python UDP socket (works on any OS, respects routing table)
  if (is.null(ip) || !nzchar(ip %||% "")) {
    ip <- tryCatch({
      out <- system2("python3", c("-c",
        shQuote("import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0]); s.close()")),
        stdout = TRUE, stderr = TRUE)
      addr <- trimws(out[1])
      if (grepl("^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$", addr)) addr else NULL
    }, error = function(e) NULL,
       warning = function(w) NULL)
  }

  # Strategy 3: hostname -I (Linux)
  if (is.null(ip) || !nzchar(ip %||% "")) {
    ip <- tryCatch({
      out <- system2("hostname", "-I", stdout = TRUE, stderr = TRUE)
      addr <- trimws(strsplit(out[1], "\\s+")[[1]][1])
      if (grepl("^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$", addr)) addr else NULL
    }, error = function(e) NULL,
       warning = function(w) NULL)
  }

  # Strategy 4: macOS — try all active interfaces, not just en0
  if (is.null(ip) || !nzchar(ip %||% "")) {
    ip <- tryCatch({
      # List active network services, try each
      services <- system2("networksetup", "-listallhardwareports",
                          stdout = TRUE, stderr = TRUE)
      devs <- grep("^Device:", services, value = TRUE)
      devs <- gsub("^Device:\\s*", "", devs)
      addr <- NULL
      for (dev in devs) {
        out <- tryCatch(
          system2("ipconfig", c("getifaddr", dev),
                  stdout = TRUE, stderr = TRUE),
          error = function(e) "",
          warning = function(w) ""
        )
        if (length(out) > 0 && grepl("^[0-9]+\\.[0-9]+", out[1])) {
          addr <- trimws(out[1])
          break
        }
      }
      addr
    }, error = function(e) NULL,
       warning = function(w) NULL)
  }

  if (is.null(ip) || !nzchar(ip %||% "")) {
    stop("Could not auto-detect local IP. ",
         "Please provide superlink_address explicitly.",
         call. = FALSE)
  }
  ip
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
