# Module: Coordinator (public SuperLink) helper
# One command to stand up a publicly reachable SuperLink that BOTH the client
# and every SuperNode dial OUT to -- the single rendezvous endpoint that makes
# NAT traversal work. Run ONCE on a host all nodes can reach.

#' Stand up a public coordinator (SuperLink)
#'
#' Launches a long-lived \code{flower-superlink} with a TLS server certificate
#' whose SAN matches \code{public_dns}, so remote SuperNodes (and the
#' researcher's \code{flwr run}) can dial it by name across NAT. Prints the
#' exact values to configure on each node (\code{dsflower.coordinator_*}); after
#' that the researcher just calls \code{ds.flower.fit()} and the address/CA are
#' auto-discovered.
#'
#' Run this ONCE on a host that every node can open an outbound connection to
#' (a small public VM, or one of the federation's public hosts).
#'
#' @param public_dns Character; the public DNS name (or IP) nodes dial, e.g.
#'   "coordinator.datashield.live". Becomes the certificate SAN.
#' @param fleet_port Integer; Fleet API port SuperNodes dial (default 9092).
#' @param control_port Integer; Control API port \code{flwr run} dials
#'   (default 9093).
#' @param serverappio_port Integer; ServerAppIO API port (default 9091).
#' @param cert_dir Character or NULL; directory for certs + logs. Defaults to a
#'   stable dir under the client data root.
#' @param cert_days Integer; certificate validity in days (default 90).
#' @return Invisible list: superlink_address, control_address, ca_cert_path,
#'   ca_cert_pem, ca_cert_b64, pid, log_path, process.
#' @export
ds.flower.coordinator.up <- function(public_dns,
                                      fleet_port = 9092L,
                                      control_port = 9093L,
                                      serverappio_port = 9091L,
                                      cert_dir = NULL,
                                      cert_days = 90L,
                                      authorized_keys = NULL) {
  if (missing(public_dns) || is.null(public_dns) || !nzchar(public_dns)) {
    stop("'public_dns' is required (the address SuperNodes will dial).",
         call. = FALSE)
  }
  .require_flwr_cli()

  base_dir <- file.path(.client_venv_root(), "coordinator_server")
  if (is.null(cert_dir)) cert_dir <- file.path(base_dir, "certs")
  log_path <- file.path(base_dir, "superlink.log")
  dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)

  # SAN: DNS name, or IP if public_dns is numeric.
  is_ip <- grepl("^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$", public_dns)
  extra_sans <- if (is_ip) paste0("IP:", public_dns) else paste0("DNS:", public_dns)
  tls_info <- .generate_tls_certs(cert_dir, extra_sans = extra_sans,
                                  cert_days = cert_days)

  # Reclaim our ports from any abandoned coordinator.
  .kill_orphan_on_port(fleet_port)
  .kill_orphan_on_port(control_port)
  .kill_orphan_on_port(serverappio_port)

  # Only the Fleet API (SuperNodes dial it) and Control API (the researcher's
  # `flwr run` dials it) need to be public. The ServerAppIO API is internal:
  # the SuperLink runs the ServerApp on THIS host and they talk over loopback,
  # so bind it to 127.0.0.1 to avoid needless public exposure.
  # SuperNode authentication (Flower >= 1.27): enable auth, then register each
  # authorized node public key via the Flower CLI after the SuperLink is up.
  # CA possession alone must not be enough to join.
  auth_args <- character(0)
  authorized_key_files <- character(0)
  if (!is.null(authorized_keys)) {
    authorized_key_files <- .collect_pubkey_files(authorized_keys, base_dir)
    if (length(authorized_key_files) > 0) {
      auth_args <- "--enable-supernode-auth"
    }
  } else {
    warning("Coordinator started WITHOUT SuperNode authentication: any node ",
            "with the CA + network reach can join. Pass authorized_keys= (see ",
            "ds.flower.keygen) for a public/untrusted coordinator.", call. = FALSE)
  }

  args <- c(
    "--ssl-certfile", tls_info$srv_cert_path,
    "--ssl-keyfile", tls_info$srv_key_path,
    "--ssl-ca-certfile", tls_info$ca_cert_path,
    auth_args,
    "--fleet-api-address", paste0("0.0.0.0:", fleet_port),
    "--control-api-address", paste0("0.0.0.0:", control_port),
    "--serverappio-api-address", paste0("127.0.0.1:", serverappio_port)
  )
  proc <- processx::process$new(
    command = .client_superlink_cmd(),
    args = args,
    stdout = log_path, stderr = "2>&1",
    cleanup = FALSE, cleanup_tree = FALSE,
    env = .client_venv_env(extra = c(FLWR_HOME = base_dir))
  )
  .wait_superlink_ready(proc, fleet_port, log_path, timeout = 20)

  # Register authorized SuperNode public keys with the running SuperLink via
  # the Flower CLI (uses a local SuperLink connection over the Control API).
  if (length(authorized_key_files) > 0) {
    cli_home <- file.path(base_dir, "cli_home")
    dir.create(cli_home, recursive = TRUE, showWarnings = FALSE)
    writeLines(c(
      "[superlink]", 'default = "coord"', "",
      "[superlink.coord]",
      paste0('address = "127.0.0.1:', control_port, '"'),
      paste0('root-certificates = "', tls_info$ca_cert_path, '"')
    ), file.path(cli_home, "config.toml"))
    registered <- 0L
    for (pub in authorized_key_files) {
      r <- tryCatch(processx::run(
        .client_flwr_cmd(), c("supernode", "register", pub, "coord"),
        env = .client_venv_env(extra = c(FLWR_HOME = cli_home)),
        error_on_status = FALSE), error = function(e) NULL)
      if (!is.null(r) && r$status == 0L) {
        registered <- registered + 1L
      } else {
        warning("Failed to register node key '", pub, "': ",
                if (!is.null(r)) r$stderr else "CLI error", call. = FALSE)
      }
    }
    message("Registered ", registered, "/", length(authorized_key_files),
            " authorized SuperNode key(s).")
  }

  superlink_address <- paste0(public_dns, ":", fleet_port)
  control_address   <- paste0(public_dns, ":", control_port)
  ca_cert_b64 <- jsonlite::base64_enc(charToRaw(tls_info$ca_cert_pem))

  info <- list(
    superlink_address = superlink_address,
    control_address   = control_address,
    ca_cert_path      = tls_info$ca_cert_path,
    ca_cert_pem       = tls_info$ca_cert_pem,
    ca_cert_b64       = ca_cert_b64,
    pid               = proc$get_pid(),
    log_path          = log_path,
    process           = proc
  )
  .dsflower_client_env$.coordinator <- info

  message(
    "\n=== dsFlower coordinator is up (PID ", proc$get_pid(), ") ===\n",
    "  Fleet API (SuperNodes): ", superlink_address, "\n",
    "  Control API (flwr run): ", control_address, "\n",
    "  CA cert: ", tls_info$ca_cert_path, "\n\n",
    "Configure each node ONCE (operator), e.g. in Rserve/Rprofile:\n",
    "  options(\n",
    "    dsflower.coordinator_address = \"", superlink_address, "\",\n",
    "    dsflower.coordinator_control_address = \"", control_address, "\",\n",
    "    dsflower.coordinator_ca = \"<path to ca.pem on the node>\"\n",
    "  )\n",
    "Copy the CA above to each node. Then the researcher just runs\n",
    "  ds.flower.fit(conns, ...)  -- the coordinator is auto-discovered.\n"
  )
  invisible(info)
}

#' Collect public-key FILE paths from paths, a directory, or literal key
#' strings (literals are written to temp .pub files). Used for CLI registration.
#' @keywords internal
.collect_pubkey_files <- function(x, scratch_dir) {
  files <- character(0)
  i <- 0L
  for (item in x) {
    if (dir.exists(item)) {
      files <- c(files, list.files(item, pattern = "\\.pub$", full.names = TRUE))
    } else if (file.exists(item)) {
      files <- c(files, item)
    } else if (grepl("^(ecdsa|ssh)-", item)) {
      i <- i + 1L
      tf <- file.path(scratch_dir, paste0("node_key_", i, ".pub"))
      writeLines(item, tf)
      files <- c(files, tf)
    }
  }
  unique(files)
}

#' Generate an ECDSA keypair for SuperNode authentication
#'
#' Wraps \code{ssh-keygen -t ecdsa -b 384}. The operator generates one keypair
#' per node, registers the \code{.pub} files with the coordinator
#' (\code{ds.flower.coordinator.up(authorized_keys=...)}), and installs the
#' private+public key on the node (\code{dsflower.node_auth_private_key} /
#' \code{dsflower.node_auth_public_key}).
#'
#' @param out_dir Character; directory to write the key pair into.
#' @param name Character; base filename for the key (default "node").
#' @return Invisible list with \code{private} and \code{public} key paths.
#' @export
ds.flower.keygen <- function(out_dir, name = "node") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  keyfile <- file.path(out_dir, name)
  if (file.exists(keyfile)) {
    stop("A key already exists at ", keyfile, "; choose another name/out_dir.",
         call. = FALSE)
  }
  ssh <- Sys.which("ssh-keygen")
  if (!nzchar(ssh)) stop("ssh-keygen not found on PATH.", call. = FALSE)
  res <- processx::run(
    ssh, c("-t", "ecdsa", "-b", "384", "-f", keyfile, "-N", "",
           "-C", paste0("dsflower-", name), "-q"),
    error_on_status = FALSE)
  if (res$status != 0L) {
    stop("ssh-keygen failed: ", res$stderr, call. = FALSE)
  }
  Sys.chmod(keyfile, "0600")
  message("Generated node auth keypair:\n  private: ", keyfile,
          "\n  public:  ", paste0(keyfile, ".pub"))
  invisible(list(private = keyfile, public = paste0(keyfile, ".pub")))
}

#' Stop the coordinator started by \code{ds.flower.coordinator.up()}
#'
#' @return Invisible TRUE.
#' @export
ds.flower.coordinator.stop <- function() {
  info <- .dsflower_client_env$.coordinator
  if (!is.null(info) && !is.null(info$process) &&
      info$process$is_alive()) {
    info$process$signal(15L)
    info$process$wait(timeout = 5000)
    if (info$process$is_alive()) info$process$kill()
  }
  .dsflower_client_env$.coordinator <- NULL
  message("Coordinator stopped.")
  invisible(TRUE)
}
