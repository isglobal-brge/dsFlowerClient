# Module: DSI transport security gate
# The inner Flower connection is plaintext on loopback, so the outer DSI
# transport must not be assumed secure. Inspect supported connectors, require
# HTTPS by default, and scope every deliberate HTTP exception to exact sites.

.dsi_member <- function(x, name) {
  if (isS4(x) && name %in% methods::slotNames(x)) {
    return(methods::slot(x, name))
  }
  if (is.environment(x)) {
    if (exists(name, envir = x, inherits = FALSE)) {
      return(get(name, envir = x, inherits = FALSE))
    }
    return(NULL)
  }
  if (is.list(x) && name %in% names(x)) return(x[[name]])
  NULL
}

.dsi_scalar_string <- function(x) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(trimws(x))) {
    return(NULL)
  }
  trimws(x)
}

.dsi_endpoint <- function(url) {
  url <- .dsi_scalar_string(url)
  if (is.null(url) || !grepl("^[A-Za-z][A-Za-z0-9+.-]*://", url)) return(NULL)

  scheme <- tolower(sub("^([A-Za-z][A-Za-z0-9+.-]*)://.*$", "\\1", url))
  authority <- sub("^[A-Za-z][A-Za-z0-9+.-]*://", "", url)
  authority <- sub("[/?#].*$", "", authority)
  authority <- sub("^.*@", "", authority)
  if (!nzchar(authority)) return(list(scheme = scheme, host = NULL))

  if (startsWith(authority, "[")) {
    close <- regexpr("]", authority, fixed = TRUE)[[1]]
    if (close < 2L) return(list(scheme = scheme, host = NULL))
    host <- substr(authority, 2L, close - 1L)
  } else {
    host <- sub(":.*$", "", authority)
  }
  host <- sub("\\.$", "", tolower(host))
  if (!nzchar(host)) return(list(scheme = scheme, host = NULL))
  list(scheme = scheme, host = host)
}

.dsi_loopback_host <- function(host) {
  host <- .dsi_scalar_string(host)
  if (is.null(host)) return(FALSE)
  host <- sub("%.*$", "", tolower(host))
  if (host %in% c("localhost", "::1", "0:0:0:0:0:0:0:1")) return(TRUE)
  parts <- strsplit(host, ".", fixed = TRUE)[[1]]
  length(parts) == 4L && identical(parts[[1]], "127") &&
    all(grepl("^[0-9]{1,3}$", parts)) &&
    all(as.integer(parts) >= 0L & as.integer(parts) <= 255L)
}

.dsi_tls_verification_state <- function(options) {
  if (!is.list(options)) return("unknown")

  peer <- options[["ssl_verifypeer"]]
  if (!is.null(peer)) {
    peer_num <- suppressWarnings(as.numeric(peer))
    if (length(peer_num) != 1L || is.na(peer_num) || peer_num <= 0) {
      return("disabled")
    }
  }

  host <- options[["ssl_verifyhost"]]
  if (!is.null(host)) {
    if (is.logical(host) && length(host) == 1L && !is.na(host)) {
      if (!isTRUE(host)) return("disabled")
      return("verified")
    }
    host_num <- suppressWarnings(as.numeric(host))
    if (length(host_num) != 1L || is.na(host_num) || host_num < 2) {
      return("disabled")
    }
  }
  "verified"
}

.dsi_httr_global_options <- function() {
  config <- getOption("httr_config", NULL)
  if (is.null(config)) return(list())
  options <- .dsi_member(config, "options")
  if (!is.list(options)) return(NULL)
  options
}

.dsi_effective_opal_tls_state <- function(local_options) {
  global_options <- .dsi_httr_global_options()
  if (!is.list(global_options) || !is.list(local_options)) {
    global_state <- .dsi_tls_verification_state(global_options)
    if (identical(global_state, "disabled")) return("disabled")
    return("unknown")
  }
  # httr applies its global config first and the request config afterwards.
  # NULL request options are compacted away, so they do not override globals.
  local_options <- local_options[
    !vapply(local_options, is.null, logical(1))
  ]
  # Mirror that precedence so an explicit safe Opal override wins, while an
  # inherited global downgrade is still detected.
  .dsi_tls_verification_state(utils::modifyList(
    global_options, local_options, keep.null = TRUE))
}

.inspect_dsi_connection <- function(conn) {
  if (inherits(conn, "DSLiteConnection")) {
    return(list(kind = "dslite", url = NULL, verification = "in_process"))
  }

  if (inherits(conn, "OpalConnection")) {
    opal <- .dsi_member(conn, "opal")
    config <- .dsi_member(opal, "config")
    return(list(
      kind = "opal",
      url = .dsi_member(opal, "url"),
      verification = .dsi_effective_opal_tls_state(
        .dsi_member(config, "options"))
    ))
  }

  if (inherits(conn, "ArmadilloConnection")) {
    handle <- .dsi_member(conn, "handle")
    global_state <- .dsi_tls_verification_state(
      .dsi_httr_global_options())
    return(list(
      kind = "armadillo",
      url = .dsi_member(handle, "url"),
      verification = if (identical(global_state, "disabled")) {
        "disabled"
      } else {
        "unknown"
      }
    ))
  }

  list(
    kind = "generic",
    url = .dsi_member(conn, "url"),
    verification = "unknown"
  )
}

.dsi_tls_attested_sites <- function() {
  value <- getOption("dsflower.dsi_tls_attested", character())
  if (is.null(value)) value <- character()
  if (!is.character(value) || anyNA(value) || any(!nzchar(value)) ||
      any(value != trimws(value)) || anyDuplicated(value)) {
    stop("Invalid dsflower.dsi_tls_attested option: expected unique, non-empty site names.",
         call. = FALSE)
  }
  unname(value)
}

.dsi_allow_insecure_loopback <- function() {
  value <- getOption("dsflower.dsi_allow_insecure_loopback", FALSE)
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    stop("Invalid dsflower.dsi_allow_insecure_loopback option: expected TRUE or FALSE.",
         call. = FALSE)
  }
  isTRUE(value)
}

.dsi_allow_insecure_http_sites <- function(
    value = getOption("dsflower.dsi_allow_insecure_http", character())) {
  if (is.null(value)) value <- character()
  if (!is.character(value) || anyNA(value) || any(!nzchar(value)) ||
      any(value != trimws(value)) || anyDuplicated(value)) {
    stop("Invalid dsflower.dsi_allow_insecure_http option: expected unique, non-empty site names.",
         call. = FALSE)
  }
  unname(value)
}

# Validate the confidentiality of the outer DSI transport.
.validate_dsi_transport_security <- function(
    conns,
    allow_insecure_http = getOption(
      "dsflower.dsi_allow_insecure_http", character())) {
  hosts <- names(conns)
  if (length(hosts) == 0L || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts)) {
    stop("Tunnel connections must have non-empty, unique node names.",
         call. = FALSE)
  }
  attested <- .dsi_tls_attested_sites()
  allow_loopback <- .dsi_allow_insecure_loopback()
  allow_http <- .dsi_allow_insecure_http_sites(allow_insecure_http)

  for (i in seq_along(conns)) {
    site <- hosts[[i]]
    details <- .inspect_dsi_connection(conns[[i]])
    if (identical(details$kind, "dslite")) next

    endpoint <- .dsi_endpoint(details$url)
    loopback <- !is.null(endpoint) && .dsi_loopback_host(endpoint$host)
    plaintext <- !is.null(endpoint) && !is.null(endpoint$host) &&
      identical(endpoint$scheme, "http")
    insecure <- identical(details$verification, "disabled") ||
      plaintext

    unverifiable <- identical(details$verification, "unknown")
    known_armadillo_https <- identical(details$kind, "armadillo") &&
      !is.null(endpoint) && !is.null(endpoint$host) &&
      identical(endpoint$scheme, "https")

    if (plaintext && site %in% allow_http) {
      warning(
        site, ": allowing explicitly configured plaintext HTTP DSI transport; ",
        "confidentiality and integrity require an independent trusted network layer.",
        call. = FALSE
      )
      next
    }

    if (loopback && (insecure ||
                     (unverifiable && !known_armadillo_https))) {
      if (allow_loopback) next
      stop(
        site, ": insecure loopback DSI transport requires the explicit ",
        "development-only loopback exception; set ",
        "options(dsflower.dsi_allow_insecure_loopback = TRUE) only for local development.",
        call. = FALSE
      )
    }

    if (identical(details$verification, "disabled")) {
      stop(site, ": TLS certificate verification is disabled for the DSI connection.",
           call. = FALSE)
    }
    if (!is.null(endpoint) && identical(endpoint$scheme, "http")) {
      stop(site, ": DSI endpoint uses plaintext HTTP; verified HTTPS is required by default. ",
           "To accept this exact site explicitly, set ",
           "options(dsflower.dsi_allow_insecure_http = c(\"", site, "\")).",
           call. = FALSE)
    }
    if (details$kind %in% c("opal", "armadillo") && !is.null(endpoint) &&
        !identical(endpoint$scheme, "https")) {
      stop(site, ": unsupported DSI endpoint scheme; verified HTTPS is required.",
           call. = FALSE)
    }

    # DSOpal retains the endpoint and curl verification options. Armadillo
    # retains its endpoint URL but not inspectable curl flags; accepting a
    # recognized HTTPS Armadillo connection keeps connector behavior symmetric,
    # while the frontend/reverse proxy remains authoritative for TLS policy.
    known_https <- !is.null(endpoint) && !is.null(endpoint$host) &&
      identical(endpoint$scheme, "https") &&
      ((identical(details$kind, "opal") &&
        identical(details$verification, "verified")) ||
       identical(details$kind, "armadillo"))
    if (known_https) next
    if (site %in% attested) next

    stop(
      site, ": DSI transport security cannot be inspected. After independently ",
      "verifying confidentiality, integrity, and peer authentication, the operator must ",
      "attest this exact site with ",
      "options(dsflower.dsi_tls_attested = c(...)).",
      call. = FALSE
    )
  }
  invisible(TRUE)
}
