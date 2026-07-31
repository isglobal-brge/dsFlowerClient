# Module: Client Utilities
# Internal utility functions for session management and symbol generation.

`%||%` <- function(x, y) if (is.null(x)) y else x

# Raw payload ceiling which remains below DSI's one-million-character
# expression parser boundary after URL-safe base64 encoding.
.dsi_max_raw_chunk_bytes <- 512L * 1024L

#' Validate one raw chunk carried inside a DSI expression
#' @keywords internal
.dsi_raw_chunk_bytes <- function(value) {
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || value <= 0 || value %% 1 != 0 ||
      value > .dsi_max_raw_chunk_bytes) {
    stop("chunk_bytes must be a single positive integer no larger than 512 KiB.",
         call. = FALSE)
  }
  as.integer(value)
}

#' Internal environment for storing dsFlowerClient session state
#' @keywords internal
.dsflower_client_env <- new.env(parent = emptyenv())

#' Generate a 128-bit capability token with a protocol-specific prefix
#'
#' Uses Python's operating-system-backed \code{secrets} generator. These tokens
#' authorize access to transient node-side resources, so R's statistical PRNG is
#' not an appropriate source even when collision probability would be low.
#'
#' @param prefix One of the protocol-owned prefixes \code{dsf}, \code{app}, or
#'   \code{usr}.
#' @return Character scalar of the form \code{prefix_[0-9a-f]\{32\}}.
#' @keywords internal
.new_capability_token <- function(prefix) {
  if (!is.character(prefix) || length(prefix) != 1L || is.na(prefix) ||
      !prefix %in% c("dsf", "app", "usr")) {
    stop("Unknown capability-token prefix.", call. = FALSE)
  }
  res <- tryCatch(
    processx::run(
      .client_python_cmd(),
      c("-c", paste(
        "import secrets, sys;",
        "print(sys.argv[1] + '_' + secrets.token_hex(16))"), prefix),
      error_on_status = FALSE,
      timeout = 5000
    ),
    error = function(e) NULL
  )
  token <- if (!is.null(res) && identical(res$status, 0L)) {
    trimws(res$stdout)
  } else {
    ""
  }
  expected <- paste0("^", prefix, "_[0-9a-f]{32}$")
  if (length(token) != 1L || !grepl(expected, token)) {
    stop("Could not generate a cryptographically secure capability token.",
         call. = FALSE)
  }
  token
}

#' Generate a unique temporary symbol name
#'
#' @param prefix Character; prefix for the generated symbol.
#' @return Character; a unique symbol string.
#' @keywords internal
.generate_symbol <- function(prefix = "dsF") {
  paste0(prefix, ".",
         paste(sample(c(letters, LETTERS, 0:9), 6,
                      replace = TRUE),
               collapse = ""))
}

#' Encode a complex R object as JSON for DataSHIELD transport
#'
#' @param x An R object to encode.
#' @return A JSON string if x is complex, or x unchanged if scalar.
#' @keywords internal
.ds_encode <- function(x) {
  if (is.list(x) || (is.vector(x) && length(x) > 1)) {
    json <- as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null"))
    # URL-safe base64: no +/= that could confuse Opal's R expression parser
    b64 <- gsub("[\r\n]", "", jsonlite::base64_enc(charToRaw(json)))
    b64 <- gsub("\\+", "-", b64)
    b64 <- gsub("/", "_", b64)
    b64 <- gsub("=+$", "", b64)
    paste0("B64:", b64)
  } else {
    x
  }
}

#' Validate and order one DSI result per requested node
#'
#' DSI 1.8 may represent a per-node aggregate error as a named NULL element or
#' drop that element from the returned list. Both forms, and every malformed or
#' misassociated outer result, fail closed here.
#' @keywords internal
.dsi_exact_node_results <- function(result, conns) {
  hosts <- names(conns)
  if (!length(hosts) || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts) || !is.list(result) ||
      length(result) != length(hosts)) {
    return(NULL)
  }
  result_names <- names(result)
  if (is.null(result_names) || anyNA(result_names) ||
      any(!nzchar(result_names)) || anyDuplicated(result_names) ||
      !setequal(result_names, hosts)) {
    return(NULL)
  }
  result[hosts]
}

#' Retry one idempotent aggregate until every node returns an explicit ACK
#'
#' The exact same call object is reused on every attempt. A named NULL (DSI's
#' per-node error representation) or a malformed outer mapping is retryable; an
#' explicit non-NULL but invalid ACK fails immediately.
#' @keywords internal
.dsi_retry_exact_aggregate <- function(conns, expr, validate, operation,
                                       attempts = 3L) {
  hosts <- names(conns)
  if (!length(hosts) || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts)) {
    stop("DSI operations require non-empty, unique node names.", call. = FALSE)
  }
  last_missing <- hosts
  for (attempt in seq_len(attempts)) {
    raw_result <- tryCatch(
      DSI::datashield.aggregate(conns, expr),
      error = function(e) NULL)
    result <- .dsi_exact_node_results(raw_result, conns)
    if (is.null(result)) {
      last_missing <- hosts
      next
    }
    missing <- hosts[vapply(result, is.null, logical(1))]
    invalid <- hosts[vapply(hosts, function(host) {
      value <- result[[host]]
      !is.null(value) && !isTRUE(validate(value, host))
    }, logical(1))]
    if (length(invalid)) {
      stop(operation, " returned an invalid ACK on: ",
           paste(invalid, collapse = ", "), ".", call. = FALSE)
    }
    if (!length(missing)) return(result)
    last_missing <- missing
  }
  stop(operation, " returned no ACK on: ",
       paste(last_missing, collapse = ", "), ".", call. = FALSE)
}

#' Require an explicit successful DSI ASSIGN callback from every node
#' @keywords internal
.dsi_assign_exact <- function(conns, operation, invoke) {
  hosts <- names(conns)
  if (!length(hosts) || anyNA(hosts) || any(!nzchar(hosts)) ||
      anyDuplicated(hosts)) {
    stop("DSI assignments require non-empty, unique node names.", call. = FALSE)
  }
  succeeded <- stats::setNames(rep(FALSE, length(hosts)), hosts)
  failed <- stats::setNames(rep(FALSE, length(hosts)), hosts)
  callback_invalid <- FALSE
  success <- function(node, ...) {
    if (length(node) != 1L || is.na(node) || !node %in% hosts) {
      callback_invalid <<- TRUE
    } else {
      succeeded[[node]] <<- TRUE
    }
  }
  error <- function(node, message = NULL, ...) {
    if (length(node) != 1L || is.na(node) || !node %in% hosts) {
      callback_invalid <<- TRUE
    } else {
      failed[[node]] <<- TRUE
    }
  }
  thrown <- tryCatch({
    invoke(success, error)
    NULL
  }, error = identity)
  bad <- hosts[!succeeded | failed]
  if (!is.null(thrown) || callback_invalid || length(bad)) {
    if (!length(bad)) bad <- hosts
    stop(operation, " failed or returned no ACK on: ",
         paste(bad, collapse = ", "), ".", call. = FALSE)
  }
  invisible(TRUE)
}

#' Require exact completion of one DSI expression assignment
#' @keywords internal
.dsi_assign_expr_exact <- function(conns, symbol, expr, operation) {
  .dsi_assign_exact(conns, operation, function(success, error) {
    DSI::datashield.assign.expr(
      conns, symbol = symbol, expr = expr,
      success = success, error = error, errors.print = FALSE)
  })
}

#' Require exact completion of one DSI resource assignment
#' @keywords internal
.dsi_assign_resource_exact <- function(conns, symbol, resource, operation) {
  .dsi_assign_exact(conns, operation, function(success, error) {
    DSI::datashield.assign.resource(
      conns, symbol = symbol, resource = resource,
      success = success, error = error, errors.print = FALSE)
  })
}

#' Resilient datashield.aggregate that tolerates per-server failures
#'
#' @param conns DSI connections object.
#' @param expr The call expression to evaluate.
#' @return Named list of results (only successful servers).
#' @keywords internal
.ds_safe_aggregate <- function(conns, expr) {
  server_names <- names(conns)
  results <- list()
  errors <- list()
  for (srv in server_names) {
    tryCatch({
      res <- DSI::datashield.aggregate(conns[srv], expr = expr)
      mapped <- .dsi_exact_node_results(res, conns[srv])
      if (is.null(mapped) || is.null(mapped[[srv]])) {
        errors[[srv]] <- "node returned no aggregate result"
      } else {
        results[srv] <- list(mapped[[srv]])
      }
    }, error = function(e) {
      errors[[srv]] <<- e$message
    })
  }
  if (length(errors) > 0) {
    attr(results, "ds_errors") <- errors
  }
  results
}

# --- Code generation helpers ---

#' Format an R value for code generation
#' @param x An R value to format as code
#' @return Character string of valid R code
#' @keywords internal
.format_r_value <- function(x) {
  if (is.null(x)) return("NULL")
  if (is.character(x) && length(x) == 1) return(paste0('"', x, '"'))
  if (is.integer(x) && length(x) == 1) return(paste0(x, "L"))
  if (is.numeric(x) && length(x) == 1) return(as.character(x))
  if (is.logical(x) && length(x) == 1) return(as.character(x))
  if (is.numeric(x)) return(paste0("c(", paste(x, collapse = ", "), ")"))
  if (is.character(x)) return(paste0('c("', paste(x, collapse = '", "'), '")'))
  deparse(x, width.cutoff = 500L)
}

#' Build an R code string for a function call
#' @param fn_name Character; fully qualified function name
#' @param ... Named arguments to include in the call
#' @return Character string of the R call
#' @keywords internal
.build_code <- function(fn_name, ...) {
  args <- list(...)
  parts <- vapply(names(args), function(nm) {
    val <- args[[nm]]
    if (is.null(val)) return(NA_character_)
    paste0(nm, " = ", .format_r_value(val))
  }, character(1))
  parts <- parts[!is.na(parts)]
  paste0(fn_name, "(", paste(parts, collapse = ", "), ")")
}

#' Ensure the Flower CLI is available
#'
#' Checks the venv first. If not healthy, attempts to create it on-the-fly.
#'
#' @return Invisible TRUE, or stops with an error.
#' @keywords internal
.require_flwr_cli <- function() {
  if (.client_venv_is_healthy()) return(invisible(TRUE))

  # Try to auto-provision if venv is missing
  tryCatch(
    .ensure_client_venv(),
    error = function(e) {
      stop("Flower Python environment not available. ",
           "Reinstall dsFlowerClient or run .ensure_client_venv(). ",
           "Error: ", conditionMessage(e), call. = FALSE)
    }
  )
  invisible(TRUE)
}
