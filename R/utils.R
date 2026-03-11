# Module: Client Utilities
# Internal utility functions for session management and symbol generation.

`%||%` <- function(x, y) if (is.null(x)) y else x

#' Internal environment for storing dsFlowerClient session state
#' @keywords internal
.dsflower_client_env <- new.env(parent = emptyenv())

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
      results[[srv]] <- res[[srv]]
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

#' Check that the flwr CLI is available
#'
#' @return Invisible TRUE, or stops with an error.
#' @keywords internal
.require_flwr_cli <- function() {
  path <- Sys.which("flwr")
  if (!nzchar(path)) {
    stop(
      "The 'flwr' CLI is not found on the PATH. ",
      "Install Flower: pip install flwr",
      call. = FALSE
    )
  }
  invisible(TRUE)
}
