# Module: Public survival contracts

.SURVIVAL_LOSSES <- c("aft_weibull_nll", "aft_lognormal_nll")

.is_survival_loss <- function(loss) {
  is.character(loss) && length(loss) == 1L && !is.na(loss) &&
    loss %in% .SURVIVAL_LOSSES
}

.survival_config <- function(params, loss) {
  if (!.is_survival_loss(loss)) return(NULL)
  p <- params
  out <- list(schema_version = 1L, time_unit = p$time_unit,
              time_origin = p$time_origin, t_min = p$t_min,
              horizon = p$horizon, time_scale = p$time_scale,
              distribution = p$distribution, dispersion = p$dispersion)
  .validate_survival_config(out, loss)
}

.validate_survival_config <- function(config, loss) {
  expected <- c("schema_version", "time_unit", "time_origin", "t_min",
                "horizon", "time_scale", "distribution", "dispersion")
  if (!is.list(config) || !setequal(names(config), expected) ||
      anyDuplicated(names(config)) ||
      !identical(as.numeric(config$schema_version), 1)) {
    stop("Invalid public survival configuration schema.", call. = FALSE)
  }
  if (!identical(config$time_unit, "days") ||
      !identical(config$time_origin, "baseline")) {
    stop("Survival schema v1 requires days from baseline.", call. = FALSE)
  }
  for (name in c("time_unit", "time_origin")) {
    x <- config[[name]]
    if (!is.character(x) || length(x) != 1L || is.na(x) ||
        !nzchar(x) || nchar(x, type = "bytes") > 64L) {
      stop("Survival ", name, " must be a public nonempty label (<=64 bytes).",
           call. = FALSE)
    }
  }
  for (name in c("t_min", "horizon", "time_scale")) {
    x <- config[[name]]
    if (!is.numeric(x) || length(x) != 1L || is.na(x) ||
        !is.finite(x) || x < 1e-6 || x > 1e6) {
      stop("Survival ", name, " must be in [1e-6, 1e6].", call. = FALSE)
    }
  }
  if (config$t_min > config$horizon) {
    stop("Survival t_min must not exceed horizon.", call. = FALSE)
  }
  distribution <- config$distribution
  if (!is.character(distribution) || length(distribution) != 1L ||
      !distribution %in% c("weibull", "lognormal") ||
      !identical(loss, paste0("aft_", distribution, "_nll"))) {
    stop("Survival distribution must agree with the trusted AFT loss.",
         call. = FALSE)
  }
  if (!is.numeric(config$dispersion) || length(config$dispersion) != 1L ||
      is.na(config$dispersion) || !config$dispersion %in% c(0.5, 1, 2)) {
    stop("Survival dispersion must be one of the public grid: 0.5, 1, 2.",
         call. = FALSE)
  }
  if (identical(distribution, "weibull") &&
      config$dispersion * (log(config$horizon / config$time_scale) + 10) > 60) {
    stop("Weibull public time domain exceeds the numerical envelope.",
         call. = FALSE)
  }
  config
}

.reject_survival_private_evaluation <- function(loss) {
  if (.is_survival_loss(loss)) {
    stop("Survival private validation, holdout, cross-validation and private ",
         "HPO are unsupported; evaluate on public or independently authorized ",
         "analyst-local held-out data.", call. = FALSE)
  }
  invisible(TRUE)
}

#' Create a fixed-dispersion accelerated failure time model
#'
#' Each privacy unit is one subject under the custodian's patient policy.
#' Supply ordered targets `c(time_column, event_column)`, where event is 1 and
#' right censoring is 0. The public horizon must be chosen before inspecting
#' private outcomes. Location is bounded to [-10, 10]. Private evaluation is
#' unsupported; use public or authorized analyst-local held-out data.
#'
#' @param horizon Public administrative censoring horizon.
#' @param distribution Either `"weibull"` or `"lognormal"`.
#' @param dispersion Fixed public Weibull shape or log-normal sigma, one of
#'   `c(0.5, 1, 2)`.
#' @param time_scale Positive public time scale (AFT density includes its Jacobian).
#' @param t_min Positive public minimum time resolution.
#' @param time_unit,time_origin Public labels for the outcome time convention.
#' @param hidden_layers Integer vector of hidden widths; empty means linear.
#' @param ... Additional public neural parameters accepted by `ds.flower.model()`.
#' @return A `dsflower_model` specification.
#' @export
ds.flower.model.pytorch_aft <- function(
    horizon, distribution = "weibull", dispersion = 1, time_scale = 1,
    t_min = 1, time_unit = "days", time_origin = "baseline",
    hidden_layers = integer(0), ...) {
  ds.flower.model("pytorch_aft", horizon = horizon,
    distribution = distribution, dispersion = dispersion,
    time_scale = time_scale, t_min = t_min, time_unit = time_unit,
    time_origin = time_origin, hidden_layers = hidden_layers, ...)
}
