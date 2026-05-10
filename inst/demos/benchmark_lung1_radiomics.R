script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}

source(file.path(script_dir(), "lib", "benchmark_utils.R"))

make_lung1_style <- function(n = as.integer(demo_env("DSFLOWER_LUNG1_SIM_N", 420L)), seed = 4242L) {
  set.seed(seed)
  stage <- sample(1:4, n, replace = TRUE, prob = c(0.22, 0.28, 0.34, 0.16))
  age <- pmin(pmax(round(stats::rnorm(n, 67, 9)), 38), 89)
  tumor_volume <- stats::rlnorm(n, meanlog = 4.2 + 0.28 * stage, sdlog = 0.55)
  energy <- log1p(tumor_volume) + stats::rnorm(n, 0, 0.35)
  compactness2 <- pmax(stats::rnorm(n, 0.65 - 0.05 * stage, 0.13), 0.05)
  glrlm_rlnu <- stats::rlnorm(n, 4.6 + 0.12 * stage, 0.42)
  wavelet_hlh_rlnu <- stats::rlnorm(n, 4.2 + 0.18 * stage, 0.48)
  firstorder_entropy <- pmax(stats::rnorm(n, 3.1 + 0.12 * stage, 0.45), 0.1)
  shape_sphericity <- pmax(pmin(stats::rnorm(n, 0.62 - 0.04 * stage, 0.11), 1), 0.05)
  gldm_dependence_non_uniformity <- stats::rlnorm(n, 4.0 + 0.16 * stage, 0.45)
  risk <- -5.6 + 0.028 * age + 0.55 * stage + 0.28 * energy -
    1.35 * compactness2 + 0.0018 * glrlm_rlnu + 0.0022 * wavelet_hlh_rlnu +
    0.42 * firstorder_entropy - 1.1 * shape_sphericity
  outcome <- stats::rbinom(n, 1, stats::plogis(risk))
  data.frame(
    id = paste0("lung1_style_", seq_len(n)),
    age = age,
    stage = stage,
    original_shape_VoxelVolume = tumor_volume,
    original_firstorder_Energy = energy,
    original_shape_Compactness2 = compactness2,
    original_glrlm_RunLengthNonUniformity = glrlm_rlnu,
    wavelet_HLH_glrlm_RunLengthNonUniformity = wavelet_hlh_rlnu,
    original_firstorder_Entropy = firstorder_entropy,
    original_shape_Sphericity = shape_sphericity,
    original_gldm_DependenceNonUniformity = gldm_dependence_non_uniformity,
    outcome = outcome,
    check.names = FALSE
  )
}

load_external_radiomics <- function(path, target = demo_env("DSFLOWER_LUNG1_TARGET", "outcome")) {
  if (!nzchar(path) || !file.exists(path)) return(NULL)
  ext <- tolower(tools::file_ext(path))
  data <- switch(
    ext,
    rds = readRDS(path),
    csv = utils::read.csv(path, check.names = FALSE),
    tsv = utils::read.delim(path, check.names = FALSE),
    stop("Unsupported radiomics feature file extension: ", ext, call. = FALSE)
  )
  if (!is.data.frame(data)) stop("External radiomics feature file must contain a data.frame.", call. = FALSE)
  if (!"id" %in% names(data)) data$id <- paste0("radiomics_", seq_len(nrow(data)))
  if (!target %in% names(data)) stop("External radiomics file does not contain target column: ", target, call. = FALSE)
  names(data)[names(data) == target] <- "outcome"
  numeric_cols <- names(data)[vapply(data, is.numeric, logical(1))]
  keep <- unique(c("id", setdiff(numeric_cols, "outcome"), "outcome"))
  data[keep]
}

external_path <- demo_env("DSFLOWER_LUNG1_FEATURES", "")
dataset <- load_external_radiomics(external_path)
data_mode <- if (is.null(dataset)) {
  dataset <- make_lung1_style()
  "deterministic LUNG1-style derived radiomics fixture"
} else {
  paste("external radiomics feature file", normalizePath(external_path))
}

features <- setdiff(names(dataset), c("id", "outcome"))

run_dsflower_benchmark(
  data = dataset,
  features = features,
  demo_id = "lung1_radiomics",
  dataset_label = "LUNG1-style radiomics federated benchmark",
  data_mode = data_mode,
  default_rounds = 2L
)
