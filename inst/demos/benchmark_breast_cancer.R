script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}

source(file.path(script_dir(), "lib", "benchmark_utils.R"))

if (!requireNamespace("mlbench", quietly = TRUE)) {
  stop("The Breast Cancer demo requires the mlbench package.", call. = FALSE)
}

data("BreastCancer", package = "mlbench")
raw <- BreastCancer
raw <- raw[stats::complete.cases(raw), , drop = FALSE]

feature_cols <- setdiff(names(raw), c("Id", "Class"))
features <- paste0("bc_", seq_along(feature_cols))

dataset <- data.frame(
  id = paste0("bcw_", seq_len(nrow(raw))),
  setNames(lapply(raw[feature_cols], function(x) as.numeric(as.character(x))), features),
  outcome = as.integer(raw$Class == "malignant"),
  check.names = FALSE
)

run_dsflower_benchmark(
  data = dataset,
  features = features,
  demo_id = "breast_cancer_wisconsin",
  dataset_label = "Breast Cancer Wisconsin (Original) benchmark",
  data_mode = "mlbench::BreastCancer / UCI Breast Cancer Wisconsin (Original)",
  default_rounds = 2L
)
