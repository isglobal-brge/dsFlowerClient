script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}

source(file.path(script_dir(), "lib", "benchmark_utils.R"))

simulate_heart_like <- function(n = 303L, seed = 4242L) {
  set.seed(seed)
  age <- pmin(pmax(round(stats::rnorm(n, 55, 9)), 29), 77)
  sex <- stats::rbinom(n, 1, 0.68)
  cp <- sample(1:4, n, replace = TRUE, prob = c(0.2, 0.18, 0.29, 0.33))
  trestbps <- pmax(round(stats::rnorm(n, 132, 17)), 90)
  chol <- pmax(round(stats::rnorm(n, 245, 52)), 120)
  fbs <- stats::rbinom(n, 1, 0.15)
  restecg <- sample(0:2, n, replace = TRUE, prob = c(0.5, 0.03, 0.47))
  thalach <- pmax(round(stats::rnorm(n, 150 - 0.45 * (age - 55), 22)), 70)
  exang <- stats::rbinom(n, 1, plogis(-1.2 + 0.03 * (age - 55) + 0.45 * (cp == 4)))
  oldpeak <- pmax(stats::rnorm(n, 1.0 + 0.7 * exang, 0.9), 0)
  slope <- sample(1:3, n, replace = TRUE, prob = c(0.45, 0.45, 0.10))
  ca <- sample(0:3, n, replace = TRUE, prob = c(0.58, 0.22, 0.13, 0.07))
  thal <- sample(c(3, 6, 7), n, replace = TRUE, prob = c(0.55, 0.06, 0.39))
  risk <- -3.7 + 0.035 * age + 0.65 * sex + 0.55 * (cp == 4) +
    0.55 * exang + 0.45 * oldpeak + 0.45 * ca + 0.35 * (thal == 7) -
    0.018 * (thalach - 140)
  outcome <- stats::rbinom(n, 1, plogis(risk))
  data.frame(
    id = paste0("heart_sim_", seq_len(n)),
    age, sex, cp, trestbps, chol, fbs, restecg, thalach,
    exang, oldpeak, slope, ca, thal,
    outcome,
    check.names = FALSE
  )
}

load_uci_cleveland <- function() {
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
  tmp <- tempfile(fileext = ".data")
  utils::download.file(url, tmp, quiet = TRUE, mode = "wb")
  cols <- c(
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
  )
  raw <- utils::read.table(tmp, sep = ",", na.strings = "?", col.names = cols)
  raw <- raw[stats::complete.cases(raw), , drop = FALSE]
  data.frame(
    id = paste0("cleveland_", seq_len(nrow(raw))),
    raw[setdiff(cols, "num")],
    outcome = as.integer(raw$num > 0),
    check.names = FALSE
  )
}

dataset <- tryCatch(
  {
    x <- load_uci_cleveland()
    attr(x, "data_mode") <- "UCI Heart Disease processed Cleveland"
    x
  },
  error = function(e) {
    warning("Could not download UCI Cleveland data; using deterministic heart-like fallback: ", conditionMessage(e))
    x <- simulate_heart_like()
    attr(x, "data_mode") <- "deterministic simulated fallback"
    x
  }
)

features <- setdiff(names(dataset), c("id", "outcome"))

run_dsflower_benchmark(
  data = dataset,
  features = features,
  demo_id = "uci_heart_disease",
  dataset_label = "UCI Heart Disease federated benchmark",
  data_mode = attr(dataset, "data_mode"),
  default_rounds = 2L
)
