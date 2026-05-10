script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}

source(file.path(script_dir(), "lib", "benchmark_utils.R"))

ensure_python_module <- function(module, package_spec = module) {
  dsFlowerClient:::.ensure_client_framework("sklearn")
  python <- dsFlowerClient:::.client_python_cmd()
  check <- processx::run(python, c("-c", sprintf("import %s", module)), error_on_status = FALSE)
  if (check$status == 0L) return(invisible(TRUE))

  uv <- dsFlowerClient:::.ensure_client_uv()
  install <- processx::run(
    uv,
    c("pip", "install", "--python", python, "--quiet", package_spec),
    echo_cmd = FALSE,
    echo = TRUE,
    timeout = 900
  )
  if (install$status != 0L) stop("Could not install Python package: ", package_spec, call. = FALSE)
  invisible(TRUE)
}

load_breastmnist <- function(limit = as.integer(demo_env("DSFLOWER_MEDMNIST_LIMIT", 600L))) {
  ensure_python_module("medmnist", "medmnist>=3.0.0")
  python <- dsFlowerClient:::.client_python_cmd()
  out <- tempfile(fileext = ".csv")
  code <- paste(
    "import numpy as np, pandas as pd",
    "from medmnist import BreastMNIST",
    sprintf("out_path = %s", jsonlite::toJSON(out, auto_unbox = TRUE)),
    sprintf("limit = int(%s)", jsonlite::toJSON(limit, auto_unbox = TRUE)),
    "rng = np.random.default_rng(4242)",
    "imgs = []",
    "labels = []",
    "for split in ['train', 'val', 'test']:",
    "    ds = BreastMNIST(split=split, download=True, size=28)",
    "    arr = np.asarray(ds.imgs)",
    "    lab = np.asarray(ds.labels).reshape(-1).astype(int)",
    "    imgs.append(arr)",
    "    labels.append(lab)",
    "imgs = np.concatenate(imgs, axis=0)",
    "labels = np.concatenate(labels, axis=0)",
    "idx = []",
    "for cls in np.unique(labels):",
    "    cls_idx = np.where(labels == cls)[0]",
    "    cls_idx = rng.permutation(cls_idx)",
    "    idx.extend(cls_idx[:max(1, limit // len(np.unique(labels)))].tolist())",
    "idx = np.array(idx[:limit])",
    "imgs = imgs[idx]",
    "labels = labels[idx]",
    "if imgs.ndim == 4:",
    "    imgs = imgs[..., 0]",
    "imgs = imgs.astype('float32') / 255.0",
    "pooled = imgs.reshape((-1, 7, 4, 7, 4)).mean(axis=(2, 4))",
    "flat = pooled.reshape((pooled.shape[0], -1))",
    "cols = [f'px_{i:02d}' for i in range(flat.shape[1])]",
    "df = pd.DataFrame(flat, columns=cols)",
    "df.insert(0, 'id', [f'breastmnist_{i+1}' for i in range(len(df))])",
    "df['outcome'] = labels.astype(int)",
    "df.to_csv(out_path, index=False)",
    sep = "\n"
  )
  run <- processx::run(python, c("-c", code), echo = TRUE, timeout = 900)
  if (run$status != 0L) stop("Could not build BreastMNIST tabularized dataset.", call. = FALSE)
  utils::read.csv(out, check.names = FALSE)
}

dataset <- load_breastmnist()
features <- grep("^px_", names(dataset), value = TRUE)

run_dsflower_benchmark(
  data = dataset,
  features = features,
  demo_id = "medmnist_breastmnist",
  dataset_label = "MedMNIST BreastMNIST tabularized pixel benchmark",
  data_mode = "medmnist::BreastMNIST pooled 7x7 pixels",
  default_rounds = 2L
)
