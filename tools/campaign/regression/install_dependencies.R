# Current R needs current compiled packages rather than jammy's R 4.1 binaries.
# Posit's documented Linux binary user agent also avoids building libarrow.
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"),
        Ncpus = 8L, timeout = 900,
        HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
          paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])))
packages <- c("rlang", "cli", "glue", "magrittr", "vctrs", "purrr", "tidyselect",
              "cpp11", "bit", "bit64", "jsonlite", "digest", "processx", "ps",
              "filelock", "curl", "openssl", "arrow", "DSI", "DSLite", "resourcer")
install.packages(packages, lib = Sys.getenv("R_LIBS"))
stopifnot(all(vapply(packages, requireNamespace, logical(1), quietly = TRUE)))
path <- tempfile(fileext = ".parquet")
arrow::write_parquet(data.frame(x = 1:3), path)
stopifnot(identical(as.data.frame(arrow::read_parquet(path))$x, 1:3))
unlink(path)
