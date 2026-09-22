options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"),
        Ncpus = 8L, timeout = 1200)
dir.create("/workspace/cells/Rlib", recursive = TRUE, showWarnings = FALSE)
.libPaths(c("/workspace/cells/Rlib", .libPaths()))
install.packages(c("arrow", "DSI", "DSLite"), lib = "/workspace/cells/Rlib")
stopifnot(all(vapply(c("arrow", "DSI", "DSLite", "nnet"),
                    requireNamespace, logical(1), quietly = TRUE)))
