#!/usr/bin/env Rscript
# Canonical channel-B local prediction of one frozen artifact.
args <- commandArgs(TRUE)
stopifnot(length(args) == 3L)
suppressPackageStartupMessages(library(dsFlowerClient))
paths <- jsonlite::fromJSON(args[[2L]])
probability <- ds.flower.predict(args[[1L]], paths, type = "prob")
stopifnot(is.matrix(probability), nrow(probability) == length(paths),
          ncol(probability) == 2L, all(is.finite(probability)))
utils::write.csv(probability, args[[3L]], row.names = FALSE)
