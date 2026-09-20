#!/usr/bin/env Rscript
# Exercise client-generated configuration against the paired server validator.
# Usage from the paired workspace: Rscript PATH_TO_THIS_SCRIPT dsFlower dsFlowerClient
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
pkgload::load_all(args[[1L]], quiet = TRUE, helpers = FALSE)
pkgload::load_all(args[[2L]], quiet = TRUE, helpers = FALSE)

testthat::test_that("segmentation client preparation satisfies paired server authority", {
  withr::local_options(list(dsflower.dp_unit = "patient",
                           dsflower.patient_column = "subject"))
  accepted <- NULL
  app_config <- NULL
  testthat::local_mocked_bindings(
    .require_flwr_cli = function(...) TRUE,
    .validate_dsi_transport_security = function(...) TRUE,
    .validate_declarative_model_preflight = function(...) TRUE,
    ds.flower.connect = function(conns, ...) list(conns = conns, symbol = "flower"),
    .assert_runner_compatibility = function(...) list(),
    ds.flower.nodes.prepare = function(conns, symbol, target_column,
                                        feature_columns, run_config) {
      dsFlower:::.validateSegmentationColumns(run_config, target_column, feature_columns)
      accepted <<- dsFlower:::.addDpConfigToRunConfig(run_config)
      invisible(NULL)
    },
    .build_submission_app = function(sub, cfg, results_dir, ...) {
      app_config <<- cfg
      stop("captured paired app build")
    },
    ds.flower.link.down = function(...) NULL,
    ds.flower.nodes.cleanup = function(...) NULL,
    .dsflower_disconnect_on_exit = function(...) NULL,
    .package = "dsFlowerClient")
  testthat::expect_error(ds.flower.submit(
    conns = list(site = TRUE), model = "pytorch_resnet18_segmentation",
    symbol = "D", target = "mask_path", data_kind = "image",
    model_params = list(mask_empty_col = "normal", subject_id_col = "subject")),
    "captured paired app build")
  testthat::expect_true(accepted$dp_enabled)
  testthat::expect_identical(accepted[["task-type"]], "segmentation")
  testthat::expect_false("num-labels" %in% names(accepted))
  testthat::expect_false(any(grepl("^num-labels\\s*=", app_config)))
})
