test_that("new private layouts accept only their fixed pooled fields", {
  valid <- dsFlowerClient:::.private_metrics_valid
  segmentation <- list(n = 12, foreground_dice = .7)
  expect_true(valid(segmentation, "segmentation"))
  expect_false(valid(c(segmentation, list(per_patient = 1)), "segmentation"))
  expect_false(valid(list(n = 12, foreground_dice = 1.1), "segmentation"))
  survival <- list(n = 12, negative_log_likelihood = -.4,
    brier = list(horizons = list(2, 4), scores = list(.1, NULL), eligible_n = list(10, 0)),
    brier_method = "observed-status")
  expect_true(valid(survival, "survival"))
  expect_false(valid(c(survival, list(concordance = .7)), "survival"))
  survival$brier$horizons <- list(4, 2)
  expect_false(valid(survival, "survival"))
  expect_identical(dsFlowerClient:::.score_metric_registry("segmentation"),
                   c(foreground_dice = "maximize"))
  expect_identical(dsFlowerClient:::.score_metric_registry("survival"),
                   c(negative_log_likelihood = "minimize"))
})

test_that("survival metric geometry is public and validates before transport", {
  sub <- dsFlowerClient:::.emit_submission(ds.flower.model.pytorch_aft(10))
  config <- dsFlowerClient:::.survival_config(sub$params, sub$loss)
  normalize <- dsFlowerClient:::.private_survival_metric_config
  expect_identical(normalize(config)[["validation-survival-horizons"]], "[10]")
  expect_equal(normalize(config, c(1, 5, 10), 7)[["validation-survival-nll-bound"]], 7)
  for (horizons in list(c(5, 1), c(1, 1), 0, 11, c(1, Inf), "5")) {
    expect_error(normalize(config, horizons), "survival_horizons")
  }
  for (bound in list(0, -1, Inf, TRUE, c(1, 2))) {
    expect_error(normalize(config, nll_bound = bound), "survival_nll_bound")
  }
  expect_error(normalize(NULL, 2), "require a survival")
})

test_that("CV forwards admitted material and infers segmentation input kind", {
  calls <- list()
  local_mocked_bindings(ds.flower.fit = function(...) {
    calls[[length(calls) + 1L]] <<- list(...)
    structure(list(), class = "dsflower_cv")
  }, .package = "dsFlowerClient")
  ds.flower.cross_validate(list(site = TRUE), target = "y", features = "x",
    public_initialisation = "client:/local/bundle.zip", folds = 2, rounds = 2)
  ds.flower.cross_validate(list(site = TRUE), target = "mask_path",
    model = ds.flower.model.pytorch_resnet18_segmentation(decoder_init = "resource:CKPT"),
    public_checkpoint_file = "/local/checkpoint.npz", folds = 2, rounds = 2)
  expect_identical(calls[[1L]]$public_initialisation, "client:/local/bundle.zip")
  expect_identical(calls[[2L]]$public_checkpoint_file, "/local/checkpoint.npz")
  expect_null(calls[[2L]]$data_kind)
  expect_equal(calls[[2L]]$cross_validation, 2)
})

test_that("initialization identity changes the job but not the private partition contract", {
  partition <- dsFlowerClient:::.cross_validation_contract(list(folds = 2L), "patient")
  summary <- list(provenance = list(manifest_sha256 = strrep("a", 64)),
    checkpoint_sha256 = strrep("b", 64), encoder_sha256 = strrep("c", 64),
    identity_version = "dsflower-public-initialisation-identity/v1")
  before <- dsFlowerClient:::.validationCvContractExtension(
    dsFlowerClient:::.public_initialisation_identity(summary))
  summary$checkpoint_sha256 <- strrep("d", 64)
  after <- dsFlowerClient:::.validationCvContractExtension(
    dsFlowerClient:::.public_initialisation_identity(summary))
  expect_false(identical(before, after))
  expect_identical(partition,
    dsFlowerClient:::.cross_validation_contract(list(folds = 2L), "patient"))
})

test_that("validation carries the effective parameter of each fitted loss", {
  config <- dsFlowerClient:::.validation_loss_config
  for (case in list(c("negbin_nll", "nb-dispersion", "nb_dispersion"),
                    c("gamma_nll", "gamma-shape", "gamma_shape"),
                    c("huber", "huber-delta", "huber_delta"),
                    c("quantile", "quantile-level", "quantile"))) {
    default <- if (case[[1L]] == "quantile") .5 else 1
    changed <- if (case[[1L]] == "quantile") .75 else 2
    expect_identical(config(case[[1L]], list()), setNames(list(default), case[[2L]]))
    expect_identical(config(case[[1L]], setNames(list(changed), case[[2L]]), wire = TRUE),
                     setNames(list(changed), case[[2L]]))
    expect_identical(config(case[[1L]], setNames(list(changed), case[[3L]])),
                     setNames(list(changed), case[[2L]]))
    expect_error(config(case[[1L]], setNames(list(0), case[[3L]])), "trusted contract")
  }
})

test_that("private segmentation and survival validation retain the HPO boundary", {
  for (task in c("segmentation", "survival")) {
    local_mocked_bindings(.resolve_validation_contract = function(...) list(task = task),
      .validate_validation_artifact_preflight = function(...) stop("must not preflight"),
      .package = "dsFlowerClient")
    expect_error(dsFlowerClient:::.hpo_evaluate_objective(function(params) {
      ds.flower.validate(list(), "model", target = "y")
    }, list()), "HPO are unsupported")
    expect_false(dsFlowerClient:::.DSFLOWER_HPO_CONTEXT$active)
  }
})
