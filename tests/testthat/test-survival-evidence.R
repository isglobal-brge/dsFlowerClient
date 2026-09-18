test_that("survival evidence distinguishes execution from failure without placeholder scores", {
  path <- testthat::test_path("..", "..", "inst", "extdata", "campaign", "survival")
  if (!dir.exists(path)) path <- system.file("extdata", "campaign", "survival", package="dsFlowerClient")
  files <- list.files(path, pattern="\\.json$", full.names=TRUE)
  expect_gt(length(files), 0L)
  for (file in files) {
    x <- jsonlite::read_json(file, simplifyVector=FALSE)
    expect_equal(x$schema_version, 1L, info=basename(file))
    expect_identical(x$task, "survival")
    if (identical(x$record_type, "summary")) {
      expect_true(x$status %in% c("executed", "incomplete"))
      next
    }
    expect_true(x$status %in% c("executed", "failed"))
    expect_identical(x$dataset$public_fixture, TRUE)
    expect_equal(x$delta, 1e-5)
    expect_equal(x$clip, 1)
    expect_identical(x$privacy_unit, "patient")
    expect_identical(x$adjacency, "replace_one")
    expect_equal(x$site_count, 3)
    expect_match(x$dataset$source$sha256, "^[a-f0-9]{64}$")
    expect_match(x$dataset$protocol_sha256, "^[a-f0-9]{64}$")
    if (identical(x$status, "failed")) {
      expect_true(is.character(x$error) && nzchar(x$error))
      expect_null(x$results)
      next
    }
    expect_identical(x$cleanup_ok, TRUE)
    expect_match(x$artifact_checksum, "^[a-f0-9]{64}$")
    expect_true(all(c("central", "central_dp", "null", "federated_dp", "site_mechanisms", "versions") %in% names(x$results)))
    for (branch in c("central", "central_dp", "null", "federated_dp")) {
      score <- x$results[[branch]]
      expect_true(is.numeric(score$c_index) && is.finite(score$c_index))
      expect_gte(score$c_index, 0)
      expect_lte(score$c_index, 1)
      expect_true(is.numeric(score$heldout_nll) && is.finite(score$heldout_nll))
    }
    expect_length(x$results$site_mechanisms, 3L)
    for (site in x$results$site_mechanisms) {
      expect_gt(site$noise_multiplier, 0)
      expect_equal(site$sample_rate, 1/ceiling(site$accounting_population/128))
      expect_equal(site$total_steps, site$steps_per_epoch * site$total_epochs)
      expect_lte(site$independently_recomputed_replace_one_epsilon, x$epsilon+1e-8)
      expect_lte(site$independently_recomputed_replace_one_delta, x$delta+1e-12)
    }
  }
})
