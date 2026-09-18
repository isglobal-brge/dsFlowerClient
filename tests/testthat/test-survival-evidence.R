test_that("survival evidence schema records failures without claiming campaign completion", {
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
      expect_equal(x$expected_matrix_cells, 90L)
      expect_match(x$envelope_interpretation, "revers")
      if (identical(x$status, "executed")) {
        expected <- c(outer(c("support2/full", "support2/small600", "support2/heterogeneous", "lung1/full"),
                            c("weibull", "lognormal", "hazard"), paste, sep="/"))
        observed <- vapply(x$groups, function(g) paste(g$dataset, g$subset, g$variant, sep="/"), character(1))
        expect_length(observed, 12L)
        expect_setequal(observed, expected)
        for (group in x$groups) {
          expect_identical(group$status, "executed")
          env <- group$envelopes
          expect_match(env$gap_sign, "reversed")
          required <- if (identical(group$subset, "heterogeneous")) "8" else c("1", "4", "8")
          expect_setequal(names(env$summaries), required)
          for (epsilon in required) {
            row <- env$summaries[[epsilon]]
            expect_setequal(names(row$heldout_nll), c("federated_dp", "central", "central_dp", "null"))
            summaries <- c(row[c("federated", "central", "central_dp", "null", "gap")], row$heldout_nll)
            expect_length(summaries, 9L)
            for (summary in summaries) {
              expect_equal(summary$n, 3L)
              expect_true(is.finite(summary$mean))
              expect_true(is.finite(summary$sd) && summary$sd >= 0)
              expect_length(summary$ci95, 2L)
              expect_equal(unlist(summary$ci95), summary$mean + c(-1,1)*stats::qt(.975,2)*summary$sd/sqrt(3))
              expect_match(summary$method, "Student-t")
            }
          }
          expect_length(env$epsilon_envelope, max(0L, length(required)-1L))
          expect_length(env$near_central, 3L * length(required))
          for (epsilon in as.numeric(required)) {
            rows <- Filter(function(z) identical(z$epsilon, epsilon) || isTRUE(z$epsilon == epsilon), env$near_central)
            expect_setequal(vapply(rows, function(z) z$seed, numeric(1)), c(1101,1102,1103))
          }
          primary <- identical(group$dataset, "support2") && identical(group$subset, "full")
          if (primary) {
            expect_type(env$utility_floor$pass, "logical")
            expect_equal(env$utility_floor$absolute_floor, .60)
            expect_equal(env$utility_floor$null_margin, .05)
          } else expect_null(env$utility_floor)
          # A package schema check permits pending reviews. The separate Python
          # completion gate requires reviewed explanations and all 93 cells.
          flags <- vapply(Filter(function(z) isTRUE(z$flag), env$epsilon_envelope),
                          function(z) paste("epsilon_envelope", z$from, z$to, sep="_"), character(1))
          if (isTRUE(env$small_n_trend$flag)) flags <- c(flags, "small_n_trend")
          flags <- c(flags, vapply(Filter(function(z) isTRUE(z$historic_flag) || isTRUE(z$minimum_site_companion_flag),
                                         env$near_central), function(z) paste("near_central", z$epsilon, z$seed, sep="_"), character(1)))
          for (flag in flags) {
            expect_true(group$investigations[[flag]]$status %in% c("pending", "reviewed"))
            expect_true(is.character(group$investigations[[flag]]$explanation) && nzchar(group$investigations[[flag]]$explanation))
          }
        }
      }
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
      expect_type(x$cleanup_ok, "logical")
      next
    }
    expect_identical(x$record_type, "cell")
    expect_identical(x$cleanup_ok, TRUE)
    for (key in c("started_utc", "finished_utc")) expect_match(x[[key]], "(Z|\\+00:00)$")
    expect_gt(x$elapsed_s, 0)
    expect_match(x$campaign_tools_commit, "^[a-f0-9]{40}$")
    for (repo in c("dsFlower", "dsFlowerClient")) expect_match(x$package_commits[[repo]], "^[a-f0-9]{40}$")
    expect_match(x$runner_sha256, "^[a-f0-9]{64}$")
    expect_identical(x$installed_build$commits, x$package_commits)
    expect_identical(x$installed_build$runner_sha256, x$runner_sha256)
    for (name in c("dsFlower", "dsFlowerClient", "R", "DSI", "DSLite", "dsBase", "resourcer")) {
      expect_true(is.character(x$package_versions[[name]]) && nzchar(x$package_versions[[name]]))
    }
    for (field in c("release", "licence")) expect_true(nzchar(x$dataset$source[[field]]))
    if (!identical(x$dataset$dataset, "synthetic-public")) {
      for (field in c("url", "licence_url", "attribution")) expect_true(nzchar(x$dataset$source[[field]]))
      expect_true(nzchar(x$dataset$subject_provenance))
      expect_true(nzchar(x$dataset$split_rule))
    }
    for (key in c("train_sha256", "test_sha256")) expect_match(x$dataset[[key]], "^[a-f0-9]{64}$")
    expect_identical(unlist(x$outcome_semantics$target_order), c("time", "event"))
    expect_equal(x$outcome_semantics$event, 1)
    expect_equal(x$outcome_semantics$censored, 0)
    for (key in c("time_unit", "baseline", "administrative_censor", "invalid", "preprocessing", "interval_convention")) {
      expect_true(is.character(x$outcome_semantics[[key]]) && nzchar(x$outcome_semantics[[key]]))
    }
    expect_identical(x$score_conventions$time_ties, "excluded")
    expect_equal(x$score_conventions$risk_ties, .5)
    expect_match(x$score_conventions$gap, "reversed")
    expect_identical(x$evaluation, "channel B, public held-out subjects only")
    expect_true(nzchar(x$mechanism_provenance))
    expect_match(x$artifact_checksum, "^[a-f0-9]{64}$")
    expect_identical(x$artifact_checksum, x$results$model_sha256)
    expect_identical(x$artifact_checksum, x$federation$model_sha256)
    expect_equal(x$federation$n_clients, 3)
    expect_equal(x$federation$n_failures, 0)
    expect_equal(x$federation$n_rounds_run, x$public_config[["num-server-rounds"]])
    expect_identical(x$federation$cleanup_ok, TRUE)
    expect_true(all(c("central", "central_dp", "null", "federated_dp", "site_mechanisms", "versions") %in% names(x$results)))
    for (branch in c("central", "central_dp", "null", "federated_dp")) {
      score <- x$results[[branch]]
      expect_true(is.numeric(score$c_index) && is.finite(score$c_index))
      expect_gte(score$c_index, 0)
      expect_lte(score$c_index, 1)
      expect_true(is.numeric(score$heldout_nll) && is.finite(score$heldout_nll))
      expect_gt(score$n_evaluated_public_subjects, 0)
      expect_gte(score$n_invalid_public_subjects, 0)
      expect_equal(score$n_evaluated_public_subjects + score$n_invalid_public_subjects, x$dataset$n_test)
    }
    expect_equal(x$results$seed, x$dataset$seed)
    expect_equal(x$results$n_train, x$dataset$n_train)
    expect_length(x$dataset$sites, 3L)
    expect_equal(sum(vapply(x$dataset$sites, function(s) s$n_subjects, numeric(1))), x$dataset$n_train)
    expect_equal(x$results$minimum_site_n, min(vapply(x$dataset$sites, function(s) s$n_subjects, numeric(1))))
    expect_length(x$results$site_mechanisms, 3L)
    expect_setequal(vapply(x$results$site_mechanisms, function(s) s$site, numeric(1)), 1:3)
    for (site in c(x$results$site_mechanisms, list(x$results$pooled_mechanism))) {
      n <- if (is.null(site$site)) x$dataset$n_train else {
        census <- Filter(function(s) isTRUE(s$site == site$site), x$dataset$sites)[[1L]]
        expect_equal(census$n_subjects, census$source_rows)
        expect_match(census$split_sha256, "^[a-f0-9]{64}$")
        census$n_subjects
      }
      expect_equal(site$accounting_population, n)
      expect_identical(site$adjacency, "replace_one")
      expect_equal(site$clipping_norm, 1)
      expect_gt(site$noise_multiplier, 0)
      expect_equal(site$steps_per_epoch, ceiling(n/x$public_config[["batch-size"]]))
      expect_equal(site$sample_rate, 1/site$steps_per_epoch)
      expect_equal(site$expected_batch_size, max(1, floor(n/site$steps_per_epoch)))
      expect_equal(site$total_epochs, x$public_config[["num-server-rounds"]] * x$public_config[["local-epochs"]])
      expect_equal(site$total_steps, site$steps_per_epoch * site$total_epochs)
      expect_match(site$policy_hash, "^[a-f0-9]{64}$")
      expect_identical(site$verification_accountant, "PRV")
      expect_lte(site$independently_recomputed_replace_one_epsilon, x$epsilon+1e-8)
      expect_lte(site$independently_recomputed_replace_one_delta, x$delta+1e-12)
    }
    for (name in c("python", "torch", "opacus", "flwr", "numpy", "scipy", "pandas", "platform",
                   "nonprivate_twin_device", "dp_twin_device", "federation_device_rule")) {
      expect_true(is.character(x$results$versions[[name]]) && nzchar(x$results$versions[[name]]))
    }
    expect_identical(x$results$versions$deterministic_algorithms, TRUE)
    expect_identical(x$results$twin_matching$architecture_loss_preprocessing_initialization_optimizer_schedule, "exact")
    expect_equal(x$results$twin_matching$initialization_seed, 0)
    expect_equal(x$results$twin_matching$pooled_epochs, x$public_config[["num-server-rounds"]] * x$public_config[["local-epochs"]])
    expect_true(nzchar(x$results$twin_matching$differences))
    expect_gt(x$results$elapsed_s, 0)
  }
})
