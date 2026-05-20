extdata_evidence <- function(file) {
  candidates <- c(
    testthat::test_path("..", "..", "inst", "extdata", file),
    system.file("extdata", file, package = "dsFlowerClient")
  )
  candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
  expect_gt(length(candidates), 0)
  candidates[[1]]
}

test_that("committed method validation evidence matches thesis constants", {
  path <- extdata_evidence("dsflower_method_validation_results.json")
  evidence <- jsonlite::fromJSON(path)

  expect_equal(nrow(evidence$results), 17)
  expect_true(all(evidence$results$validation_status == "pass"))
  expect_true(all(is.na(evidence$results$federated_n_failures) |
                    evidence$results$federated_n_failures == 0))

  row <- evidence$results[evidence$results$method == "sklearn_svm", ]
  expect_equal(row$centralized_loss, 0.8285259, tolerance = 1e-6)
  expect_equal(row$federated_loss, 1.0934755, tolerance = 1e-6)
  expect_equal(row$delta_loss, 0.2649496, tolerance = 1e-6)

  row <- evidence$results[evidence$results$method == "pytorch_tcn", ]
  expect_equal(row$federated_loss, 0.6795301, tolerance = 1e-6)
  expect_equal(row$delta_loss, 0.0136913, tolerance = 1e-6)
})

test_that("committed vision validation evidence matches thesis constants", {
  path <- extdata_evidence("dsflower_vision_validation_results.json")
  evidence <- jsonlite::fromJSON(path)

  expect_equal(nrow(evidence$results), 3)
  expect_true(all(evidence$results$validation_status == "pass"))
  expect_true(all(evidence$results$federated_n_failures == 0))

  resnet <- evidence$results[evidence$results$method == "pytorch_resnet18", ]
  expect_equal(resnet$centralized_loss, 0.5605, tolerance = 1e-6)
  expect_equal(resnet$federated_loss, 0.5399, tolerance = 1e-6)
  expect_equal(resnet$delta_loss, -0.0206, tolerance = 1e-6)

  unet <- evidence$results[evidence$results$method == "pytorch_unet2d", ]
  expect_equal(unet$federated_loss, 1.4397, tolerance = 1e-6)
  expect_equal(unet$delta_loss, 0.1298, tolerance = 1e-6)
})

test_that("committed LUNG1 radiomics evidence matches thesis constants", {
  path <- extdata_evidence("dsflower_lung1_radiomics_results.json")
  evidence <- jsonlite::fromJSON(path, simplifyVector = FALSE)

  expect_equal(evidence$resource, "dsdemo.lung1_full_study")
  expect_equal(evidence$target, "os_2yr_alive")
  expect_equal(unlist(evidence$dimensions[[1]]), c(142, 20))
  expect_equal(unlist(evidence$dimensions[[2]]), c(143, 20))
  expect_equal(unlist(evidence$dimensions[[3]]), c(137, 20))
  expect_equal(unlist(evidence$dimensions[[4]]), c(422, 20))
  expect_equal(length(evidence$features), 6)
  expect_equal(vapply(evidence$history, `[[`, numeric(1), "loss"),
               c(0.6503, 0.6474), tolerance = 1e-6)
  expect_equal(sum(vapply(evidence$history, `[[`, numeric(1), "n_failures")), 0)
})

test_that("committed LUNG1 direct-image evidence matches thesis constants", {
  path <- extdata_evidence("dsflower_lung1_direct_image_results.json")
  evidence <- jsonlite::fromJSON(path, simplifyVector = FALSE)

  expect_equal(evidence$resource, "dsdemo.lung1_study")
  expect_equal(evidence$model, "pytorch_resnet18")
  expect_equal(evidence$rounds, 1)
  expect_equal(sum(as.integer(unlist(evidence$metadata_n))), 9)
  expect_equal(unname(as.integer(unlist(evidence$metadata_n))), c(3L, 3L, 3L))
  expect_equal(evidence$local_baseline$eval$loss, 0.554012457529704,
               tolerance = 1e-12)
  expect_equal(evidence$local_baseline$eval$accuracy, 0.777777777777778,
               tolerance = 1e-12)
  expect_equal(evidence$local_vs_federated$federated$loss, 0.584744469987022,
               tolerance = 1e-12)
  expect_equal(
    evidence$local_vs_federated$delta$loss_federated_minus_centralized,
    0.0307320124573178,
    tolerance = 1e-12
  )
  expect_equal(evidence$local_vs_federated$federated$n_failures, 0)
  expect_true(all(as.integer(unlist(evidence$active_supernodes_after)) == 0))
})
