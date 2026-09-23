test_that("analysts cannot supply cache or deadline controls", {
  controls <- c("release_cache_dir", "release-cache-bytes", "releaseCacheBytes",
                "cache", "hook-deadline", "deadlineSeconds")
  for (key in controls) {
    config <- setNames(list(1024), key)
    expect_error(ds.flower.nodes.prepare(
      conns = list(), target_column = "y", run_config = config),
      "node administrator")
    expect_error(ds.flower.run.start(NULL, run_config = config),
                 "node administrator")
    expect_error(dsFlowerClient:::.canonical_hook_app_params(list(nested = config)),
                 "reserved")
  }
  expect_silent(dsFlowerClient:::.reject_release_cache_controls(
    list("num-server-rounds" = 2L)))
})
