# Tests for R/nodes.R — Node Orchestration

test_that(".auto_resolve_superlink errors when no SuperLink running", {
  # Ensure no SuperLink is tracked
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old <- env$.superlink
  env$.superlink <- NULL
  on.exit(env$.superlink <- old)

  expect_error(
    dsFlowerClient:::.auto_resolve_superlink(NULL, "flower"),
    "No SuperLink running"
  )
})

test_that(".auto_resolve_superlink returns Docker address for Docker nodes", {
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old <- env$.superlink

  # Mock a running SuperLink
  mock_proc <- list(is_alive = function() TRUE)
  env$.superlink <- list(
    process = mock_proc,
    pid = 999,
    fleet_address = "127.0.0.1:9092",
    control_address = "127.0.0.1:9093",
    fleet_port = 9092L,
    control_port = 9093L,
    serverappio_port = 9091L,
    flwr_home = tempdir(),
    log_path = tempfile(),
    started_at = Sys.time()
  )
  on.exit(env$.superlink <- old)

  # Mock .ds_safe_aggregate to return Docker capabilities
  mock_caps <- list(
    opal1 = list(is_docker = TRUE, hostname = "abc123"),
    opal2 = list(is_docker = TRUE, hostname = "def456")
  )

  local_mocked_bindings(
    .ds_safe_aggregate = function(conns, expr) mock_caps
  )

  result <- dsFlowerClient:::.auto_resolve_superlink(NULL, "flower")
  expect_equal(result, "host.docker.internal:9092")
})

test_that(".auto_resolve_superlink returns per-node addresses for mixed env", {
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old <- env$.superlink

  mock_proc <- list(is_alive = function() TRUE)
  env$.superlink <- list(
    process = mock_proc,
    pid = 999,
    fleet_address = "127.0.0.1:9092",
    control_address = "127.0.0.1:9093",
    fleet_port = 9092L,
    control_port = 9093L,
    serverappio_port = 9091L,
    flwr_home = tempdir(),
    log_path = tempfile(),
    started_at = Sys.time()
  )
  on.exit(env$.superlink <- old)

  # Mixed: one Docker, one bare-metal
  mock_caps <- list(
    opal1 = list(is_docker = TRUE, hostname = "abc123"),
    opal2 = list(is_docker = FALSE, hostname = "baremetal1")
  )

  local_mocked_bindings(
    .ds_safe_aggregate = function(conns, expr) mock_caps,
    .detect_local_ip = function() "192.168.1.100"
  )

  result <- dsFlowerClient:::.auto_resolve_superlink(NULL, "flower")
  expect_type(result, "list")
  expect_equal(result$opal1, "host.docker.internal:9092")
  expect_equal(result$opal2, "192.168.1.100:9092")
})
