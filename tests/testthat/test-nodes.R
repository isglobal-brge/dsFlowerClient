# Tests for R/nodes.R — Node Orchestration

test_that(".auto_resolve_superlink errors when no SuperLink running", {
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

  mock_proc <- list(is_alive = function() TRUE)
  env$.superlink <- list(
    process = mock_proc, pid = 999,
    fleet_address = "127.0.0.1:9092",
    control_address = "127.0.0.1:9093",
    fleet_port = 9092L, control_port = 9093L,
    serverappio_port = 9091L,
    flwr_home = tempdir(), log_path = tempfile(),
    started_at = Sys.time()
  )
  on.exit(env$.superlink <- old)

  mock_caps <- list(
    opal1 = list(is_docker = TRUE, hostname = "abc123"),
    opal2 = list(is_docker = TRUE, hostname = "def456")
  )

  local_mocked_bindings(
    .ds_safe_aggregate = function(conns, expr) mock_caps,
    .check_node_connectivity = function(conns, srv, address) {
      list(reachable = TRUE, error = NULL)
    }
  )

  result <- suppressMessages(
    dsFlowerClient:::.auto_resolve_superlink(NULL, "flower")
  )
  expect_equal(result, "host.docker.internal:9092")
})

test_that(".auto_resolve_superlink returns per-node addresses for mixed env", {
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old <- env$.superlink

  mock_proc <- list(is_alive = function() TRUE)
  env$.superlink <- list(
    process = mock_proc, pid = 999,
    fleet_address = "127.0.0.1:9092",
    control_address = "127.0.0.1:9093",
    fleet_port = 9092L, control_port = 9093L,
    serverappio_port = 9091L,
    flwr_home = tempdir(), log_path = tempfile(),
    started_at = Sys.time()
  )
  on.exit(env$.superlink <- old)

  mock_caps <- list(
    opal1 = list(is_docker = TRUE, hostname = "abc123"),
    opal2 = list(is_docker = FALSE, hostname = "baremetal1")
  )

  local_mocked_bindings(
    .ds_safe_aggregate = function(conns, expr) mock_caps,
    .detect_local_ip = function() "192.168.1.100",
    .check_node_connectivity = function(conns, srv, address) {
      list(reachable = TRUE, error = NULL)
    }
  )

  result <- suppressMessages(
    dsFlowerClient:::.auto_resolve_superlink(NULL, "flower")
  )
  expect_type(result, "list")
  expect_equal(result$opal1, "host.docker.internal:9092")
  expect_equal(result$opal2, "192.168.1.100:9092")
})

test_that(".auto_resolve_superlink errors when all nodes fail connectivity", {
  env <- getFromNamespace(".dsflower_client_env", "dsFlowerClient")
  old <- env$.superlink

  mock_proc <- list(is_alive = function() TRUE)
  env$.superlink <- list(
    process = mock_proc, pid = 999,
    fleet_address = "127.0.0.1:9092",
    control_address = "127.0.0.1:9093",
    fleet_port = 9092L, control_port = 9093L,
    serverappio_port = 9091L,
    flwr_home = tempdir(), log_path = tempfile(),
    started_at = Sys.time()
  )
  on.exit(env$.superlink <- old)

  mock_caps <- list(
    opal1 = list(is_docker = TRUE, hostname = "abc123")
  )

  local_mocked_bindings(
    .ds_safe_aggregate = function(conns, expr) mock_caps,
    .check_node_connectivity = function(conns, srv, address) {
      list(reachable = FALSE, error = "Connection refused")
    }
  )

  expect_error(
    suppressWarnings(
      dsFlowerClient:::.auto_resolve_superlink(NULL, "flower")
    ),
    "No Opal node can reach the SuperLink"
  )
})

test_that(".detect_local_ip returns a valid IPv4 address", {
  ip <- tryCatch(
    dsFlowerClient:::.detect_local_ip(),
    error = function(e) NULL
  )
  skip_if(is.null(ip), "Could not detect local IP (no network?)")
  expect_type(ip, "character")
  expect_true(grepl("^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$", ip))
})

# --- Federation verification ---

test_that(".verify_federation passes when all IDs match", {
  results <- list(
    opal1 = list(federation_id = "fl-abc123"),
    opal2 = list(federation_id = "fl-abc123")
  )
  expect_silent(dsFlowerClient:::.verify_federation(results, "fl-abc123"))
})

test_that(".verify_federation warns on mismatch", {
  results <- list(
    opal1 = list(federation_id = "fl-abc123"),
    opal2 = list(federation_id = "fl-DIFFERENT")
  )
  expect_warning(
    dsFlowerClient:::.verify_federation(results, "fl-abc123"),
    "Federation ID mismatch"
  )
})

test_that(".verify_federation warns on missing IDs (mixed versions)", {
  results <- list(
    opal1 = list(federation_id = "fl-abc123"),
    opal2 = list(federation_id = NULL)
  )
  expect_warning(
    dsFlowerClient:::.verify_federation(results, "fl-abc123"),
    "did not report a federation_id"
  )
})

test_that(".verify_federation is silent when expected ID is NULL", {
  results <- list(
    opal1 = list(federation_id = "fl-abc123"),
    opal2 = list(federation_id = "fl-DIFFERENT")
  )
  # When researcher didn't start SuperLink via our API, federation_id is NULL
  # -> skip verification entirely
  expect_silent(dsFlowerClient:::.verify_federation(results, NULL))
})
