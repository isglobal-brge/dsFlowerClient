# Tests for R/utils.R — Client Utilities

test_that(".generate_symbol produces valid symbols", {
  sym <- dsFlowerClient:::.generate_symbol()
  expect_type(sym, "character")
  expect_true(grepl("^dsF\\.", sym))
  expect_equal(nchar(sym), 10) # "dsF." + 6 chars
})

test_that(".generate_symbol uses custom prefix", {
  sym <- dsFlowerClient:::.generate_symbol("myPfx")
  expect_true(startsWith(sym, "myPfx."))
})

test_that("transient capabilities use 128 bits from an OS CSPRNG", {
  tokens <- vapply(c("dsf", "app", "usr"),
                   dsFlowerClient:::.new_capability_token, character(1))
  expect_match(tokens[["dsf"]], "^dsf_[0-9a-f]{32}$")
  expect_match(tokens[["app"]], "^app_[0-9a-f]{32}$")
  expect_match(tokens[["usr"]], "^usr_[0-9a-f]{32}$")
  expect_length(unique(tokens), 3L)
  expect_error(dsFlowerClient:::.new_capability_token("other"), "Unknown")
})

test_that(".ds_encode encodes lists as B64 JSON", {
  encoded <- dsFlowerClient:::.ds_encode(list(a = 1, b = "x"))
  expect_type(encoded, "character")
  expect_true(startsWith(encoded, "B64:"))
})

test_that(".ds_encode passes scalars through", {
  expect_equal(dsFlowerClient:::.ds_encode(42), 42)
  expect_equal(dsFlowerClient:::.ds_encode("hello"), "hello")
  expect_equal(dsFlowerClient:::.ds_encode(TRUE), TRUE)
})

test_that(".ds_encode encodes vectors", {
  encoded <- dsFlowerClient:::.ds_encode(c("a", "b", "c"))
  expect_true(startsWith(encoded, "B64:"))
})

test_that(".format_r_value handles all types", {
  expect_equal(dsFlowerClient:::.format_r_value(NULL), "NULL")
  expect_equal(dsFlowerClient:::.format_r_value("hello"), '"hello"')
  expect_equal(dsFlowerClient:::.format_r_value(42), "42")
  expect_equal(dsFlowerClient:::.format_r_value(TRUE), "TRUE")
  expect_equal(dsFlowerClient:::.format_r_value(5L), "5L")
})

test_that(".build_code generates valid function calls", {
  code <- dsFlowerClient:::.build_code("fn", a = 1, b = "x")
  expect_equal(code, 'fn(a = 1, b = "x")')
})

test_that(".build_code skips NULL arguments", {
  code <- dsFlowerClient:::.build_code("fn", a = 1, b = NULL, c = "y")
  expect_equal(code, 'fn(a = 1, c = "y")')
})

test_that(".require_flwr_cli accepts provisioned client environment", {
  attempted_provision <- FALSE
  local_mocked_bindings(
    .client_venv_is_healthy = function() TRUE,
    .ensure_client_venv = function(...) {
      attempted_provision <<- TRUE
      stop("healthy environments must not be reprovisioned")
    },
    .package = "dsFlowerClient"
  )

  expect_true(isTRUE(dsFlowerClient:::.require_flwr_cli()))
  expect_false(attempted_provision)
})
