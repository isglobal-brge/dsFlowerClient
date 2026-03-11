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

test_that(".require_flwr_cli errors when not found", {
  withr::with_path("", action = "replace", {
    expect_error(dsFlowerClient:::.require_flwr_cli(), "flwr.*not found")
  })
})
