#!/usr/bin/env Rscript
# Check actual dsImaging admission on a training collection; disclose no labels.
args <- commandArgs(TRUE)
stopifnot(length(args) == 1L)
root <- normalizePath(args[[1]])
suppressPackageStartupMessages({library(DSI); library(DSLite); library(dsImaging)})
collection <- file.path(root, "prepared/vision/20260919/site1")
options(dsimaging.registry_path = file.path(collection, "registry.yaml"))
resource <- resourcer::newResource(name = "images", url = "imaging+dataset://busbra.site1")
server <- DSLite::newDSLiteServer(resources = list(images = resource),
  config = DSLite::defaultDSConfiguration(include = "dsImaging"))
assign("vision_admission", server, .GlobalEnv)
conn <- DSLite::dsConnect(DSLite::DSLite(), name = "site1", url = "vision_admission")
DSI::dsFetch(DSI::dsAssignResource(conn, "resource", "images", async = FALSE))
DSI::dsFetch(DSI::dsAssignExpr(conn, "D", quote(imagingInitDS("resource")), async = FALSE))
handle <- server$getSessionData(conn@sid, "D")
stopifnot(is.list(handle), is.character(handle$capability))
DSI::dsDisconnect(conn)
cat("DSIMAGING_ADMISSION_PASSED\n")
