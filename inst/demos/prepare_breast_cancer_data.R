# Prepare the federated breast-cancer demo data.
# Splits the Breast Cancer Wisconsin dataset into 3 stratified
# partitions and uploads one to each Opal node as table
# dsflower_demo.breast_cancer. Run once by an operator with Opal admin.
#
# The partitions (~190 patients each, ~71 malignant / ~119 benign) satisfy the
# clinical_default trust-profile thresholds (>=100 rows, >=20 per class).

suppressMessages(library(opalr))

nodes <- c("nairobi", "dakar", "douala")
csvs  <- c("bc_part1.csv", "bc_part2.csv", "bc_part3.csv")  # Breast Cancer Wisconsin, stratified 3-way

for (i in seq_along(nodes)) {
  h  <- nodes[i]
  df <- read.csv(csvs[i])
  df$id <- as.character(df$id)              # entity identifier column

  o <- opal.login("administrator", "password",
                  url = paste0("https://", h, ".datashield.live"))
  if (!isTRUE(tryCatch(opal.project_exists(o, "dsflower_demo"), error = function(e) FALSE))) {
    opal.project_create(o, "dsflower_demo", database = "mongodb")
  }
  opal.table_save(o, df, project = "dsflower_demo", table = "breast_cancer",
                  overwrite = TRUE, force = TRUE, id.name = "id")
  message(h, ": uploaded ", nrow(df), " patients")
  opal.logout(o)
}
