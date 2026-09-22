"""Reuse the logistic CDC cohort and splits without examining held-out values."""
import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile
import urllib.request


# Evaluate only these existing preparation functions, avoiding federation setup.
# Split an index-only frame by the original diabetes strata before routing BMI.
PREPARE_R = r'''
args <- commandArgs(TRUE)
p <- jsonlite::fromJSON(args[1])
cache <- args[3]
out <- args[4]
original <- new.env(parent = globalenv())
keep <- c("CAMPAIGN_COHORT_SEED", "campaign_load_cohort", "campaign_split")
for (expression in parse(args[2])) {
  if (is.call(expression) && identical(expression[[1]], as.name("<-")) &&
      is.symbol(expression[[2]]) && as.character(expression[[2]]) %in% keep) {
    eval(expression, envir = original)
  }
}
stopifnot(all(vapply(keep, exists, logical(1), envir = original,
                    inherits = FALSE)),
          original$CAMPAIGN_COHORT_SEED == p$dataset$cohort_seed)
source_path <- file.path(cache, "cdc_diabetes_health_indicators.csv")
stopifnot(file.copy(source_path, file.path(out, basename(source_path))))
cohort <- original$campaign_load_cohort(p$dataset$cohort, out)$df
prepared <- file.path(out, paste0("prepared_", p$dataset$cohort, ".csv"))
prepared_sha <- digest::digest(file = prepared, algo = "sha256")
stopifnot(identical(prepared_sha, p$dataset$logistic_prepared_sha256),
          nrow(cohort) == p$dataset$n_total)

# Preserve source identities using the exact fixed-cohort sampler.
full <- utils::read.csv(source_path)
labels <- as.integer(full$Diabetes_binary)
set.seed(original$CAMPAIGN_COHORT_SEED)
idx <- unlist(lapply(split(seq_len(nrow(full)), labels), function(ix) {
  take <- round(p$dataset$n_total * length(ix) / nrow(full))
  sample(ix, take)
}), use.names = FALSE)
idx <- sort(idx)
if (length(idx) > p$dataset$n_total) idx <- idx[seq_len(p$dataset$n_total)]
stopifnot(length(idx) == nrow(cohort), !anyDuplicated(idx),
          identical(labels[idx], cohort$target),
          identical(as.numeric(as.matrix(full[idx, c(p$features, p$target)])),
                    as.numeric(as.matrix(cohort[, c(p$features, p$target)]))),
          !anyDuplicated(full$ID[idx]))
rows <- data.frame(cohort_index = seq_len(nrow(cohort)), source_row = idx,
                   source_id = full$ID[idx], target = cohort$target)
model_data <- cohort[, c(p$features, p$target), drop = FALSE]
stopifnot(!any(c("ID", "Diabetes_binary", "target") %in% names(model_data)))
splits <- list()
for (seed in p$seeds) {
  split <- original$campaign_split(rows, seed, p$sites)
  train_idx <- split$train$cohort_index
  test_idx <- split$test$cohort_index
  site_idx <- lapply(split$sites, function(x) x$cohort_index)
  stopifnot(!length(intersect(train_idx, test_idx)),
            !anyDuplicated(unlist(site_idx)),
            setequal(unlist(site_idx), train_idx),
            setequal(c(train_idx, test_idx), seq_len(nrow(cohort))),
            length(train_idx) == p$split$n_train,
            length(test_idx) == p$split$n_test)
  folder <- file.path(out, paste0("cdcbmi_seed", seed))
  dir.create(folder)
  write.csv(model_data[train_idx, , drop = FALSE], file.path(folder, "train.csv"),
            row.names = FALSE)
  write.csv(model_data[test_idx, , drop = FALSE], file.path(folder, "test.csv"),
            row.names = FALSE)
  for (site in seq_len(p$sites)) {
    write.csv(model_data[site_idx[[site]], , drop = FALSE],
              file.path(folder, paste0("site", site, ".csv")), row.names = FALSE)
  }
  identity <- function(frame) list(cohort_indices = frame$cohort_index,
    source_rows = frame$source_row, uci_ids = frame$source_id)
  metadata <- list(seed = seed, index_base = 1L,
    n_train = length(train_idx), n_test = length(test_idx),
    n_per_site = lengths(site_idx), stratification = "Diabetes_binary",
    train = identity(split$train), test = identity(split$test),
    sites = lapply(split$sites, identity))
  jsonlite::write_json(metadata, file.path(folder, "split.json"),
                       auto_unbox = TRUE, pretty = TRUE, digits = NA)
  splits[[as.character(seed)]] <- metadata[c("seed", "n_train", "n_test", "n_per_site")]
}
jsonlite::write_json(list(n_source = nrow(full), n_total = nrow(cohort),
  cohort_seed = original$CAMPAIGN_COHORT_SEED,
  logistic_prepared_sha256 = prepared_sha,
  r_version = R.version.string, rng_kind = RNGkind(), splits = splits),
  file.path(out, "preparation.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
'''


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/cells")
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    script_dir = pathlib.Path(__file__).resolve().parent
    protocol_path = script_dir / "cdcbmi_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    campaign_lib = script_dir.parent / "campaign_lib.R"
    cache = pathlib.Path(args.root).resolve() / "data_cache"
    cache.mkdir(parents=True, exist_ok=True)
    provenance_path = cache / "cdcbmi_provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance["protocol_sha256"] != sha256(protocol_path):
            raise RuntimeError("Prepared CDC BMI protocol changed; refusing reuse")
        for relative, checksum in provenance["checksums"].items():
            if sha256(cache / relative) != checksum:
                raise RuntimeError("Prepared CDC BMI checksum mismatch: " + relative)
        print(json.dumps(provenance, indent=2))
        return
    folders = [f"cdcbmi_seed{seed}" for seed in protocol["seeds"]]
    if any((cache / folder).exists() for folder in folders):
        raise RuntimeError("Partial CDC BMI preparation exists; refusing overwrite")
    source = cache / "cdc_diabetes_health_indicators.csv"
    if not source.exists():
        with urllib.request.urlopen(protocol["dataset"]["source_url"], timeout=120) as response:
            with source.with_suffix(".download").open("wb") as output:
                shutil.copyfileobj(response, output)
        source.with_suffix(".download").replace(source)
    if sha256(source) != protocol["dataset"]["source_sha256"]:
        raise RuntimeError("CDC source checksum differs from the frozen source")

    with tempfile.TemporaryDirectory(prefix="cdcbmi_prep_", dir=cache) as temp:
        temporary = pathlib.Path(temp)
        r_script = temporary / "prepare.R"
        r_script.write_text(PREPARE_R)
        subprocess.run([args.rscript, "--vanilla", str(r_script),
                        str(protocol_path), str(campaign_lib), str(cache), temp],
                       check=True)
        preparation = json.loads((temporary / "preparation.json").read_text())
        for folder in folders:
            shutil.move(str(temporary / folder), str(cache / folder))

    files = [source] + [path for folder in folders
                        for path in sorted((cache / folder).iterdir())]
    checksums = {str(path.relative_to(cache)): sha256(path) for path in files}
    provenance = dict(protocol["dataset"], **preparation,
                      prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      protocol_sha256=sha256(protocol_path),
                      loader_sha256=sha256(pathlib.Path(__file__).resolve()),
                      logistic_campaign_lib_sha256=sha256(campaign_lib),
                      checksums=checksums,
                      heldout_policy="Preparation only routes rows; no held-out BMI summaries, model fitting, predictions, metrics, or setting selection")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    (cache / "cdcbmi_CHECKSUMS.sha256").write_text(
        "".join(f"{value}  {name}\n" for name, value in checksums.items()))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
