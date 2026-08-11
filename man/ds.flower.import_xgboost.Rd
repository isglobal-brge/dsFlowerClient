\name{ds.flower.import_xgboost}
\alias{ds.flower.import_xgboost}
\title{Import a Bounded External XGBoost JSON Model}
\usage{
ds.flower.import_xgboost(
  artifact,
  model,
  features,
  feature_bounds,
  feature_cuts,
  target,
  target_levels = NULL,
  target_bounds = NULL,
  output_dir
)
}
\arguments{
\item{artifact}{Path to one bounded regular XGBoost JSON model file.}

\item{model}{A valid \code{ds.flower.model.xgboost()} public profile.}

\item{features}{Ordered public feature names.}

\item{feature_bounds}{Public \code{list(lower=..., upper=...)} vectors.}

\item{feature_cuts}{One ordered public cut vector per feature.}

\item{target}{Public target name used by the model schema.}

\item{target_levels}{Two ordered public levels for a binary model.}

\item{target_bounds}{Public \code{list(lower=..., upper=...)} for regression.}

\item{output_dir}{A new destination directory. It is published only after
the complete bundle has passed the trusted parser and prediction probe.}
}
\value{
The absolute path to the imported model directory.
}
\description{
Converts one external XGBoost 3.4.0 JSON model into the same canonical,
data-only ensemble and prediction sidecar used by dsFlower's trusted local
predictor and private validation runner. The importer never loads XGBoost,
pickle, joblib, callbacks, or executable model payloads.
}
\details{
The supplied \code{model} is a public admission profile, not a training
request: its tree count, depth, learning rate and leaf bound must admit the
external artifact. Feature cuts and bounds must exactly describe every
accepted split. The resulting bundle is emitted as
\code{external-unverified}; dsFlower does not claim that the imported model
was trained with differential privacy. A later \code{ds.flower.validate()}
call still releases its pooled metrics through the normal node-DP mechanism.
The external origin is bound into the hashed ensemble contract, but this is
local format provenance rather than a cryptographic signature against the
model-directory owner. External leaf values can contain ordinary non-private
statistics, so the bundle records that unnoised statistics may be present.
}
