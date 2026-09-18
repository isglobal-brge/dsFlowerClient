#!/usr/bin/env Rscript
# Public survival experiment; privacy policy is set only inside custodian workers.
args <- commandArgs(TRUE)
if (length(args) != 7L) stop('Usage: run_cell.R workspace split variant epsilon out rounds epochs')
workspace <- normalizePath(args[[1]])
split_dir <- normalizePath(args[[2]])
variant <- match.arg(args[[3]], c('weibull', 'lognormal', 'hazard'))
epsilon <- as.numeric(args[[4]])
out <- normalizePath(args[[5]], mustWork = FALSE)
rounds <- as.integer(args[[6]])
epochs <- as.integer(args[[7]])
.libPaths(c(file.path(workspace, 'runtime', 'rlib'), .libPaths()))
Sys.setenv(DSFLOWER_CLIENT_VENV_ROOT=file.path(workspace,'runtime'),
           DSFLOWER_VENV_ROOT=file.path(workspace,'runtime','server'),
           DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET='1',
           UV_CACHE_DIR=file.path(workspace,'runtime','uv-cache'),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
tools_dir <- file.path(workspace,'dsFlowerClient','tools','campaign')
source(file.path(tools_dir,'campaign_lib.R'))
python <- file.path(workspace,'runtime','venv','bin','python')
meta <- jsonlite::read_json(file.path(split_dir,'split.json'), simplifyVector=TRUE)
features <- meta$features
sites <- lapply(1:3,function(i) read.csv(file.path(split_dir,paste0('site',i,'.csv'))))
test <- read.csv(file.path(split_dir,'test.csv'))
common <- list(learning_rate=.05,batch_size=128L,local_epochs=epochs,
               optimizer='sgd',weight_decay=0,scheduler='none',hidden_layers=integer(0))
if (variant=='hazard') {
  model_params <- c(common,list(edges=c(0,7,14,21,30,45,60,90,120,180,270,365,540,730,1095,1460,1825)))
  contract <- 'pytorch_discrete_hazard'
} else {
  model_params <- c(common,list(horizon=1825,time_scale=365,distribution=variant,dispersion=1))
  contract <- 'pytorch_aft'
}
model <- do.call(ds.flower.model,c(list(name=contract),model_params))
sub <- dsFlowerClient:::.emit_submission(model)
cfg <- dsFlowerClient:::.neural_training_config(sub$params,sub$loss)
cfg[['model-spec-b64']] <- dsFlowerClient:::.spec_to_b64(sub$spec)
cfg[['loss-name']] <- sub$loss
cfg[['num-features']] <- length(features)
cfg[['num-classes']] <- 2L
cfg[['num-labels']] <- 2L
cfg[['num-server-rounds']] <- rounds
cfg[['batch-size']] <- 128L
cfg[['local-epochs']] <- epochs
dir.create(out,recursive=TRUE,showWarnings=FALSE)
config_file <- file.path(out,'config.json')
jsonlite::write_json(cfg,config_file,auto_unbox=TRUE,pretty=TRUE,digits=NA)
started <- Sys.time()
sha <- function(path) digest::digest(file=path,algo='sha256')
commit <- function(repo) trimws(system2('git',c('-C',shQuote(file.path(workspace,repo)),'rev-parse','HEAD'),stdout=TRUE))
build <- jsonlite::read_json(file.path(workspace,'runtime','build.json'),simplifyVector=FALSE)
runner_hash <- dsFlowerClient:::.compute_local_runner_hash()
stopifnot(identical(runner_hash,build$runner_sha256))
evidence <- list(schema_version=1L,record_type='cell',task='survival',status='running',
  started_utc=format(started,'%Y-%m-%dT%H:%M:%SZ',tz='UTC'),
  dataset=meta,variant=variant,contract=contract,epsilon=epsilon,delta=1e-5,clip=1,
  adjacency='replace_one',privacy_unit='patient',site_count=3L,
  package_commits=build$commits,installed_build=build,
  campaign_tools_commit=commit('dsFlowerClient'),
  package_versions=list(dsFlower=as.character(packageVersion('dsFlower')),dsFlowerClient=as.character(packageVersion('dsFlowerClient')),R=R.version.string,
    DSI=as.character(packageVersion('DSI')),DSLite=as.character(packageVersion('DSLite')),
    dsBase=as.character(packageVersion('dsBase')),resourcer=as.character(packageVersion('resourcer'))),
  runner_sha256=runner_hash,public_config=cfg,
  mechanism_provenance='Effective sigma/q/steps recomputed from public fixture subject census using audited unchanged runtime; independent PRV verification. No new private endpoint.',
  evaluation='channel B, public held-out subjects only',
  outcome_semantics=list(target_order=c('time','event'),event=1,censored=0,
    time_unit='days',baseline=meta$time_origin %||% 'synthetic baseline',
    administrative_censor='time>1825 => time1825,event0; event at1825 retained',
    invalid='time below1 or nonfinite/event not0or1/duplicate => valid0, never dropped',
    preprocessing='fixed public bounds scaled to[-1,1]; safe placeholders; no private fitted moments',
    interval_convention='event(left,right]; censor completed periods only'),
  score_conventions=list(time_ties='excluded',risk_ties=.5,hazard_risk='negative left-endpoint restricted mean',gap='central minus federated (historic sign reversed)'))
# No transient running record is put in the archived evidence directory.
result <- tryCatch({
  fed <- campaign_run_federated(sites,test,features,meta$feature_bounds,
     epsilon=epsilon,delta=1e-5,rounds=rounds,model_params=model_params,
     work_dir=file.path(out,'federation'),venv_root=file.path(workspace,'runtime','server'),
     contract=contract,target=c('time','event'),patient_column='subject_id',
     score_function=function(fit,test,features) {
       # Exercise package channel-B inference before independent scoring.
       risk <- ds.flower.predict(fit,test[,features,drop=FALSE],type='risk')
       stopifnot(length(risk)==nrow(test),all(is.finite(risk)))
       list(local_prediction_verified=TRUE)
     })
  script <- file.path(tools_dir,'survival','central_and_score.py')
  output <- file.path(out,'scores.json')
  processx::run(python,c(script,'--config',config_file,'--split',split_dir,
     '--out',output,'--epsilon',as.character(epsilon),
     '--federated-model',file.path(fed$artifact_dir,'model.pt')),
     error_on_status=TRUE,echo=TRUE,timeout=3600)
  scores <- jsonlite::read_json(output,simplifyVector=FALSE)
  evidence$status <- 'executed'
  evidence$results <- scores
  evidence$federation <- fed[setdiff(names(fed),c('metrics','artifact_dir'))]
  evidence$artifact_checksum <- fed$model_sha256
  evidence$campaign_tool_sha256 <- lapply(setNames(
    file.path(tools_dir,c('campaign_lib.R','survival/run_cell.R',
                         'survival/central_and_score.py','survival/metrics.py')),
    c('campaign_lib.R','run_cell.R','central_and_score.py','metrics.py')),sha)
  evidence$topology <- 'three isolated DSLite custodian workers and local Flower transport'
  evidence$cleanup_ok <- isTRUE(fed$cleanup_ok)
  evidence
},error=function(e) {
  evidence$status <- 'failed'
  # Public campaign errors only; never serialize private training fixtures.
  evidence$error <- conditionMessage(e)
  evidence$cleanup_ok <- !isTRUE(ds.flower.superlink.status()$running)
  evidence
})
result$finished_utc <- format(Sys.time(),'%Y-%m-%dT%H:%M:%SZ',tz='UTC')
result$elapsed_s <- as.numeric(difftime(Sys.time(),started,units='secs'))
jsonlite::write_json(result,file.path(out,'evidence.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA,null='null',na='null')
cat('SURVIVAL_CELL',result$status,variant,epsilon,'\n')
if (result$status!='executed') quit(status=1L)
