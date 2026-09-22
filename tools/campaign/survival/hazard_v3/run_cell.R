#!/usr/bin/env Rscript
# Public survival experiment; privacy policy is set only inside custodian workers.
args <- commandArgs(TRUE)
if (length(args)!=6L) stop('Usage: run_cell.R workspace split epsilon out config phase')
workspace <- normalizePath(args[[1]])
split_dir <- normalizePath(args[[2]])
variant <- 'hazard'
epsilon <- as.numeric(args[[3]])
out <- normalizePath(args[[4]], mustWork = FALSE)
v3 <- jsonlite::read_json(args[[5]], simplifyVector=TRUE)
phase <- match.arg(args[[6]], c('development','confirmation','pilot'))
rounds <- as.integer(v3$rounds)
epochs <- as.integer(v3$local_epochs)
stopifnot(v3$protocol_version==3L, length(v3$edges)==v3$K+1L)
protocol_file <- 'hazard_v3/PROTOCOL.md'
.libPaths(c(file.path(workspace, 'runtime', 'rlib'), .libPaths()))
server_root <- normalizePath(file.path(workspace,'runtime','server'))
Sys.setenv(DSFLOWER_CLIENT_VENV_ROOT=file.path(workspace,'runtime'),
           DSFLOWER_VENV_ROOT=server_root,
           DSFLOWER_TEST_ALLOW_EPHEMERAL_SECRET='1',
           UV_CACHE_DIR=file.path(workspace,'runtime','uv-cache'),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
tools_dir <- file.path(workspace,'dsFlowerClient','tools','campaign')
source(file.path(tools_dir,'campaign_lib.R'))
# The existing campaign helper hardcodes FedAvg. Bind its public fit call in a
# private lexical environment; no package namespace or canonical helper changes.
fit_env <- new.env(parent=environment(campaign_run_federated))
fit_env$ds.flower.fit <- function(..., strategy) {
  dsFlowerClient::ds.flower.fit(..., strategy=v3$strategy,
    strategy_params=if (v3$strategy=='fedavgm') list(server_momentum=.9) else list())
}
environment(campaign_run_federated) <- fit_env
python <- file.path(workspace,'runtime','venv','bin','python')
meta <- jsonlite::read_json(file.path(split_dir,'split.json'), simplifyVector=TRUE)
sha <- function(path) digest::digest(file=path,algo='sha256')
stopifnot(identical(sha(file.path(split_dir,'train.csv')),meta$train_sha256),
          identical(sha(file.path(split_dir,'test.csv')),meta$test_sha256),
          identical(sha(file.path(tools_dir,'survival',protocol_file)),meta$protocol_sha256))
for (i in seq_len(nrow(meta$sites))) stopifnot(identical(
  sha(file.path(split_dir,paste0('site',i,'.csv'))),
  meta$sites$split_sha256[meta$sites$site==i]))
features <- meta$features
sites <- lapply(seq_len(nrow(meta$sites)),function(i) read.csv(file.path(split_dir,paste0('site',i,'.csv'))))
test <- read.csv(file.path(split_dir,'test.csv'))
common <- list(learning_rate=v3$learning_rate,batch_size=as.integer(v3$batch_size),
               local_epochs=epochs,optimizer=v3$optimizer,weight_decay=0,
               scheduler='none',hidden_layers=integer(0))
model_params <- c(common,list(edges=v3$edges))
contract <- 'pytorch_discrete_hazard'
model <- do.call(ds.flower.model,c(list(name=contract),model_params))
sub <- dsFlowerClient:::.emit_submission(model)
cfg <- dsFlowerClient:::.neural_training_config(sub$params,sub$loss)
cfg[['model-spec-b64']] <- dsFlowerClient:::.spec_to_b64(sub$spec)
cfg[['loss-name']] <- sub$loss
cfg[['num-features']] <- length(features)
cfg[['num-classes']] <- 2L
cfg[['num-labels']] <- 2L
cfg[['num-server-rounds']] <- rounds
cfg[['batch-size']] <- common$batch_size
cfg[['local-epochs']] <- epochs
cfg[['strategy']] <- v3$strategy
if (v3$strategy=='fedavgm') cfg[['strategy-server-momentum']] <- .9
dir.create(out,recursive=TRUE,showWarnings=FALSE)
config_file <- file.path(out,'config.json')
jsonlite::write_json(cfg,config_file,auto_unbox=TRUE,pretty=TRUE,digits=NA)
started <- Sys.time()
commit <- function(repo) trimws(system2('git',c('-C',shQuote(file.path(workspace,repo)),'rev-parse','HEAD'),stdout=TRUE))
build <- jsonlite::read_json(file.path(workspace,'runtime','build.json'),simplifyVector=FALSE)
runner_hash <- dsFlowerClient:::.compute_local_runner_hash()
stopifnot(identical(runner_hash,build$runner_sha256))
evidence <- list(schema_version=1L,record_type='cell',task='survival',status='running',
  started_utc=format(started,'%Y-%m-%dT%H:%M:%SZ',tz='UTC'),
  dataset=meta,variant=variant,contract=contract,epsilon=epsilon,delta=1e-5,clip=1,
  adjacency='replace_one',privacy_unit='patient',site_count=length(sites),
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
evidence$protocol_version <- 3L
evidence$hazard_v3_config <- v3
evidence$hazard_v3_phase <- phase
evidence$evaluation <- meta$evaluation_role
evidence$pod_id <- 'x0w6ewmpinpsuk'
# No transient running record is put in the archived evidence directory.
result <- tryCatch({
  fed <- campaign_run_federated(sites,test,features,meta$feature_bounds,
     epsilon=epsilon,delta=1e-5,rounds=rounds,model_params=model_params,
     work_dir=file.path(out,'federation'),venv_root=server_root,
     contract=contract,target=c('time','event'),patient_column='subject_id',
     score_function=function(fit,test,features) {
       # Exercise package channel-B inference before independent scoring.
       risk <- ds.flower.predict(fit,test[,features,drop=FALSE],type='risk')
       stopifnot(length(risk)==nrow(test),all(is.finite(risk)))
       list(local_prediction_verified=TRUE)
     })
  script <- file.path(tools_dir,'survival','hazard_v3','central_and_score.py')
  output <- file.path(out,'scores.json')
  processx::run(python,c(script,'--config',config_file,'--split',split_dir,
     '--out',output,'--epsilon',as.character(epsilon),
     '--federated-model',file.path(fed$artifact_dir,'model.pt'),
     if (phase=='development') '--development' else character()),
     error_on_status=TRUE,echo=TRUE,timeout=3600)
  scores <- jsonlite::read_json(output,simplifyVector=FALSE)
  evidence$status <- 'executed'
  evidence$results <- scores
  evidence$federation <- fed[setdiff(names(fed),c('metrics','artifact_dir'))]
  evidence$artifact_checksum <- fed$model_sha256
  evidence$campaign_tool_sha256 <- lapply(setNames(
    file.path(tools_dir,c('campaign_lib.R','survival/hazard_v3/run_cell.R',
                         'survival/hazard_v3/central_and_score.py','survival/metrics.py')),
    c('campaign_lib.R','run_cell.R','central_and_score.py','metrics.py')),sha)
  evidence$topology <- paste(length(sites),'isolated DSLite custodian workers and local Flower transport')
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
