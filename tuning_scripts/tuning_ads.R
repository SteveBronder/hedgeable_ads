library(data.table)
library(mlr)
library(parallelMap)

# Read in the data, set names
ad_data <- fread("./data/data", header = FALSE,na.strings = "?")
meta_data <- fread("./data/column_names.txt", sep = ":", header = FALSE, blank.lines.skip = TRUE)

# Fixing column names
setnames(meta_data, c("col_names", "types"))
setnames(ad_data, make.names(meta_data[,col_names], unique = TRUE))

# If you want tune non-impute learner, uncomment this
#ad_data <- ad_data[complete.cases(ad_data),]
# Make a holdout set
test_ad_data_index <- sample.int(nrow(ad_data), size = 100)
train_ad_data <- ad_data[-test_ad_data_index,]
test_ad_data <- ad_data[test_ad_data_index,]

# Create mlr classification task
ad_task <- makeClassifTask(id = "Hedgeable Ad Identification",
                           data = train_ad_data,
                           target = "classes",
                           positive = "ad.")

# Do imputation for the task
load("./models/height_tune.RData")
impute_ad_list <- impute(ad_task,
                         cols = list(height = imputeLearner(setHyperPars(makeLearner("regr.ctree"),
                                                                         par.vals = height_tune$x)),
                                     width  = imputeLearner(setHyperPars(makeLearner("regr.ctree"),
                                                                         par.vals = height_tune$x)),
                                     aratio = imputeLearner(setHyperPars(makeLearner("regr.ctree"),
                                                                         par.vals = height_tune$x)),
                                     local  = imputeHist()))

impute_ad_task <- impute_ad_list$task

# Impute on holdout set
test_ad_data_impute <- reimpute(test_ad_data, impute_ad_list$desc)

# Save impute object
save(impute_ad_list, file = "./models/impute_ad_list.RData")

# Notice that there are over 6x as many nonads as ads
# Going to do smote, to synthetically oversample the ads
#ad_task_smote <- smote(impute_ad_task, rate = 3, nn = 10,standardize = TRUE, alt.logic = TRUE)

## Set up the tuning process
# Define
# 1. Create Learners
# 2. Define Parameter space
# 2. Define Tuning Control
# 3. Resampling Method
##

# Get list of available parameters for hdrda and C50
# Package Link: https://cran.r-project.org/web/packages/sparsediscrim/
getLearnerParamSet("classif.hdrda")
getLearnerParamSet("classif.C50")


###############
# 1. Learner ##
###############
# NOTE: hdrda does not accept NAs                              
hdrda_learner <- makeLearner("classif.hdrda")

c50_learner <- makeLearner("classif.C50")

#######################
# 2. Parameter Space ##
#######################

hdrda_parset <- makeParamSet(
  makeNumericParam("lambda", lower = .0000001, upper = .999999),
  makeNumericParam("gamma", lower = -15, upper = 15, trafo = function(x) 2^x)
)

c50_parset <- makeParamSet(
  makeIntegerParam("trials", lower = 1, upper = 25),
  makeIntegerParam("minCases", lower = 1, upper = 25),
  makeLogicalParam("rules"),
  makeLogicalParam("winnow"),
  makeLogicalParam("noGlobalPruning"),
  makeNumericParam("CF", lower = .15, upper = .45)
)

#######################
# 3. Tuning Control  ##
#######################
ctrl <- makeTuneControlIrace(maxExperiments = 250)

#########################
# 4. Define Resampling ##
#   Use b632+ sampling ##
#########################
# Link: http://stats.stackexchange.com/questions/96739/what-is-the-632-rule-in-bootstrapping
bt_sample <- makeResampleDesc(method = "Bootstrap", iters = 10, predict = "both")


############################################
# Begin: Tuning Process                   ##
#  Excecuted in parallel over eight cores ##
############################################

# Tune C50 model
parallelStartSocket(8)
configureMlr(on.learner.error = "warn")
tune_mod_c50 <- tuneParams(learner = c50_learner, task = impute_ad_task,
                       measures = setAggregation(kappa, b632plus), resampling = bt_sample,
                       par.set = c50_parset, control = ctrl )
parallelStop()

# Train Final Model and check performance on holdout
c50_final <- setHyperPars(c50_learner, par.vals = tune_mod_c50$x)
c50_train <- train(c50_final, impute_ad_task)
c50_pred  <- predict(c50_train, newdata = test_ad_data_impute)
performance(c50_pred, kappa)

# Tune hdrda Model
# FIXME: b632plus caused memory errors?
parallelStartSocket(7)
configureMlr(on.learner.error = "warn")
tune_mod_hdrda <- tuneParams(learner = hdrda_learner, task = impute_ad_task,
                       measures = setAggregation(kappa, b632), resampling = bt_sample,
                       par.set = hdrda_parset, control = ctrl )
parallelStop()

# Train Final Model and check performance on holdout
hdrda_final <- setHyperPars(hdrda_learner, par.vals = tune_mod_hdrda$x)
hdrda_train <- train(hdrda_final, impute_ad_task)
hdrda_pred <- predict(hdrda_train, newdata = test_ad_data_impute)
performance(hdrda_pred, kappa)

## Save the tuned models
save(hdrda_train, file = "./models/hdrda_train_mod.RData")
save(tune_mod_hdrda, file = "./models/hdrda_tune_mod.RData")
save(c50_train, file = "./models/c50_train_mod.RData")
save(tune_mod_c50, file = "./models/c50_tune_mod.RData")

test_stackLearn <- makeLearners(c("classif.hdrda", "classif.C50"))
test_stackLearn[[1]] <- setHyperPars(test_stackLearn[[1]], par.vals = tune_mod_hdrda$x)
test_stackLearn$classif.hdrda$predict.type = "prob"
test_stackLearn[[2]] <- setHyperPars(test_stackLearn[[2]], par.vals = tune_mod_c50$x)
test_stackLearn$classif.C50$predict.type = "prob"
test_stack <- makeStackedLearner(test_stackLearn, method = "hill.climb", predict.type = "prob")

train_stack <- train(test_stack, impute_ad_task)
