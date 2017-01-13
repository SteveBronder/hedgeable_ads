# Script to make final models

library(data.table)
library(mlr)
library(parallelMap)

# Read in the data, set names
ad_data <- fread("./data/data", header = FALSE,na.strings = "?")
meta_data <- fread("./data/column_names.txt", sep = ":", header = FALSE, blank.lines.skip = TRUE)

# Fixing column names
setnames(meta_data, c("col_names", "types"))
setnames(ad_data, make.names(meta_data[,col_names], unique = TRUE))

# Create mlr classification task
ad_task <- makeClassifTask(id = "Hedgeable Ad Identification",
                           data = ad_data,
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

save(impute_ad_list, file = "./models/impute_ad_list.RData")

# Load the final model with best tuning parameters
load("./models/hdrda_tune_mod.RData")
load("./models/c50_tunemod.RData")
load("./models/stack_learner_tune_mod.RData")

# Final Models over all Data
c50_final <- setHyperPars(c50_learner, par.vals = tune_mod_c50$x)
c50_train <- train(c50_final, impute_ad_task)

hdrda_final <- setHyperPars(hdrda_learner, par.vals = tune_mod_hdrda$x)
hdrda_train <- train(hdrda_final, impute_ad_task)



parallelStartSocket(8)
configureMlr(on.learner.error = "warn")
train_stack <- train(stack_learner, impute_ad_task)
parallelStop()

save(c50_train, file = "./models/c50_final_mod.RData")
save(hdrda_train, file = "./models/hdrda_final_mod.RData")
save(train_stack, file = "./models/stack_final_mod.RData")
