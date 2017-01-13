library(data.table)
library(mlr)
library(parallelMap)

# Read in the data, set names
ad_data <- fread("./data/data", header = FALSE,na.strings = "?")
meta_data <- fread("./data/column_names.txt", sep = ":", header = FALSE, blank.lines.skip = TRUE)

# Fixing column names
setnames(meta_data, c("col_names", "types"))
setnames(ad_data, make.names(meta_data[,col_names], unique = TRUE))

#finding missings
ad_data_drop_target <- copy(ad_data)
# Remove the target for imputation
ad_data_drop_target[,classes := NULL]
test_ad_data_index <- sample.int(nrow(ad_data_drop_target), size = 50)
train_ad_data_drop_target <- ad_data_drop_target[-test_ad_data_index,]
test_ad_data_drop_target <- ad_data_drop_target[test_ad_data_index,]

# Due to time constraints only the height task is used
# But a better model could be made by tuning a model for each task to impute
ad_missings_height <- makeRegrTask(id = "Find Missings Imputation for height",
                                   train_ad_data_drop_target[!is.na(height),], target = "height")

ad_missings_width <- makeRegrTask(id = "Find Missings Imputation for width",
                                  ad_data_drop_target[!is.na(width)], target = "width")

ad_missings_aratio <- makeRegrTask(id = "Find Missings Imputation for aratio",
                                   ad_data_drop_target[!is.na(aratio)], target = "aratio")

ad_missings_local <- makeClassifTask(id = "Find Missings Imputation for local",
                                     ad_data_drop_target[!is.na(local)], target = "local")

## Set up the tuning process
# Define
# 1. Create Learners
# 2. Define Parameter space
# 2. Define Tuning Control
# 3. Resampling Method
##

# Get list of available parameters for ctree
getLearnerParamSet("regr.ctree")


###############
# 1. Learner ##
###############
ctree_learner <- makeLearner("regr.ctree")


#######################
# 2. Parameter Space ##
#######################

ctree_parset <- makeParamSet(
  makeIntegerParam("minsplit", lower = 1, upper = 50),
  makeIntegerParam("minbucket", lower = 1, upper = 20),
  makeIntegerParam("mtry", lower = 0, upper = 10),
  makeIntegerParam("maxdepth", lower = 0, upper = 5)
)


#######################
# 3. Tuning Control  ##
#######################

ctrl <- makeTuneControlIrace(maxExperiments = 250)

#########################
# 4. Define Resampling ##
#   Use b632 sampling ##
#########################
bt_sample <- makeResampleDesc(method = "Bootstrap", iters = 5, predict = "both")

############################################
# Begin: Tuning Process                   ##
#  Excecuted in parallel over seven cores ##
############################################

library("parallelMap")
parallelStartSocket(7)
configureMlr(on.learner.error = "warn")
height_tune <- tuneParams(ctree_learner, ad_missings_height, bt_sample, ctree_parset, ctree_control, measures = rmse)
parallelStop()

# Train Final Model and check performance on holdout
height_ctree_final <- setHyperPars(ctree_learner, par.vals = height_tune$x)
height_train <- train(height_ctree_final, task = ad_missings_height)
height_predict <- predict(height_train, newdata = test_ad_data_drop_target)
performance(height_predict, rmse)

save(height_tune, file = "./models/ctree_height_impute_tune.RData")
save(height_train, file = "./models/height_tune.RData")