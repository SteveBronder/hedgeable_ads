# For new data

# Read in the new data and set the names by the meta data
################# _NEW Data Goes Here_ #################
ad_data <- fread("./data/data", header = FALSE,na.strings = "?")
########################################################
meta_data <- fread("./data/column_names.txt", sep = ":", header = FALSE, blank.lines.skip = TRUE)

# Fixing column names
setnames(meta_data, c("col_names", "types"))
setnames(ad_data, make.names(meta_data[,col_names], unique = TRUE))

# Load the impute models
load("./models/impute_ad_list.RData")

# reimpute new data
ad_data_impute <- reimpute(ad_data, impute_ad_list$desc)

# Load Models

load("./models/hdrda_train_mod.RData")
load("./models/c50_train_mod.RData")
load("./models/stacked_hdrda_c50_train_mod.RData")

c50_predict <- predict(c50_train, newdata = ad_data)
hdrda_predict <- predict(c50_train, newdata = ad_data)
stack_predict <- predict(train_stack, newdata = test_ad_data_impute)

performance(c50_predict, kappa)
performance(hdrda_predict, kappa)
performance(stack_predict, kappa)


