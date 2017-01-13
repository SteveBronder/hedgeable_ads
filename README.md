# hedgeable_ads
Contains Hedgeable Machine Learning Assignement For Prediction

From the Instructions:

***

## Problem
Using a dataset of past advertisements on the Internet, can we accurately predict what image will be an advertisement based on attributes of that image?

## Project
The features encode the geometry of the image (if available) as well as phrases occurring in the URL, 
the image's URL and alt text, the anchor text, and words occurring near the anchor text.

Number of Instances: 3,279 (2,821 non ads, 458 ads)

Number of Attributes: 1,558 (3 continous; others binary)

28% of instances are missing some of the continuous attributes.

Missing values should be interpreted as "unknown"

Class Distribution- number of instances per class: 2,821 non ads, 458 ads.

The task is to predict whether an image is an advertisement ("ad") or not ("non ad").

## Deliverables
Please send us the following:

1. Code, and associated files, used for the project. You can send us a zipfile, or upload the project to a public github repo.
2. The algorithm you developed to make your predictions
3. How we can run the algorithm on a test data set
4. The process you used to analyze the data and came to your conclusions

***

## Folder Description

The files and folders in this repository contain the code and associated files to predict whether an image is an ad based on its underlying properties. In each folder are:

1. data
 - The data and meta data for advertisements
2. tuning_scripts
 - The code to tune the data
   - tuning_missing: contains the code to build the model which imputes missing data. Because of time constraints a model was built to impute height, and because width and aratio are most likely strongly correlated with height the tuned parameters from the ctree model used to predict height were also used to build the width and aratio model. the local variable was a binary variable and, due to time constraints, a histogram imputation strategy is used to calculate random values to impute into local.
   - tuning_ads: contains the code to impute and tune the C5.0 and hdrda models
   - tuning_c50: contains code to tune a C5.0 model that also tunes the threshold for the hard prediction of ad vs. nonad. Due to time constraints this was not run, but is given to show what how a better model could be built.
2. models
 - The final models created from the tuning and training routine as well as the tuning objects and imputation objects.
   - **_train_mod: the final tuned model for the given task
   - **_tune_mod: the tuning object returned from mlr. This is mostly used to analyze how the model reacted to new hyperparameters. height_tune is used to build the imputation models in tuning_ads and tuning_c50
3. img
 - Contains pictures that analyze the respective tuning object hyperparameters
4. analysis
 - Code used to generate analysis
5. predict
 - Contains script that can be used to predict with new data for the C5.0 model
 
## Requirements

Code is written in `R` and requires the packages `mlr`, `parallelMap`, and `data.table`. The models used require the packages `C50`, `hdrda`, `party` and `randomForest`. Tuning is performed in parallel and so the number of cores in `parallelStartSocket()` should be changed for your particular computer.

## Data

This section uses code from `tuning_ads` up until line 37. The provided data is wrapped into a `ClassifTask` using the package `mlr`. The classification task is the building block for the rest of the functions in `mlr` and contains meta-information for the data.

```r
ad_task

# Supervised task: Hedgeable Ad Identification
# Type: classif
# Target: classes
# Observations: 3179
# Features:
# numerics  factors  ordered 
#     1558        0        0 
# Missings: TRUE
# Has weights: FALSE
# Has blocking: FALSE
# Classes: 2
#    ad. nonad. 
#    442   2737 
# Positive class: ad.
```

With some variables having NA's there are three possible routes

1. Remove all NA observations and only use the observations with complete cases
2. Use a model that accounts for NA's in the training data inherently
3. Use an imputation method on the observations with NAs, then use the imputed data to train the final model.

The first route has the draw back that our model will be unable to make predictions for data which has any NAs. (2) will work, but will limit the types of models we are allowed to employ. This analysis chooses (3) and builds a `ctree` model to predict the continuous variables while generating random values with probabilities calculated from the variables histogram for binary variables.

An imputation method is used on the data for the variables `height`, `width`, `aratio`, and `local` which creates a new task labeled `impute_ad_task`. A Conditional Inference Tree model is used to develop imputations for the continuous variables with missing data as it is a simple model that is known to produce quick and reasonable results. More information on the inner workings of `ctree` can be found in the details of the help file [`?party::ctree()`] Model development for the imputation technique can be viewed in the file marked `tuning_missing` in the folder `tuning_scripts`.


The `mlr` package's `impute()` function has a reproducible benefit, in that imputation can be easily performed on testing data by simply calling `reimpute()` [line 40]. For our purposes, 100 observations were randomly chosen to be the final holdout data that our models will be assessed on at the end of the tuning procedure.

```r
impute_ad_task

# Supervised task: Hedgeable Ad Identification
# Type: classif
# Target: classes
# Observations: 3179
# Features:
# numerics  factors  ordered 
#     1558        0        0 
# Missings: FALSE
# Has weights: FALSE
# Has blocking: FALSE
# Classes: 2
#    ad. nonad. 
#    442   2737 
# Positive class: ad.
```


## Models

Two models were tuned over the imputed data.

1. C5.0 ([wiki](https://en.wikipedia.org/wiki/C4.5_algorithm))
- Desc: 

2. hdrda ([paper](https://arxiv.org/pdf/1602.01182v1.pdf))

