# Analysis of C5.0 model

library(mlr)
load("./models/c50_tune_mod.RData")

c50_partDepDat <- generateHyperParsEffectData(tune_mod_c50, partial.dep = TRUE)

head(c50_partDepDat$data)
# Because of some internal mlr problems we have to replace b632plus with test.mean
colnames(c50_partDepDat$data)[7] <- "kappa.test.mean"
c50_partDepDat$measures <- "kappa.test.mean"
# Have to go through and make all logical into integer 0 1 vars
for (i in 3:5)
  c50_partDepDat$data[,i] <- as.integer(c50_partDepDat$data[,i])
plotHyperParsEffect(c50_partDepDat, x="trials" , y = "kappa.test.mean", z = "minCases",
                    partial.dep.learn = "regr.earth", plot.type = "line")
