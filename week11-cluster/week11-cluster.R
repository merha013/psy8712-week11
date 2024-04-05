# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
library(parallel) 
library(doParallel)
library(tictoc)

# Data Import and Cleaning
gss_import_tbl <- read_spss("../data/GSS2016.sav") %>% 
  filter(!is.na(MOSTHRS)) %>%
  select(-HRS1, -HRS2)
gss_tbl <- gss_import_tbl[, colSums(is.na(gss_import_tbl)) < 
                            .75 * nrow(gss_import_tbl)] %>% 
  mutate_all(as.numeric)

# Analysis
holdout_indices <- createDataPartition(
  gss_tbl$MOSTHRS,
  p = .25,
  list = T)$Resample1
test_tbl <- gss_tbl[holdout_indices,]
train_tbl <- gss_tbl[-holdout_indices,]
train_folds <- createFolds(train_tbl$MOSTHRS)

## ORIGIONAL (non-parallelized version of code)
tic()
set.seed(8712)
model1 <- train( 
  MOSTHRS ~ .,
  train_tbl,
  method="lm",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model1_time <-  toc()

tic()
set.seed(8712)
model2 <- train(
  MOSTHRS ~ .,
  train_tbl,
  method="glmnet",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model2_time <-  toc()

tic()
set.seed(8712)
model3 <- train(
  MOSTHRS ~ .,
  train_tbl,
  method="ranger",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model3_time <-  toc()

tic()
set.seed(8712)
model4 <- train(
  MOSTHRS ~ .,
  train_tbl,
  method="xgbLinear",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model4_time <-  toc()

## PARALLELIZED
local_cluster <- makeCluster(127) # amdsmall cores per node (128) - 1 
registerDoParallel(local_cluster)

tic()
set.seed(8712)
model1.par <- train( 
  MOSTHRS ~ .,
  train_tbl,
  method="lm",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model1.par_time <-  toc()

tic()
set.seed(8712)
model2.par <- train(
  MOSTHRS ~ .,
  train_tbl,
  method="glmnet",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model2.par_time <-  toc()

tic()
set.seed(8712)
model3.par <- train(
  MOSTHRS ~ .,
  train_tbl,
  method="ranger",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model3.par_time <-  toc()

tic()
set.seed(8712)
model4.par <- train(
  MOSTHRS ~ .,
  train_tbl,
  method="xgbLinear",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model4.par_time <-  toc()

stopCluster(local_cluster)
registerDoSEQ()


# Publication
cv_model1 <- model1$results$Rsquared
holdout_model1 <- cor(predict(model1, test_tbl, na.action = na.pass),
                      test_tbl$MOSTHRS)^2

cv_model2 <- max(model2$results$Rsquared)
holdout_model2 <- cor(predict(model2, test_tbl, na.action = na.pass),
                      test_tbl$MOSTHRS)^2

cv_model3 <- max(model3$results$Rsquared)
holdout_model3 <- cor(predict(model3, test_tbl, na.action = na.pass),
                      test_tbl$MOSTHRS)^2

cv_model4 <- max(model4$results$Rsquared)
holdout_model4 <- cor(predict(model4, test_tbl, na.action = na.pass),
                      test_tbl$MOSTHRS)^2

summary(resamples(list(model1, model2, model3, model4)), metric="Rsquared")
dotplot(resamples(list(model1, model2, model3, model4)), metric="Rsquared")

Table_3 <- tibble(
  algo = c("OLS Regression", 
           "Elastic Net", 
           "Random Forest", 
           "eXtreme Gradient Boosting"),
  cv_rsq = c(
    str_remove(formatC(cv_model1, format = 'f', digits = 2), "^0"),
    str_remove(formatC(cv_model2, format = 'f', digits = 2), "^0"),
    str_remove(formatC(cv_model3, format = 'f', digits = 2), "^0"),
    str_remove(formatC(cv_model4, format = 'f', digits = 2), "^0")),
  ho_rsq = c( # round all values to 2 decimal places & with no leading zero
    str_remove(formatC(holdout_model1, format = 'f', digits = 2), "^0"),
    str_remove(formatC(holdout_model2, format = 'f', digits = 2), "^0"),
    str_remove(formatC(holdout_model3, format = 'f', digits = 2), "^0"),
    str_remove(formatC(holdout_model4, format = 'f', digits = 2), "^0"))
)

Table_4 <- tibble(
  algo = c("OLS Regression", 
           "Elastic Net", 
           "Random Forest", 
           "eXtreme Gradient Boosting"),
  supercomputer = (c(as.numeric(abs(model1_time$tic-model1_time$toc)), 
                as.numeric(abs(model2_time$tic-model2_time$toc)), 
                as.numeric(abs(model3_time$tic-model3_time$toc)), 
                as.numeric(abs(model4_time$tic-model4_time$toc)))),
  supercomputer_127 = c(as.numeric(abs(model1.par_time$tic-model1.par_time$toc)), 
                   as.numeric(abs(model2.par_time$tic-model2.par_time$toc)), 
                   as.numeric(abs(model3.par_time$tic-model3.par_time$toc)), 
                   as.numeric(abs(model4.par_time$tic-model4.par_time$toc)))
)

# Save Files
write_csv(Table_3, "table3.csv")
table4.csv(Table_4, "table4.csv")

# Questions
# 1. Which models benefited most from moving to the supercomputer and why?
## 

# 2. What is the relationship between time and the number of cores used?
## 

# 3. If your supervisor asked you to pick a model for use in a production model, # would you recommend using the supercomputer and why? Consider all four tables 
# when providing an answer.
## 
