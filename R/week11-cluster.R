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

# Visualization
gss_tbl %>%
  ggplot(aes(x=MOSTHRS)) +
  geom_histogram(fill = "lightblue", color = "darkblue") +
  labs(title = "Distribution of Work Hours",
       x = "Work Hours",
       y = "Frequency") +
  theme_bw()

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
model4 <- train(
  MOSTHRS ~ .,
  train_tbl,
  method="xgbLinear",
  na.action = na.pass,
  tuneLength = 1,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model4_time <-  toc() 

## PARALLELIZED
local_cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(local_cluster)

tic()
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
model4.par <- train(
  MOSTHRS ~ .,
  train_tbl,
  method="xgbLinear",
  na.action = na.pass,
  tuneLength = 1,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model4.par_time <-  toc() 

stopCluster(local_cluster)
registerDoSEQ()

## Show details from each of the models
model1
cv_model1 <- model1$results$Rsquared
holdout_model1 <- cor(predict(model1, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS)^2

model2
cv_model2 <- max(model2$results$Rsquared)
holdout_model2 <- cor(predict(model2, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS)^2

model3
cv_model3 <- max(model3$results$Rsquared)
holdout_model3 <- cor(predict(model3, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS)^2

model4
cv_model4 <- max(model4$results$Rsquared)
holdout_model4 <- cor(predict(model4, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS)^2

summary(resamples(list(model1, model2, model3, model4)), metric="Rsquared")
dotplot(resamples(list(model1, model2, model3, model4)), metric="Rsquared")

# Publication
table1_tbl <- tibble(
  algo = c("OLS regression", 
           "elastic net", 
           "random forest", 
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

table2_tbl <- tibble(
  algo = c("OLS regression", 
           "elastic net", 
           "random forest", 
           "eXtreme Gradient Boosting"),
  original = (c(as.numeric(abs(model1_time$tic-model1_time$toc)), 
                as.numeric(abs(model2_time$tic-model2_time$toc)), 
                as.numeric(abs(model3_time$tic-model3_time$toc)), 
                as.numeric(abs(model4_time$tic-model4_time$toc)))),
  parallelized = c(as.numeric(abs(model1.par_time$tic-model1.par_time$toc)), 
                   as.numeric(abs(model2.par_time$tic-model2.par_time$toc)), 
                   as.numeric(abs(model3.par_time$tic-model3.par_time$toc)), 
                  as.numeric(abs(model4.par_time$tic-model4.par_time$toc)))
  )