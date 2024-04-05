# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(haven)
library(caret)
library(parallel) # added to enable parrallelization
library(doParallel) # added to enable parrallelization 
library(tictoc) # added to track times

# Data Import and Cleaning
gss_import_tbl <- read_spss("../data/GSS2016.sav") %>% 
  # used read_spss instead of read_sav b/c of class debrief. both should work
  filter(!is.na(MOSTHRS)) %>%
  select(-HRS1, -HRS2)
  # I also split up gss_import_tbl and gss_tbl because of the class debrief
  # and to make things cleaner.
gss_tbl <- gss_import_tbl[, colSums(is.na(gss_import_tbl)) < 
                            .75 * nrow(gss_import_tbl)] %>% 
                            # pulls only variables w/ < 75% NAs
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
## I got rid of the loop I used previously and matched the class debrief
## to prevent long run times or potential glitches.
## Also removed the tuneGrid I'd previously used because it apparently 
## wasn't necessary.
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
  MOSTHRS ~ ., #using MOSTHRS instead of work_hours since I didn't rename it
  train_tbl,
  method="lm",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model1_time <-  toc() # 5.58 sec elapsed

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
model2_time <-  toc() # 10.17 sec elapsed

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
model3_time <-  toc() # 85.1 sec elapsed

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
model4_time <-  toc() # 244.69 sec elapsed

## PARALLELIZED
local_cluster <- makeCluster(detectCores() - 1) 
  ## to customize the number of cores used
registerDoParallel(local_cluster)
  ## signals to R that anytime something can be parrallelized, to do so.

tic()
set.seed(8712)
model1.par <- train( 
  MOSTHRS ~ ., #using MOSTHRS instead of work_hours since I didn't rename it
  train_tbl,
  method="lm",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = train_folds)
)
model1.par_time <-  toc() # 5.95 sec elapsed

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
model2.par_time <-  toc() # 2.2 sec elapsed

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
model3.par_time <-  toc() # 61.3 sec elapsed

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
model4.par_time <-  toc() # 135.64 sec elapsed

stopCluster(local_cluster)
registerDoSEQ()


# Publication
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

## I reverted to my previous version of this table before I put everything 
## into a loop.
table1_tbl <- tibble(
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

table2_tbl <- tibble(
  algo = c("OLS Regression", 
           "Elastic Net", 
           "Random Forest", 
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

# Questions
# 1. Which models benefited most from parallelization and why?
## Everything but OLS regression strongly benefited from parallelization, but 
## eXtreme Gradient Boosting benefited the most since it also had the longest
## origional run time, and parallelization nearly cut the processing time in 
## half.

# 2. How big was the difference between the fastest and slowest parallelized model? Why?
## First, I ran the code multiple times, and each time the processing time was 
## different and varied by several seconds. In the final version I just ran, the ## fastest parallelized model, Elastic Net, took under 3 seconds! Meanwhile, 
## eXtreme Gradient Boosting still took the longest, as expected becuase it is 
## the most complex. Additionally, OLS regression, actually took slightly longer ## under parallelization compared to the origional due to the unnecessary 
## overhead time added to run the code. 

# 3. If your supervisor asked you to pick a model for use in a production model, which would you recommend and why? Consider both Table 1 and Table 2 when providing an answer.
## I would recommend using random forest via parallelization as it resulted in 
## the R^2 value for cross validation was nearly as high as that for eXtreme 
## Gradient Boosting, and it had a more reasonable processing time than 
## xgbLinear. Additionally, random forest also had the highest holdout R^2 
## value.