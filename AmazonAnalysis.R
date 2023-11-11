## Data Wrangling
library(tidymodels)
library(embed)
library(vroom)
library(themis)

train_data <- vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(ACTION = as.factor(ACTION))

#the_recipe <- recipe(ACTION~., data = train_data) %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% # change all predictors into factors
  #step_other(all_nominal_predictors(), threshold = 0.01) %>%
  #step_dummy(all_nominal_predictors())

#prep(the_recipe)
#bake(prep(the_recipe), new_data = train_data)

balance_recipe <- recipe(ACTION~., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_downsample(under_ratio = 1)

## LOGISTIC REGRESSION
logistic_mod <- logistic_reg() %>%
  set_engine("glm")

logistic_wf <- workflow() %>%
  add_recipe(balance_recipe) %>%
  add_model(logistic_mod) %>%
  fit(data = train_data)

logistic_preds <- predict(logistic_wf,
                          new_data = test_data,
                          type = "prob") %>%
  mutate(Action=ifelse(.pred_1>.85, 1, 0)) %>%
  bind_cols(., test_data) %>%
  select(id, Action) %>%
  rename(Id = id)

vroom_write(logistic_preds, file = "BalancedLogisticPreds.csv", delim = ",")


## PENALIZED LOGISTIC REGRESSION
library(glmnet)
#train_data <- vroom("train.csv")

#test_data <- vroom("test.csv")

#train_data <- train_data %>%
  #mutate(ACTION = as.factor(ACTION))

#the_recipe <- recipe(ACTION~., data = train_data) %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% # change all predictors into factors
  #step_other(all_nominal_predictors(), threshold = 0.001) %>% # puts all predictors with less than 1% of the total data into an "other" category
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) # target encoding

my_mod <- logistic_reg(mixture = tune(),
                       penalty = tune()) %>%
  set_engine("glmnet")

pen_logistic_wf <- workflow() %>%
  add_recipe(balance_recipe) %>%
  add_model(my_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(train_data, v = 10, repeats = 1)

cv_results <- pen_logistic_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

best_tune <- cv_results %>%
  select_best("roc_auc")

final_wf <- pen_logistic_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train_data)

pen_logistic_preds <- predict(final_wf,
                          new_data = test_data,
                          type = "prob") %>%
  mutate(Action=ifelse(.pred_1>.85, 1, 0)) %>%
  bind_cols(., test_data) %>%
  select(id, Action) %>%
  rename(Id = id)

vroom_write(pen_logistic_preds, file = "BalancedPenalizedPreds.csv", delim = ",")


## RANDOM FORESTS
library(ranger)
#library(doParallel)

# Set up parallel computing
#cl <- makePSOCKcluster(12)
#registerDoParallel(cl)

forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(balance_recipe) %>%
  add_model(forest_mod)

forest_tuning_grid <- grid_regular(mtry(range = c(1,10)),
                                   min_n(),
                                   levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

CV_results <- forest_wf %>%
  tune_grid(resamples = folds,
            grid = forest_tuning_grid,
            metrics = metric_set(roc_auc))

best_tune_forest <- CV_results %>%
  select_best("roc_auc")

final_forest_wf <- forest_wf %>%
  finalize_workflow(best_tune_forest) %>%
  fit(data = train_data)

forest_preds <- predict(final_forest_wf,
                              new_data = test_data,
                              type = "prob") %>%
  mutate(Action=ifelse(.pred_1>.95, 1, 0)) %>%
  bind_cols(., test_data) %>%
  select(id, Action) %>%
  rename(Id = id)

vroom_write(forest_preds, file = "BalancedForestPreds.csv", delim = ",")

#stopCluster(cl)


# Naive Bayes Model
library(naivebayes)
#library(doParallel)
library(discrim)

#cl <- makePSOCKcluster(12)
#registerDoParallel(cl)

#train_data <- vroom("train.csv")

#test_data <- vroom("test.csv")

#train_data <- train_data %>%
  #mutate(ACTION = as.factor(ACTION))

nb_model <- naive_Bayes(Laplace = tune(),
                        smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

#the_recipe <- recipe(ACTION~., data = train_data) %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  #step_other(all_nominal_predictors(), threshold = 0.001) %>%
  #step_dummy(all_nominal_predictors()) %>%
  #step_normalize(all_predictors()) %>%
  #step_pca(all_predictors(), threshold = 0.85)

nb_wf <- workflow() %>%
  add_recipe(balance_recipe) %>%
  add_model(nb_model)

nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = nb_tuning_grid,
            metrics = metric_set(roc_auc))

best_tune_nb <- CV_results %>%
  select_best("roc_auc")

final_nb_wf <- nb_wf %>%
  finalize_workflow(best_tune_nb) %>%
  fit(data = train_data)

nb_preds <- predict(final_nb_wf,
                        new_data = test_data,
                        type = "prob") %>%
  mutate(Action=ifelse(.pred_1>.9, 1, 0)) %>%
  bind_cols(., test_data) %>%
  select(id, Action) %>%
  rename(Id = id)

vroom_write(nb_preds, file = "BalancedNBPreds.csv", delim = ",")

#stopCluster(cl)

# K-Nearest Neighbors Model
library(kknn)
#cl <- makePSOCKcluster(12)
#registerDoParallel(cl)

#train_data <- vroom("train.csv")

#test_data <- vroom("test.csv")

#train_data <- train_data %>%
  #mutate(ACTION = as.factor(ACTION))

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(balance_recipe) %>%
  add_model(knn_model)

tuning_grid_knn <- grid_regular(neighbors(),
                                levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_knn,
            metrics = metric_set(roc_auc))

best_tune_knn <- CV_results %>%
  select_best("roc_auc")

final_knn_wf <- knn_wf %>%
  finalize_workflow(best_tune_knn) %>%
  fit(data = train_data)

knn_preds <- predict(final_knn_wf,
                        new_data = test_data,
                        type = "prob") %>%
  mutate(Action=ifelse(.pred_1>.95, 1, 0)) %>%
  bind_cols(., test_data) %>%
  select(id, Action) %>%
  rename(Id = id)

vroom_write(knn_preds, file = "BalancedKNNPreds.csv", delim = ",")

#stopCluster(cl)


# Support Vector Machine Model
library(kernlab)
#cl <- makePSOCKcluster(12)
#registerDoParallel(cl)

#train_data <- vroom("train.csv")

#test_data <- vroom("test.csv")

#train_data <- train_data %>%
  mutate(ACTION = as.factor(ACTION))

#the_recipe <- recipe(ACTION~., data = train_data) %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% # change all predictors into factors
  #step_other(all_nominal_predictors(), threshold = 0.001) %>% # puts all predictors with less than 1% of the total data into an "other" category
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

svmPoly <- svm_poly(degree = 2,
                    cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_model(svmPoly) %>%
  add_recipe(balance_recipe)

tuning_grid_svm <- grid_regular(cost(),
                                levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

CV_results <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_svm,
            metrics = metric_set(roc_auc))

best_tune_svm <- CV_results %>%
  select_best("roc_auc")

final_svm_wf <- svm_wf %>%
  finalize_workflow(best_tune_sv) %>%
  fit(data = train_data)

svm_preds <- predict(final_svm_wf,
                     new_data = test_data,
                     type = "prob") %>%
  mutate(Action=.pred_1) %>%
  bind_cols(., test_data) %>%
  select(id, Action) %>%
  rename(Id = id)

vroom_write(svm_preds, file = "BalancedSVMPreds.csv", delim = ",")

#stopCluster(cl)


## FINAL MODEL
library(ranger)
library(tidymodels)
library(embed)
library(vroom)
library(themis)

train_data <- vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(ACTION = as.factor(ACTION))

final_recipe <- recipe(ACTION~., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.85) %>%
  step_smote(all_outcomes(), neighbors = 5)

forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(final_recipe) %>%
  add_model(forest_mod)

forest_tuning_grid <- grid_regular(mtry(range = c(1,10)),
                                   min_n(),
                                   levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

CV_results <- forest_wf %>%
  tune_grid(resamples = folds,
            grid = forest_tuning_grid,
            metrics = metric_set(roc_auc))

best_tune_forest <- CV_results %>%
  select_best("roc_auc")

final_forest_wf <- forest_wf %>%
  finalize_workflow(best_tune_forest) %>%
  fit(data = train_data)

forest_preds <- predict(final_forest_wf,
                        new_data = test_data,
                        type = "prob") %>%
  mutate(Action=.pred_1) %>%
  bind_cols(., test_data) %>%
  select(id, Action) %>%
  rename(Id = id)

vroom_write(forest_preds, file = "FinalForestPreds.csv", delim = ",")
