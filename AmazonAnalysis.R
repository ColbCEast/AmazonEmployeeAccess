## Data Wrangling
library(tidymodels)
library(ggmosaic)
library(embed)
library(vroom)

train_data <- vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(ACTION = as.factor(ACTION))

the_recipe <- recipe(ACTION~., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # change all predictors into factors
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

prep(the_recipe)
bake(prep, new_data = train_data)


## LOGISTIC REGRESSION
logistic_mod <- logistic_reg() %>%
  set_engine("glm")

logistic_wf <- workflow() %>%
  add_recipe(the_recipe) %>%
  add_model(logistic_mod) %>%
  fit(data = train_data)

logistic_preds <- predict(logistic_wf,
                          new_data = test_data,
                          type = "prob") %>%
  mutate(Action=ifelse(.pred_1>.85, 1, 0)) %>%
  bind_cols(., test_data) %>%
  select(id, Action) %>%
  rename(Id = id)

vroom_write(logistic_preds, "LogisticPreds.csv", delim = ",")


## PENALIZED LOGISTIC REGRESSION
library(glmnet)
train_data <- vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(ACTION = as.factor(ACTION))

the_recipe <- recipe(ACTION~., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # change all predictors into factors
  step_other(all_nominal_predictors(), threshold = 0.001) %>% # puts all predictors with less than 1% of the total data into an "other" category
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) # target encoding

my_mod <- logistic_reg(mixture = tune(),
                       penalty = tune()) %>%
  set_engine("glmnet")

pen_logistic_wf <- workflow() %>%
  add_recipe(the_recipe) %>%
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

vroom_write(pen_logistic_preds, "PenLogisticPreds.csv", delim = ",")
best_tune
