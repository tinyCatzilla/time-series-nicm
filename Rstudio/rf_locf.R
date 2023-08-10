library(tidyverse)
library(randomForestSRC)
library(data.table)
library(caret) 
library(Metrics)
library(pROC)

options(scipen = 25, digits = 4)

# Load data
labels <- fread("nicm/labels_processed.csv")
dx <- fread("nicm/dx_processed.csv")
test <- fread("nicm/data_test.csv")

# Merge data based on patient id
merged_data <- labels %>%
  inner_join(dx, by = "pid")

# Merge test data with labels
test <- labels %>%
  inner_join(test, by = "pid")

# Preprocess data
merged_data <- merged_data %>%
  arrange(pid, desc(dtime)) %>%
  group_by(pid) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(across(C61:Z98.890, ~replace_na(as.numeric(.) * 1L, 0)))

# Apply same preprocessing to test data
test <- test %>%
  arrange(pid, desc(dtime)) %>%
  group_by(pid) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(across(C61:Z98.890, ~replace_na(as.numeric(.) * 1L, 0)))


# Exclude any rows in merged_data that exist in test from the training set
training_set <- anti_join(merged_data, test, by = "pid")

# Extract the subset for model fitting
model_data <- training_set %>%
  select(nicm:dilated, C61:Z98.890)

# Define the list to store models and predictions
rf_models <- list()
predictions <- list()

# Build univariate models for all the columns from nicm to dilated
outcomes <- names(model_data %>% select(nicm:dilated))

aucs <- c()
f1_scores <- c()
precisions <- c()
recalls <- c()
total_TP <- 0
total_FP <- 0
total_FN <- 0

true_label_matrix <- matrix(0, nrow(test), length(outcomes))
predicted_label_matrix <- matrix(0, nrow(test), length(outcomes))
predicted_prob_matrix <- matrix(0, nrow(test), length(outcomes))
for (outcome in outcomes) {
  # Change to factor
  training_set[[outcome]] <- as.factor(training_set[[outcome]])
  
  column_indices <- which(colnames(training_set) %in% c("C61", "Z98.890"))
  column_names <- colnames(training_set)[column_indices[1]:column_indices[2]]
  
  # Exclude outcome variables from predictors
  predictor_columns <- setdiff(column_names, outcomes)
  current_training_data <- training_set[, c(outcome, predictor_columns)]
  
  # Fit the model
  univariate_model <- imbalanced(as.formula(paste(outcome, "~ .")), data = as.data.frame(current_training_data), nodesize=8, mtry=15, ntree = 1500, importance=TRUE, na.action="na.omit")
  
  # Store the model in the list
  rf_models[[outcome]] <- univariate_model
  
  # Predict on the test set and store the predictions
  current_test_data <- test[, c(outcome, predictor_columns)]
  predictions[[outcome]] <- predict(rf_models[[outcome]], newdata = as.data.frame(current_test_data))
  
  # # Print outcome
  # print(paste("Outcome:", outcome))
  # 
  # # Print model summary
  # print(paste("Model Summary for", outcome, ":"))
  # print(rf_models[[outcome]])
  
  # Print model imbalanced performance
  performance = get.imbalanced.performance(rf_models[[outcome]])
  print(paste("Imbalanced Performance of Model for", outcome, ":"))
  print(performance)
  
  # Print predictions
  print(paste("Predictions for", outcome, ":"))
  print(predictions[[outcome]])
  
  true_labels <- as.numeric(as.character(test[[outcome]]))
  predicted_classes <- as.numeric(as.character(predictions[[outcome]]$class))
  predicted_probs <- predictions[[outcome]]$predicted[,2]
  
  true_label_matrix[, which(outcomes == outcome)] <- true_labels
  predicted_label_matrix[, which(outcomes == outcome)] <- predicted_classes
  predicted_prob_matrix[, which(outcomes == outcome)] <- predicted_probs
  
  # Compute TP and FP for the current class
  class_TP <- sum(true_labels == 1 & predicted_classes == 1)
  class_FP <- sum(true_labels == 0 & predicted_classes == 1)
  class_FN <- sum(true_labels == 1 & predicted_classes == 0)
  
  # Accumulate the values
  total_TP <- total_TP + class_TP
  total_FP <- total_FP + class_FP
  total_FN <- total_FN + class_FN
  
  # Print imbalanced performance of predictions
  print(paste("Imbalanced Performance of Predictions for", outcome, ":"))
  print(get.imbalanced.performance(predictions[[outcome]]))
  
  # Compute AUC
  aucs <- c(aucs, performance['auc'])
  f1_scores <- c(f1_scores, performance['F1'])
  precisions <- c(precisions, performance['prec'])
  recalls <- c(recalls, performance['sens'])
  
  # print separation line
  cat("\n--------------------------------------------------\n")
}
# Compute micro-averaged precision
micro_precision <- total_TP / (total_TP + total_FP)
micro_recall <-  total_TP / (total_TP + total_FN)
micro_f1 <- 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
correct_predictions_count <- sum(rowSums(true_label_matrix == predicted_label_matrix) == length(outcomes))
subset_accuracy <- correct_predictions_count / nrow(test)

# Print micro-averaged metrics
cat("Micro-averaged F1 Score:", micro_f1, "\n")
cat("Micro-averaged Precision:", micro_precision, "\n")
cat("Micro-averaged Recall:", micro_recall, "\n")
cat("subsetacc", subset_accuracy, '\n')

# Reshape the matrices to vectors
all_true_labels <- as.vector(true_label_matrix)
all_predicted_probs <- as.vector(predicted_prob_matrix)

# Compute micro-averaged ROC and AUC
micro_roc <- roc(all_true_labels, all_predicted_probs)
micro_auc <- auc(micro_roc)

# Print the micro-averaged AUC
cat("Micro-averaged AUC:", micro_auc, "\n")
