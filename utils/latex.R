library(data.table, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(bit, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(bit64, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")

# Specify your root directory
root_directory <- "/data/aiiih/projects/ts_nicm/results/base"

# List all csv files under the root directory, including those in subdirectories
csv_files <- list.files(root_directory, pattern = "_eval_calibrated.csv$", recursive = TRUE)

# Define the columns to keep
cols_to_keep <- c("Model", "Subset Accuracy", "ROC AUC Score", "F1 Score", "Precision", "Recall")

# Initialize a list to hold all data tables
data_list <- list()

# Iterate over all csv files
for (csv_file in csv_files) {
  # Read the csv file keeping only the specified columns
  dt <- fread(file.path(root_directory, csv_file), select = cols_to_keep)

  # Filter for the 'Full Model' row
  full_model_row <- dt[Model == "Full Model"]
  
  # Retrieve model name from the file name and add it as a column
  model_name <- unlist(strsplit(basename(sub("_1_eval_calibrated.csv", "", csv_file)), "/"))[1]
  full_model_row$Model <- model_name
  
  # Add the filtered row with model name to the list
  data_list[[csv_file]] <- full_model_row
}

# Combine all data tables into a single data table
combined_dt <- rbindlist(data_list)

# Generate LaTeX table
latex_table <- sprintf("\\begin{tabular}{r|ccccc}\n")
latex_table <- paste(latex_table, "\\toprule \n", sep = "")
latex_table <- paste(latex_table, " \\textbf{Model} & \\textbf{Subset Acc.} & \\textbf{F1} & \\textbf{ROC AUC} & \\textbf{Recall} & \\textbf{Prec}\\\\\n", sep = "")
latex_table <- paste(latex_table, " \\midrule\n", sep = "")

# Append each row
for (i in 1:nrow(combined_dt)) {
  model <- combined_dt$Model[i]
  values <- sprintf("%.4f", as.numeric(combined_dt[i, .SD, .SDcols = -"Model"]))
  latex_row <- sprintf("%s & %s & %s & %s & %s & %s \\\\\n", model, values[1], values[3], values[2], values[5], values[4])
  latex_table <- paste(latex_table, latex_row, sep = "")
}

latex_table <- paste(latex_table, "\\bottomrule\n", sep = "")
latex_table <- paste(latex_table, "\\end{tabular}\n", sep = "")

cat(latex_table)