# install.packages("R.utils", lib="/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library", repos = "https://cran.r-project.org")
# install.packages("R.oo", lib="/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library", repos = "https://cran.r-project.org")
# install.packages("R.methodsS3", lib="/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library", repos = "https://cran.r-project.org")

library(data.table, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(bit, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(bit64, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(R.methodsS3, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(R.oo, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(R.utils, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")


# Specify your root directory
root_directory <- "/data/aiiih/projects/ts_nicm/results/base"

# List all csv files under the root directory, including those in subdirectories
csv_files <- list.files(root_directory, pattern = "_eval_calibrated.csv$", recursive = TRUE)

# Define the columns to keep
cols_to_keep <- c("Model", "Accuracy", "ROC AUC Score", "F1 Score", "Recall", "Precision")

# Initialize a list to hold all data tables
data_list <- list()

# Iterate over all csv files
for (csv_file in csv_files) {
  # Read the csv file keeping only the specified columns
  dt <- fread(file.path(root_directory, csv_file), select = cols_to_keep)
  
  # Add the data table to the list
  data_list[[csv_file]] <- dt
}

# Combine all data tables into a single data table
combined_dt <- rbindlist(data_list)

# Group by 'Model', and calculate median for each numeric column
median_values <- combined_dt[, lapply(.SD, median, na.rm = TRUE), by = Model]

# Generate LaTeX table
latex_table <- "\\begin{tabular}{lccccc}\n"
latex_table <- paste(latex_table, "\\toprule \n")
latex_table <- paste(latex_table, "Class & F1 Score & ROC AUC Score & Recall & Precision \\\\\n")
latex_table <- paste(latex_table, "\\midrule\n")

# Append each row
for (i in 1:nrow(median_values)) {
  model <- median_values$Model[i]
  values <- sprintf("%.4f", as.numeric(median_values[i, .SD, .SDcols = -"Model"]))
  latex_row <- sprintf("%s & %s & %s & %s & %s \\\\\n", model, values[3], values[2], values[4], values[5])
  latex_table <- paste(latex_table, latex_row)
}

latex_table <- paste(latex_table, "\\bottomrule\n")
latex_table <- paste(latex_table, "\\end{tabular}\n")

cat(latex_table)