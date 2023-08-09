# Install and load the required packages
library(dplyr)
library(data.table)

# Read the data into data.table
df <- fread("nicm/dx_nicm.csv")

# Count the number of unique values in 'K_PAT_KEY' column
num_unique_values <- df %>% select(K_PAT_KEY) %>% unique() %>% nrow()
num_unique_values_labels <- labels %>% select(pid) %>% unique() %>% nrow()

# Print the number of unique values
print(paste("Number of unique values in 'K_PAT_KEY':", num_unique_values))
print(paste("Number of unique values in 'labels':", num_unique_values_labels))


# Create a frequency table
freq_table <- table(df$K_PAT_KEY)

# Count the number of entries that appear only once
num_single_occurrences <- sum(freq_table == 1)

# Print the result
print(paste("Number of 'K_PAT_KEY' values that appear only once:", num_single_occurrences))
