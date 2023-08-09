library(data.table, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(bit, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(bit64, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")

labels <- fread("/data/aiiih/projects/ts_nicm/data/nicm_labels.csv")
echo <- fread("/data/aiiih/data/ehr/domains/proc_echo.csv")

# # Exclude rows with NA in 'Event_Stop_Date' or 'Event_Stop_Time'
# echo <- echo[complete.cases(echo[, .(Event_Stop_Date, Event_Stop_Time)]), ]

# Calculate number of rows and unique ids
initial_rows <- nrow(echo)
initial_unique_ids <- length(unique(echo$Clarity_PAT_ID))
cat("Number of initial rows in echo: ", format(initial_rows, big.mark = ","), "\n")
cat("Number of initial unique ids in echo: ", format(initial_unique_ids, big.mark = ","), "\n")

# Exclude rows with NA or "NULL" in 'Event_Stop_Date' or 'Event_Stop_Time'
echo <- echo[complete.cases(echo[, .(Event_Stop_Date, Event_Stop_Time)]) 
             & echo$Event_Stop_Date != "NULL" & echo$Event_Stop_Time != "NULL", ]

# Calculate number of rows and unique ids after removal of NA
final_rows <- nrow(echo)
final_unique_ids <- length(unique(echo$Clarity_PAT_ID))

# Calculate and print number of removed rows and unique ids
removed_rows <- initial_rows - final_rows
removed_unique_ids <- initial_unique_ids - final_unique_ids
cat("Number of removed rows: ", format(removed_rows, big.mark = ","), "\n")
cat("Number of removed unique ids: ", format(removed_unique_ids, big.mark = ","), "\n")

# print divider
cat("--------------------------------------------------\n")


# Get unique 'PatientID' from labels and 'Clarity_PAT_ID' from echo
unique_labels <- unique(labels$PAT_ID)
unique_echo <- unique(echo$Clarity_PAT_ID)
cat("Number of remaining unique echo ids: ", format(length(unique_echo), big.mark = ","), "\n")
cat("Number of unique nicm labels: ", format(length(unique_labels), big.mark = ","), "\n")


# Find shared, unique values
common <- intersect(unique_labels, unique_echo)

# Filter labels and echo to only include rows with common IDs
labels_common <- labels[labels$PAT_ID %in% common, ]
echo_common <- echo[echo$Clarity_PAT_ID %in% common, ]

# Convert 'Event_Stop_Date' to Date object
echo_common[, Event_Stop_Date := as.Date(Event_Stop_Date)]

# Sort the data by 'Event_Stop_Date'
setorder(echo_common, -Event_Stop_Date)

# Select the most recent 'Event_Stop_Date' for each 'Clarity_PAT_ID'
echo_common <- echo_common[, .SD[1], by = Clarity_PAT_ID]

# Merge the two data tables on 'PAT_ID' and 'Clarity_PAT_ID'
merged_common <- merge(labels_common, echo_common, by.x = "PAT_ID", by.y = "Clarity_PAT_ID")

# Select the 'K_PAT_KEY' from labels and 'Event_Stop_Date' from echo_common
common_dt <- merged_common[, .(pid = K_PAT_KEY, echo_dt = Event_Stop_Date)]

# Cast 'pid' to character
common_dt[, pid := as.character(pid)]

# Write the 'K_PAT_KEY' values and most recent 'Event_Stop_Date' to a CSV file
fwrite(common_dt, "/data/aiiih/projects/ts_nicm/data/common_ids.csv", row.names = FALSE)

# Print the number of common IDs
cat("Number of common IDs after processing: ", format(length(common), big.mark = ","), "\n")