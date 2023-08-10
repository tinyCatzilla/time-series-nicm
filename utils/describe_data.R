library(data.table, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(bit, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")
library(bit64, lib.loc="~/data/aiiih/pkgs/R/R-4.2.2/lib64/R/library")

labels_clean <- fread("/data/aiiih/projects/ts_nicm/data/nicm_labels_clean.csv") 
labels_cmr <- fread("/data/aiiih/projects/ts_nicm/data/cmr/labels_processed.csv") 
labels_echo <- fread("/data/aiiih/projects/ts_nicm/data/echo/labels_processed.csv") 
dx <- fread("/data/aiiih/projects/ts_nicm/data/dx_nicm.csv")
dx_cmr <- fread("/data/aiiih/projects/ts_nicm/data/cmr/dx_processed.csv")
dx_echo <- fread("/data/aiiih/projects/ts_nicm/data/echo/dx_processed.csv")


#subset echo to only include patients in cmr
dx_echo <- dx_echo[dx_echo$pid %in% dx_cmr$pid,]
labels_echo <- labels_echo[labels_echo$pid %in% dx_echo$pid,]

# Compute generic stats for datasets with sequence data (dx datasets)
compute_sequence_stats <- function(dt, pid_col, dtime_col = NULL) {

    num_unique_patients <- length(unique(dt[[pid_col]]))

    # If there's a dtime_col, compute sequence-based stats
    if (!is.null(dtime_col)) {
        avg_seq_len <- nrow(dt) / num_unique_patients
        last_observation <- dt[order(-dt[[dtime_col]]), .SD[1], by = pid_col]
        avg_ones_last_obs <- mean(rowSums(last_observation[, !c(pid_col, dtime_col), with = FALSE] == 1, na.rm = TRUE))
    }
    else {
        avg_seq_len <- 0
        avg_ones_last_obs <- 0
    }

    return(list(num_unique_patients, avg_seq_len, avg_ones_last_obs))
}

# Modified get_class_stats to account for unique pids
get_class_stats <- function(dt, pid_col) {
    class_cols <- colnames(dt)[-(1:2)]
    class_counts <- sapply(class_cols, function(col) {
        sum(!is.na(dt[[col]]) & dt[[col]] > 0)
    }, simplify = "vector")
    
    # Count unique patients for proportion calculation
    total_patients <- length(unique(dt[[pid_col]]))
    class_proportions <- class_counts / total_patients
    
    return(list(class_counts, class_proportions))
}


# Table 1: General statistics
latex_table1 <- "\\begin{tabular}{lrrr}\n"
latex_table1 <- paste(latex_table1, "\\toprule \n", sep = "")
latex_table1 <- paste(latex_table1, " \\textbf{Dataset} & \\textbf{Num. Patients} & \\textbf{Avg. sequence length} & \\textbf{Avg. unique diagnoses per patient} \\\\\n", sep = "")
latex_table1 <- paste(latex_table1, " \\midrule\n", sep = "")

datasets_pairs <- list(clean=list(labels=labels_clean, dx=dx), cmr=list(labels=labels_cmr, dx=dx_cmr), echo=list(labels=labels_echo, dx=dx_echo))
names <- c("clean", "cmr", "echo")
# Populate Table 1
for(name in names) {
    dx_dt <- datasets_pairs[[name]]$dx
    
    if (name == "clean") {
        seq_stats <- compute_sequence_stats(dx_dt, "K_PAT_KEY")
    } else {
        seq_stats <- compute_sequence_stats(dx_dt, "pid", "dtime")
    }
    
    latex_row <- paste0(name, " & ", seq_stats[[1]], " & ", round(seq_stats[[2]], 2), " & ", round(seq_stats[[3]], 2), " \\\\\n")
    latex_table1 <- paste(latex_table1, latex_row, sep = "")
}

latex_table1 <- paste(latex_table1, "\\bottomrule\n", sep = "")
latex_table1 <- paste(latex_table1, "\\end{tabular}\n", sep = "")
cat(latex_table1)

# Table 2: Class data statistics
latex_table2 <- "\\begin{tabular}{l"
for (name in names) {
    latex_table2 <- paste0(latex_table2, "rr")
}
latex_table2 <- paste0(latex_table2, "}\n")

latex_table2 <- paste(latex_table2, "\\toprule \n", sep = "")
header_row <- " \\textbf{Class} "
for (name in names) {
    header_row <- paste0(header_row, "& \\textbf{Count (", name, ")} & \\textbf{Prop. (", name, ")} ")
}
header_row <- paste0(header_row, "\\\\\n")
latex_table2 <- paste(latex_table2, header_row, sep = "")
latex_table2 <- paste(latex_table2, " \\midrule\n", sep = "")


# Populate Table 2
for(col in colnames(labels_clean)[-c(1,2)]) {
    latex_row <- col
    for(name in names) {
        pid_col <- ifelse(name == "clean", "K_PAT_KEY", "pid")
        # Subset labels_dt based on common pid_col values in the dx dataset
        labels_dt <- datasets_pairs[[name]]$labels[datasets_pairs[[name]]$labels[[pid_col]] %in% datasets_pairs[[name]]$dx[[pid_col]], ]
        
        stats <- get_class_stats(labels_dt, pid_col)
        count = stats[[1]][col]
        prop = stats[[2]][col]
        latex_row <- paste0(latex_row, " & ", count, " & ", round(prop, 4))
    }
    latex_row <- paste0(latex_row, " \\\\\n")
    latex_table2 <- paste(latex_table2, latex_row, sep = "")
}

latex_table2 <- paste(latex_table2, "\\bottomrule\n", sep = "")
latex_table2 <- paste(latex_table2, "\\end{tabular}\n", sep = "")
cat(latex_table2)