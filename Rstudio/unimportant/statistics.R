# install.packages("tidyverse")
library(tidyverse)
# install.packages("ggplot2")
library(ggplot2)
# install.packages("fuzzyjoin")
library(fuzzyjoin)
# install.packages("ggpubr")
library(ggpubr)
library(randomForestSRC)
library(tidyr)
library(dplyr)



echo_vars <- read_csv('cm_echo_vars.csv')
disease_vars <- read_csv("combined_disease_label.csv")

# Fix weird columns
disease_vars = subset(disease_vars, select = -StudyDate_x)
disease_vars <- disease_vars %>% rename("StudyDate" = "StudyDate_y")

if( any(echo_vars$ef_echo == "4440"))
{
  echo_vars[ which(echo_vars$ef_echo == "4440")[1] ,"ef_echo"] <- "44.00"
}

# Convert the StudyDate in disease_vars to Date type
disease_vars <- disease_vars %>%
  mutate(StudyDate = as.Date(as.character(StudyDate), format = "%Y%m%d"))

# Find the most recent date for each group in disease_vars, keep only one record per group
recent_disease_vars <- disease_vars %>%
  group_by(Enterprise_ID) %>%
  slice_max(StudyDate, n = 1, with_ties=FALSE) %>%
  ungroup()

# Left join to merge with the most recent date
newdata <- left_join(echo_vars, recent_disease_vars, 
                     by = c("enterprise_patient_id" = "Enterprise_ID"))

# Remove temporary recent_disease_vars
rm(recent_disease_vars)

# Rename some columns
newdata <- newdata %>% rename("non_compaction" = "non-compaction")
newdata <- newdata %>% rename("aortic_dis" = "aortic-dis")


# Select a range of columns from disease_vars
# Dont forget echotypenew... eventually...
selected_columns <- names(newdata %>% select(nicm:tavr))
# List of possible dependent variables
dependent_vars <- names(newdata %>% select(ef_echo:rvedvol))

# Empty list to store names of categorical columns
dep_cat <- list()
dep_quant <- list()

# Function to check if a column contains only numbers, NULLs, and NAs
is_numeric_col <- function(col) {
  all(sapply(col, function(x) {
    is.na(x) | !is.na(as.numeric(as.character(x))) | x == "NULL"
  }))
}

# Iterate through each column and cast to numeric or add to vois_cat
for (col_name in dependent_vars) {
  col_data <- newdata[[col_name]]
  if (is_numeric_col(col_data)) {
    # Cast to numeric
    newdata[[col_name]] <- as.numeric(newdata[[col_name]])
    dep_quant <- c(dep_quant,col_name)
  } else {
    # Add to list of categorical columns
    dep_cat <- c(dep_cat, col_name)
  }
}

# List of date columns to exclude
date_columns <- c("dtm_echo", "StudyDate")

# Columns to perform the replacement on
non_date_columns <- setdiff(colnames(newdata), date_columns)

# Iterate through non-date columns
for (column_name in non_date_columns) {
  # Replace "NULL" with NA in the current column
  newdata[[column_name]][newdata[[column_name]] == "NULL"] <- NA
  newdata[[column_name]][newdata[[column_name]] == "N/A"] <- NA
}

for (column_name in dep_cat) {
  # Convert strings to uppercase
  newdata[[column_name]] <- toupper(as.character(newdata[[column_name]]))
  
  # Change to factor
  newdata[[column_name]] <- as.factor(newdata[[column_name]])
}

#di_fun choice is questionable
newdata[["di_fun"]][is.na(newdata[["di_fun"]])] <- "NOT_EVALUATED"
newdata[["ttime"]][newdata[["ttime"]]=="BASE"] <- "BASELINE"
newdata[["ttime"]][newdata[["ttime"]]=="BASELINE  TVI OF"] <- "BASELINE"
newdata[["lv_vasbs"]][newdata[["lv_vasbs"]]=="RCA,LAD,LCX"] <- "LAD,LCX,RCA"
newdata[["lv_vasbs"]][newdata[["lv_vasbs"]]=="RCA,LCX,LAD"] <- "LAD,LCX,RCA"
#lv_vasbs RX etc.
#av_sten Y vs MILD/MOD.... N vs NA.
#di_funrc con vs restrictive 
#la_throm NA vs N? Y vs SMALL/LARGE
#lv_testr isc vs iscar, nd vs neg, scar vs svia?



allplots <- list()
# Loop through each independent variable
for (independent_var in selected_columns) {
  
  # Change to factor
  newdata[[independent_var]] <- as.factor(newdata[[independent_var]])
  
  # Data for the current iteration
  current_data <- newdata[, c(dependent_vars, independent_var)]
  
  # mForest imputation
  imputed_data <- impute(data = current_data, ntree=1000, nsplit=10, fast=TRUE, mf.q = 0.25, max.iter = 50, verbose=TRUE)
  
  # Build random forest model
  rf_model <- rfsrc(as.formula(paste(independent_var, "~ .")), data = imputed_data,
                    ntree = 1500, na.action = "na.impute", importance=TRUE)
  
  # Extract feature importance
  importance_scores <- rf_model$importance[,1]
  
  # Sort the importance scores in descending order
  sorted_importance <- sort(importance_scores, decreasing = TRUE)
  
  # Calculate relative importance
  sorted_importance <- sorted_importance/sorted_importance[1]
  
  # Initialize list for the current independent variable
  ivar_plots <- list()
  
  # Store all feature importance in ivar_plots, and >=.02 in vois
  ivar_plots$feature_importance <- sorted_importance
  vois <- names(sorted_importance[sorted_importance >= 0.02])
  
  # Make new dataset to not bork rf
  plotdata <- newdata
  
  # List of columns
  vois_quant <- intersect(vois,dep_quant)
  vois_cat <- intersect(vois,dep_cat)

  
  # Code for violin plots
  quant_plot_list <- list()
  for (dependent_var in vois_quant) {
    
    # Create a violin plot for the current dependent variable
    p_violin <- ggplot(plotdata, aes_string(x = independent_var, y = dependent_var, color = independent_var)) +
      geom_violin(trim = TRUE) +
      geom_jitter(width = 0, height = 0, size = 0.7, color="black",
                  data = subset(plotdata, eval(parse(text = paste0("abs(", dependent_var, " - median(", dependent_var, ", na.rm = TRUE)) > 15 * mad(", dependent_var, ", na.rm = TRUE)"))))) +
      stat_summary(fun = mean, geom = "crossbar", size=0.2, width=0.3, color = "sienna3") +
      stat_summary(fun = median, geom = "point", shape = 4, size = 2, color = "slateblue1") +
      labs(title = paste("Violin Plot of", independent_var, "vs", dependent_var),
           x = independent_var,
           y = dependent_var)
    
    # Add the plots to the list
    quant_plot_list[[length(quant_plot_list) + 1]] <- p_violin
  }
  
  # Store violin plots in ivar_plots
  ivar_plots$violin_plots <- quant_plot_list
  allplots[[independent_var]] <- ivar_plots
  
  
  # Code for bar charts
  bar_plot_list <- list()
  for (dependent_var in unlist(vois_cat)) {
    
    # Find max for y axis
    max_value <- max(table(plotdata[[independent_var]], plotdata[[dependent_var]])) * 1.05

      # Create a bar plot for the current dependent variable
    p <- ggplot(plotdata, aes_string(x = independent_var, fill = dependent_var)) +
      geom_bar(position = "dodge", stat="count") +
      labs(title = paste("Grouped Bar Chart of", independent_var, "vs", dependent_var),
           x = independent_var,
           y = "Number of Observations") +
      geom_text(stat = "count", aes(label = after_stat(count), colour = ..fill..), vjust = -1, position=position_dodge(0.9)) + 
      ylim(0, max_value) + 
      guides(colour = "none")
    
    # Add the plot to bar_plot_list
    bar_plot_list[[length(bar_plot_list) + 1]] <- p
    
  }
  # Store bar charts in ivar_plots
  ivar_plots$bar_charts <- bar_plot_list
  allplots[[independent_var]] <- ivar_plots
}


# rm(vois,vois_quant,vois_cat,bar_plot_list,importance_scores,dependent_var,independent_var, dependent_vars,
#    sorted_importance, col_data, col_name,quant_plot_list,ivar_plots,rf_model,plotdata)

save.image("~/echo-proj/workspace.RData")

