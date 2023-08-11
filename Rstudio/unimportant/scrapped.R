
# This cares about time data, but it seems we are lacking information as a result.
# # Create a temporary date column in echo_vars without time for the join
# echo_vars <- echo_vars %>%
#   mutate(temp_date = as.Date(substr(dtm_echo, 1, 10), format = "%Y-%m-%d"))
# 
# # Convert the StudyDate_y in disease_vars to Date type
# disease_vars <- disease_vars %>%
#   mutate(StudyDate_y = as.Date(as.character(StudyDate_y), format = "%Y%m%d"))
# 
# # Perform left join using join_by with non-equality conditions
# newdata <- echo_vars %>%
#   left_join(
#     disease_vars,
#     by = join_by(
#       enterprise_patient_id == Enterprise_ID,
#       ccfid == PatientID,
#       temp_date >= StudyDate_y
#     ),
#     copy = TRUE
#   ) %>%
#   group_by(enterprise_patient_id, ccfid, temp_date) %>%
#   slice_head(n = 1) %>%
#   ungroup() %>%
#   select(-temp_date) # Remove the temporary date column





# Original code for violin plots
# Loop through each independent variable
for (column_name in selected_columns) {
  
  # Create a list to store plots for each dependent variable
  plot_list <- list()
  
  # Loop through each dependent variable
  for (dependent_var in dependent_vars) {
    
    # Create a violin plot for the current dependent variable
    p <- ggplot(newdata, aes_string(x = column_name, y = dependent_var, fill = as.factor(newdata[[column_name]]))) +
      geom_violin() +
      labs(title = paste("Violin Plot of", column_name, "vs", dependent_var),
           x = column_name,
           y = dependent_var) +
      scale_fill_discrete(name = "Value")
    
    # Add the plot to the list
    plot_list[[length(plot_list) + 1]] <- p
  }
  
  # Combine plots for each dependent variable into a single plot
  combined_plot <- ggpubr::ggarrange(plotlist = plot_list, ncol = 1)
  
  # Display the combined plot
  print(combined_plot)
}
