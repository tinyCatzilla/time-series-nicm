# Load the necessary library
library(ggplot2)


table1 <- data.frame(
  Model = c("RNN", "LSTM", "GRU", "TransformerRNN", "TransformerLSTM", "TransformerGRU", "TransformerModel", "TST", "RNNAttention", "LSTMAttention", "GRUAttention", "ensemble"),
  F1_Score = c(0.5709, 0.5620, 0.5694, 0.5736, 0.5593, 0.5873, 0.5719, 0.5853, 0.5651, 0.5693, 0.5645, 0.6032),
  Table = rep("With LOCF", 12)
)

table2 <- data.frame(
  Model = c("RNN", "LSTM", "GRU", "TransformerRNN", "TransformerLSTM", "TransformerGRU", "TransformerModel", "TST", "RNNAttention", "LSTMAttention", "GRUAttention", "ensemble"),
  F1_Score = c(0.3279, 0.4884, 0.4925, 0.5439, 0.5197, 0.5422, 0.5492, 0.5289, 0.5323, 0.5097, 0.5261, 0.5585),
  Table = rep("Without LOCF", 12)
)



# Combine the two tables
combined_data <- rbind(table1, table2)

# Create a more refined color palette
color_palette <- c("With LOCF" = "#00BFC4", "Without LOCF" = "#F8766D")

# Set the order of the bars
combined_data$Model <- factor(combined_data$Model, levels = c("RNN", "LSTM", "GRU", "TransformerRNN", "TransformerLSTM", "TransformerGRU", "TransformerModel", "TST", "RNNAttention", "LSTMAttention", "GRUAttention", "ensemble"))

# Create the graph as before
ggplot(combined_data, aes(fill=Table, y=F1_Score, x=Model)) + 
  geom_bar(position=position_dodge(width=0.8), stat="identity", width=0.7) +
  scale_fill_manual(values=color_palette) + # Set color palette
  labs(
    y = "F1 Score",
  ) +
  theme_minimal() + 
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.x = element_blank(),
    legend.title = element_blank(),
    legend.position = "bottom"
  )
