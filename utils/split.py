# Split dataset
trainFrac = 0.70  # fraction of data to use for training
valFrac = 0.15  # fraction of data to use for validation
testFrac = 0.15  # fraction of data to use for testing
input_filepath = "/data/aiiih/projects/ts_nicm/data/cmr/nicm_processed.csv"
label_filepath = "/data/aiiih/projects/ts_nicm/data/cmr/labels_processed.csv"


# Read the dataframe
df = pd.read_csv(input_filepath)

# Load label dataframe
df_labels = pd.read_csv(label_filepath)

# Get unique patient IDs
patient_ids = df[patient_id_col].unique()

# Assert that they add up to 1
assert trainFrac + valFrac + testFrac == 1, "Fractions do not add up to 1!"

# Create train/validation/test split
train_pids, temp_pids = train_test_split(patient_ids, test_size=1-trainFrac, random_state=400)
val_pids, test_pids = train_test_split(temp_pids, test_size=testFrac/(valFrac + testFrac), random_state=400)