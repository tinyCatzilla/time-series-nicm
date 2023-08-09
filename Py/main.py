from m_dataset import mDataset

from fastai.callback.progress import CSVLogger
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.metrics import Metric, accuracy_multi, F1ScoreMulti, HammingLossMulti, RecallMulti, PrecisionMulti, RocAucMulti, accuracy, RocAuc, F1Score, Recall, Precision
from fastai.callback.schedule import minimum, steep, valley, slide
from fastai.test_utils import *
from fastai.losses import *

from tsai.models.TSTPlus import TSTPlus
from tsai.models.PatchTST import PatchTST
from tsai.models.RNNPlus import RNNPlus, LSTMPlus, GRUPlus
from tsai.models.RNNAttention import RNNAttention, LSTMAttention, GRUAttention
from tsai.models.TransformerModel import TransformerModel
from tsai.models.TransformerRNNPlus import TransformerRNNPlus, TransformerLSTMPlus, TransformerGRUPlus
from tsai.data.all import TSDatasets, TSDataLoader, TSDataLoaders
from tsai.data.transforms import MultiCategorize
from tsai.analysis import feature_importance
from tsai.learner import ts_learner

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, recall_score
from scipy.special import expit  # sigmoid function

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, math, csv


from imblearn.over_sampling import RandomOverSampler



################################## VARIABLES ###################################

# Split dataset
trainFrac = 0.70  # fraction of data to use for training
valFrac = 0.15  # fraction of data to use for validation
testFrac = 0.15  # fraction of data to use for testing
input_filepath = "/data/aiiih/projects/ts_nicm/data/nicm_combined.csv"
label_filepath = "/data/aiiih/projects/ts_nicm/data/labels_sani.csv"
data_dir = "/data/aiiih/projects/ts_nicm/data"
patient_id_col = 'pid'


# Create dataloaders
batch_size = 64
n_features = 185
n_classes = 7

# Training parameters
max_epochs = 45
verbose = False
# metrics can be changed in ### METRICS ### section
# loss_func can be changed in ### LOSS ### section
classes_to_fit = [0,1,2,3,4,5,6] # with isotonic regression, starting w 0

# Output
output_dir = "/data/aiiih/projects/ts_nicm/results/base"

################################## INSTANTIATE ###################################
# Read the dataframe
df = pd.read_csv(input_filepath)

# fill NaNs with 0 before dataset
df = df.fillna(0)

# Load label dataframe
df_labels = pd.read_csv(label_filepath)

# normalize labels
class_names = df_labels.columns.tolist()
index_to_classname = {i-2: classname for i, classname in enumerate(class_names)}
print(index_to_classname)


# Get unique patient IDs
patient_ids = df[patient_id_col].unique()

# Assert that they add up to 1
assert trainFrac + valFrac + testFrac == 1, "Fractions do not add up to 1!"

# Create train/validation/test split
train_pids, temp_pids = train_test_split(patient_ids, test_size=1-trainFrac, random_state=400)
val_pids, test_pids = train_test_split(temp_pids, test_size=testFrac/(valFrac + testFrac), random_state=400)

# Split the dataframe into training, validation, and testing according to the pids
train_df = df[df[patient_id_col].isin(train_pids)]
val_df = df[df[patient_id_col].isin(val_pids)]
test_df = df[df[patient_id_col].isin(test_pids)]

# Save the train, validation, and test datasets
train_df.to_csv(os.path.join(data_dir, "data_train.csv"), index=False)
val_df.to_csv(os.path.join(data_dir, "data_val.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "data_test.csv"), index=False)
print("Train, validation, and test datasets saved")

# Split the dataframe into training and validation according to the pids
train_df = df[df[patient_id_col].isin(train_pids)]
val_df = df[df[patient_id_col].isin(val_pids)]
test_df = df[df[patient_id_col].isin(test_pids)]

# assuming mDataset is predefined
train_ds = mDataset(train_df, df_labels)
val_ds = mDataset(val_df, df_labels)
test_ds = mDataset(test_df, df_labels)

X_train = [train_ds[i][0] for i in range(len(train_pids))]
y_train = [train_ds[i][1] for i in range(len(train_pids))]

X_valid = [val_ds[i][0] for i in range(len(val_pids))]
y_valid = [val_ds[i][1] for i in range(len(val_pids))]

X_test = [test_ds[i][0] for i in range(len(test_pids))]
y_test = [test_ds[i][1] for i in range(len(test_pids))]

# Find the maximum sequence length
max_seq_len = max(max(len(x) for x in X_train), max(len(x) for x in X_valid), max(len(x) for x in X_test))

X_train_list = list(X_train)
X_valid_list = list(X_valid)
X_test_list = list(X_test)

all_sequences = X_train_list + X_valid_list + X_test_list

# Average sequence length
average_seq_len = sum(len(x) for x in all_sequences) / len(all_sequences)

# Number of '1's in the last observation of each sequence
num_ones_last_observation = [np.sum(x[-1] == 1) for x in all_sequences]

# Average number of '1's in the last observation
average_ones_last_observation = sum(num_ones_last_observation) / len(all_sequences)

print(f"Average sequence length: {average_seq_len}", flush=True)
print(f"Average number of '1's in the last observation: {average_ones_last_observation}", flush=True)


# Pad sequences in X_train and X_valid
X_train_padded = [np.pad(x, ((0, max_seq_len - len(x)), (0, 0)), constant_values=np.nan) for x in X_train]
X_valid_padded = [np.pad(x, ((0, max_seq_len - len(x)), (0, 0)), constant_values=np.nan) for x in X_valid]
X_test_padded = [np.pad(x, ((0, max_seq_len - len(x)), (0, 0)), constant_values=np.nan) for x in X_test]

# Convert lists to numpy arrays
X_train_padded = np.array(X_train_padded)
X_valid_padded = np.array(X_valid_padded)
X_test_padded = np.array(X_test_padded)

# Stack y_train and y_valid
y_train = np.stack(y_train)
y_valid = np.stack(y_valid)
y_test = np.stack(y_test)

# Concatenate X_train_padded and X_valid_padded, and y and y_valid
X_all = np.concatenate([X_train_padded, X_valid_padded])
X_all = X_all.transpose(0, 2, 1)
y_all = np.concatenate([y_train.squeeze(), y_valid.squeeze()], axis=0)
print(f"X_all shape: {X_all.shape}, y_all shape: {y_all.shape}", flush=True)
print(f"X_all[0]: {X_all[0]}, y_all[0]: {y_all[0]}", flush=True)

# Test set
X_test_padded = X_test_padded.transpose(0, 2, 1)
y_test = y_test.squeeze()
print(f"X_test_padded shape: {X_test_padded.shape}, y_test shape: {y_test.shape}", flush=True)
print(f"X_test_padded[0]: {X_test_padded[0]}, y_test[0]: {y_test[0]}", flush=True)

# Create splits
splits = (range(len(X_train_padded)), range(len(X_train_padded), len(X_all)))


print(f"n_features: {n_features}, n_classes: {n_classes}, max_seq_len: {max_seq_len}", flush=True)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


################################## METRICS ###################################

def subset_accuracy(y_pred, y_true):
    y_pred = y_pred.sigmoid() > 0.5
    return (y_pred == y_true).all(dim=-1).float().mean()

class AvgProbPositiveClass(Metric):
    def __init__(self): self.preds_positives, self.count = [], 0

    def accumulate(self, learn):
        preds, targs = learn.pred, learn.y
        preds_sigmoid = torch.sigmoid(preds)
        self.preds_positives.append(preds_sigmoid[targs.bool()].cpu().numpy())
        self.count += targs.sum().item()
        
    @property
    def value(self):
        all_preds = np.concatenate(self.preds_positives)
        return all_preds.mean() if len(all_preds) > 0 else None

class AvgProbMissedPositiveClass(Metric):
    def __init__(self): self.preds_missed, self.count = [], 0

    def accumulate(self, learn):
        preds, targs = learn.pred, learn.y
        preds_sigmoid = torch.sigmoid(preds)
        # Binary predictions using a threshold of 0.5
        preds_binary = (preds_sigmoid > 0.5).int()
        # Identify the missed positives: true positives that are not predicted as positives
        missed_positives_mask = (targs.bool() & (preds_binary == 0))
        self.preds_missed.append(preds_sigmoid[missed_positives_mask].cpu().numpy())
        self.count += missed_positives_mask.sum().item()
        
    @property
    def value(self):
        all_preds = np.concatenate(self.preds_missed)
        return all_preds.mean() if len(all_preds) > 0 else None


class AvgDiffOrdinalPos(Metric):
    def __init__(self): self.pos_diffs, self.count = [], 0

    def accumulate(self, learn):
        preds, targs = learn.pred, learn.y
        preds_sigmoid = torch.sigmoid(preds)
        for targ_row, pred_row in zip(targs, preds_sigmoid):
            sorted_pred_indices = pred_row.argsort(descending=True)
            for true_pos_index in targ_row.nonzero(as_tuple=True)[0]:
                model_rank = (sorted_pred_indices == true_pos_index).nonzero(as_tuple=True)[0]
                self.pos_diffs.append(model_rank.item())
                self.count += 1

    @property
    def value(self):
        return np.mean(self.pos_diffs) if self.pos_diffs else None

################################### LOSS ####################################

# Convert labels to integer
df_labels.iloc[:, 2:9] = df_labels.iloc[:, 2:9].astype(int)

# Calculate frequencies of each label
label_freq = df_labels.iloc[:, 2:9].sum(axis=0) / len(df_labels)

# Calculate inverse class frequencies
inverse_freq = 1 / label_freq

print(f"label_freq: {label_freq}", flush=True)

weights = torch.tensor(inverse_freq.values)
loss_func = BCEWithLogitsLossFlat(pos_weight=weights.to(device))

# loss_func = BCEWithLogitsLossFlat()

################################## MODELS ###################################
# List of models to try
models = [
    {
        'model': RNNPlus,
        'params': [
            {
                "hidden_size": 512,
                "n_layers": 1,
                "bidirectional": True,
            },
        ]
    },
    {
        'model': LSTMPlus,
        'params': [
            {
                "hidden_size": 512,
                "n_layers": 1,
                "bidirectional": True,
            },
        ]
    },
    {
        'model': GRUPlus,
        'params': [
            {
                "hidden_size": 512,
                "n_layers": 1,
                "bidirectional": True,
            },
        ]
    },
    {
        'model': TransformerRNNPlus,
        'params': [
            {
                "nhead": 8,
                "d_model": 128,
                "dim_feedforward": 256,
                "num_encoder_layers": 1,
                "num_rnn_layers": 1,
                "dropout": 0.1,
                "bidirectional": True,
            },
        ]
    },
    {
        'model': TransformerLSTMPlus,
        'params': [
            {
                "nhead": 8,
                "d_model": 128,
                "dim_feedforward": 256,
                "num_encoder_layers": 1,
                "num_rnn_layers": 1,
                "dropout": 0.1,
                "bidirectional": True,
            },
        ]
    },
    {
        'model': TransformerGRUPlus,
        'params': [
            {
                "nhead": 8,
                "d_model": 256,
                "dim_feedforward": 512,
                "num_encoder_layers": 1,
                "num_rnn_layers": 1,
                "dropout": 0.1,
                "bidirectional": True,
            },
        ]
    },
    {
        'model': TransformerModel,
        'params': [
            {
                "n_head": 8,
                "d_model": 128,
                "d_ffn": 256,
                "n_layers": 1,
                "activation": 'gelu',
            },
        ]
    },
    {
        'model': TSTPlus,
        'params': [
            {
                "d_model": 256,
                "n_heads": 8,
                "d_ff": 512,
                "n_layers": 2,
                "dropout": 0.1,
                "act": 'gelu'
            },
        ]
    },
    # won't work without special instatiation
    # {
    #     'model': PatchTST
    #     'params': [
    #         {
    #             "d_model": 32,
    #             "n_heads": 1,
    #             "d_ff": 64,
    #             "n_layers": 3,
    #             "dropout": 0.1,
    #         },
    #     ]
    # },
    {
        'model': RNNAttention,
        'params': [
            {
                "hidden_size": 128,
                "bidirectional": True,
                "dropout": 0.1,
                "n_heads": 8,
                "d_ff": 256,
            },
        ]
    },
    {
        'model': LSTMAttention,
        'params': [
            {
                "hidden_size": 128,
                "bidirectional": True,
                "dropout": 0.1,
                "n_heads": 8,
                "d_ff": 256,
            },
        ]
    },
    {
        'model': GRUAttention,
        'params': [
            {
                "hidden_size": 256,
                "bidirectional": True,
                "dropout": 0.1,
                "n_heads": 8,
                "d_ff": 512,
            },
        ]
    },
]

################################## TRAIN ###################################

prob_sums = None
pred_sums = None
targets = None


for model in models:
    model_class = model['model']
    hyperparams_list = model['params']
    for i, hyperparams in enumerate(hyperparams_list):
        print(f"Training model {model_class.__name__} {i+1} with hyperparameters: {hyperparams}", flush=True)

        # Instantiate the model
        if model_class != TransformerModel:
            model_instance = model_class(
                c_in=n_features, 
                c_out=n_classes, 
                seq_len=max_seq_len,
                **hyperparams
            ).to(device)
        else:
            model_instance = model_class(
                c_in=n_features, 
                c_out=n_classes, 
                # seq_len=max_seq_len,
                **hyperparams
            ).to(device)

        if model_class == TSTPlus:
            # Create TSDatasets
            dsets = TSDatasets(X_all, y_all, splits=splits, tfms=[None, [MultiCategorize()]])
            test_ds = TSDatasets(X_test_padded, y_test, tfms=[None, [MultiCategorize()]], splits=None) # No splits needed for test set

            # Create DataLoaders
            dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size])
            test_dl =  TSDataLoader(test_ds, batch_size=batch_size)

        else:
            # replace na with 0
            X_all = np.nan_to_num(X_all)
            X_test_padded = np.nan_to_num(X_test_padded)

            # Create TSDatasets
            dsets = TSDatasets(X_all, y_all, splits=splits, tfms=[None, [MultiCategorize()]])
            test_ds = TSDatasets(X_test_padded, y_test, tfms=[None, [MultiCategorize()]], splits=None)

            # Create DataLoaders
            dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size])
            test_dl =  TSDataLoader(test_ds, batch_size=batch_size)


        # Create output directory if it doesn't exist
        output_dir_path = f'{output_dir}/{model_class.__name__}'
        os.makedirs(output_dir_path, exist_ok=True)
            
        # Create the learner
        learn = ts_learner(dls, arch=model_instance, metrics=[accuracy_multi, subset_accuracy, RocAucMulti(average='micro'),
            F1ScoreMulti(average='micro'), RecallMulti(average='micro'),
            #  SamplewiseRecall(),
              PrecisionMulti(average='micro'),
            AvgProbPositiveClass(), AvgProbMissedPositiveClass(), AvgDiffOrdinalPos(),
            ],
            loss_func = loss_func,
            cbs = [CSVLogger(fname=f'{output_dir}/{model_class.__name__}/{model_class.__name__}_{i+1}_logs.csv')])
                        # EarlyStoppingCallback(monitor='roc_auc_score', comp=np.greater, min_delta=0.0005, patience=7)],

        if model_class in [TransformerModel]:
            lrs_max = 7e-4
            max_epochs = 35
        elif model_class in [TSTPlus]:
            lrs_max = 5e-4
            max_epochs = 30
        elif model_class in [RNNAttention]:
            lrs_max = 1e-4
            max_epochs = 35
        elif model_class in [GRUAttention]:
            lrs_max = 5e-5
            max_epochs = 35
        elif model_class in [LSTMAttention]:
            lrs_max = 15e-5
            max_epochs = 25
        elif model_class in [RNNPlus]:
            lrs_max = 2e-4
            max_epochs = 70
        elif model_class in [GRUPlus]:
            lrs_max = 1e-3
            max_epochs = 35
        elif model_class in [LSTMPlus]:
            lrs_max = 7e-4
            max_epochs = 35
        elif model_class in [TransformerRNNPlus]:
            lrs_max = 1e-3
            max_epochs = 30
        elif model_class in [TransformerGRUPlus]:
            lrs_max = 9e-4
            max_epochs = 35
        elif model_class in [TransformerLSTMPlus]:
            lrs_max = 15e-4
            max_epochs = 45

        
        

        # Train the model
        with learn.no_logging():
            with learn.no_bar():
                learn.fit_one_cycle(n_epoch=max_epochs, lr_max = lrs_max)

        ################################## PROBABILITY  ###################################

        class IdentityIsotonicRegression:
            def fit(self, X, y):
                pass
            
            def fit_transform(self, X, y):
                return X

            def transform(self, X):
                return X

        class LogisticCalibrator(BaseEstimator, TransformerMixin):
            def __init__(self):
                self.lr = LogisticRegression()
                
            def fit(self, X, y):
                X = self._reshape(X)
                self.lr.fit(X, y)
                return self

            def transform(self, X):
                X = self._reshape(X)
                return self.lr.predict_proba(X)[:, 1]

            def fit_transform(self, X, y):
                self.fit(X, y)
                return self.transform(X)
            
            def _reshape(self, X):
                if len(X.shape) == 1:
                    return X.reshape(-1, 1)
                return X

        def calibrate_probs(dataloader):
            # Get predictions and targets
            probs, targets = learn.get_preds(dl=dataloader)

            # Convert to numpy for use with sklearn
            probs = probs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            calibrated_probs = []
            isotonic_models = []

            # Iterate through each class to calibrate the probabilities with isotonic regression
            for i in range(targets.shape[1]):
                if i in classes_to_fit:
                    isotonic = IsotonicRegression(out_of_bounds='clip')
                    calibrated_probs.append(isotonic.fit_transform(probs[:, i], targets[:, i]))
                    isotonic_models.append(isotonic)
                else:
                    isotonic = IdentityIsotonicRegression()
                    calibrated_probs.append(isotonic.fit_transform(probs[:, i], targets[:, i]))
                    isotonic_models.append(isotonic)

            # Convert back to torch tensor
            calibrated_probs = torch.tensor(calibrated_probs, device=device).T
            targets = torch.tensor(targets, device=device)
            
            return calibrated_probs, targets, isotonic_models


        def apply_calibration(probs, isotonic_models):
            # Convert to numpy for use with sklearn
            probs = probs.detach().cpu().numpy()

            # Initialize the list for calibrated probabilities
            calibrated_probs = []

            # Iterate through each class to apply the calibration
            for i in range(len(isotonic_models)):
                # Apply calibration with isotonic regression
                calibrated_probs.append(isotonic_models[i].transform(probs[:, i]))

            # Convert back to torch tensor
            calibrated_probs = torch.tensor(calibrated_probs, device=device).T
            
            return calibrated_probs
        
        def make_calibration_curve(probs, labels, calibrated_probs, model_name, optimal_thresholds=None):
            probs = probs.cpu()
            labels = labels.cpu()
            calibrated_probs = calibrated_probs.cpu()

            # Determine the size of the grid
            num_labels = labels.shape[1]
            grid_size = math.ceil(num_labels)

            fig, axs = plt.subplots(grid_size, 2, figsize=(15, 2*grid_size))  # 2 columns for uncalibrated and calibrated

            # Loop through each label
            for i in range(num_labels):
                fraction_of_positives, mean_predicted_value = calibration_curve(labels[:, i], probs[:, i], n_bins=15)
                calibrated_fraction_of_positives, calibrated_mean_predicted_value = calibration_curve(labels[:, i], calibrated_probs[:, i], n_bins=15)

                # Plot uncalibrated curve
                axs[i, 0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                axs[i, 0].plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Uncalibrated {i+1}")
                axs[i, 0].axvline(x=0.5, color='y', linestyle='--', label='Threshold 0.5')
                axs[i, 0].set_ylabel('Fraction of positives')
                axs[i, 0].set_xlabel('Mean predicted value')
                axs[i, 0].legend()
                if i == 0:
                    axs[i, 0].set_title("Uncalibrated")

                # Plot calibrated curve
                axs[i, 1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                axs[i, 1].plot(calibrated_mean_predicted_value, calibrated_fraction_of_positives, "s-", label=f"Calibrated {i+1}")
                if optimal_thresholds is not None:
                    axs[i, 1].axvline(x=optimal_thresholds[i], color='r', linestyle='--', label=f'Threshold {optimal_thresholds[i]:.2f}')
                axs[i, 1].set_ylabel('Fraction of positives')
                axs[i, 1].set_xlabel('Mean predicted value')
                axs[i, 1].legend()
                if i == 0:
                    axs[i, 1].set_title("Calibrated")

            # Save the figure
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            plt.savefig(f'{model_dir}/calibration_curve_grid.png')
            plt.close()  # Close the figure to free up memory



        ################################## OPTIMIZE ###################################

        def find_optimal_threshold(predicted, target):
            thresholds = np.linspace(0, 1, 101)  # Adjust the number of thresholds as needed
            f1_scores = []
            
            for threshold in thresholds:
                # Convert predicted probabilities to binary predictions based on the threshold
                binary_predictions = (predicted >= threshold).int()

                f1_scores.append(F1ScoreMulti(average='micro')(binary_predictions, target))
            
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            return optimal_threshold

        def optimize_thresholds(dataloader, calibrated_probs=None, targets=None):
            # Get predictions and targets
            if calibrated_probs is None:
                probs, targets = learn.get_preds(dl=dataloader)
            else:
                targets = targets
                probs = calibrated_probs

            # Initialize the list for optimal thresholds
            optimal_thresholds = []

            # Iterate through each class
            for i in range(targets.shape[1]):
                if i in classes_to_fit:
                    optimal_thresholds.append(find_optimal_threshold(probs[:, i], targets[:, i]))
                else:
                    # optimal_thresholds.append(0.5)
                    optimal_thresholds.append(find_optimal_threshold(probs[:, i], targets[:, i]))

            return optimal_thresholds


        ################################## EVALUATE ###################################

        def make_preds(dataloader, isotonic_models, optimal_thresholds, calibrated=False):
            # Get predictions and targets
            if calibrated:
                probs, targets = learn.get_preds(dl=dataloader)
                probs = apply_calibration(probs, isotonic_models)
                probs_np = probs.cpu().numpy()
            else:
                probs, targets = learn.get_preds(dl=dataloader)
                probs_np = np.array(probs.tolist())
            
            # Convert the optimal thresholds to numpy array
            optimal_thresholds_np = np.array(optimal_thresholds)

            # Initialize predictions with zeros
            preds = np.zeros_like(probs_np)

            # Apply thresholds to get the predictions
            for i in range(probs_np.shape[1]):
                preds[:, i] = (probs_np[:, i] >= optimal_thresholds_np[i]).astype(int)

            if calibrated:
                preds = torch.tensor(preds, device=device)
                targets = targets.to(device)
            else:
                preds = torch.from_numpy(preds)
            
            return preds, targets, probs

        def binary_accuracy(preds, targets):
            return (preds == targets).float().mean()
        
        def write_metrics_to_csv(metrics_list, model_name, calibrated):
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            if calibrated:
                filepath = os.path.join(model_dir, f'{model_name}_{i+1}_eval_calibrated.csv')
            else:
                filepath = os.path.join(model_dir, f'{model_name}_{i+1}_eval.csv')

            df = pd.DataFrame(metrics_list)
            df.to_csv(filepath, index=False)

        def evaluate(dataloader, isotonic_models, optimal_thresholds, calibrated=False):
            preds, targets, probs = make_preds(dataloader, isotonic_models, optimal_thresholds, calibrated)

            metrics_list = []

            model_metrics = {
                "Model": "Full Model",
                "Accuracy": accuracy_multi(preds, targets).tolist(),
                "Subset Accuracy": subset_accuracy(preds, targets).tolist(),
                "ROC AUC Score": RocAucMulti(average='micro')(probs, targets),
                "F1 Score": F1ScoreMulti(average='micro')(preds, targets),
                "Recall": RecallMulti(average='micro')(preds, targets),
                "Precision": PrecisionMulti(average='micro')(preds, targets)
            }
            metrics_list.append(model_metrics)

            print("Metrics:")
            for name, value in model_metrics.items():
                print(f"{name}: {value}")

            print("Per class metrics:")
            for i in range(targets.shape[1]):
                targets_i = targets[:, i]
                preds_i = preds[:, i]
                probs_i = probs[:, i]

                class_metrics = {
                    "Model": f"Class {index_to_classname[i]}",
                    "Accuracy": binary_accuracy(preds_i, targets_i).tolist(),
                    "ROC AUC Score": RocAuc()(probs_i, targets_i),
                    "F1 Score": F1Score()(preds_i, targets_i),
                    "Recall": Recall()(preds_i, targets_i),
                    "Precision": Precision()(preds_i, targets_i)
                }

                # for name, value in class_metrics.items():
                #     print(f"  {name}: {value}")

                metrics_list.append(class_metrics)

            write_metrics_to_csv(metrics_list, model_class.__name__, calibrated)

        
        # Calibrate probabilities
        calibrated_probs, targets, isotonic_models = calibrate_probs(dls.train)

        # Find optimal thresholds without calibration
        optimal_thresholds = optimize_thresholds(dls.train)

        print(f"Optimal thresholds: {optimal_thresholds}")

        # Find optimal thresholds with calibration
        optimal_thresholds_cali = optimize_thresholds(dls.train, calibrated_probs=calibrated_probs, targets=targets)

        print(f"Optimal calibrated thresholds: {optimal_thresholds_cali}")
        
        # Evaluate using valiation set

        # # Evaluate with no thresholds
        # print("Evaluating validation set without thresholds...")
        # evaluate(dls.valid, isotonic_models, [0.5] * len(optimal_thresholds))

        # # Evaluate with calibration and thresholds
        # print("Evaluating validation set with calibration and calibrated thresholds...")
        # evaluate(dls.valid, isotonic_models, optimal_thresholds_cali, calibrated=True)

        # # Plot calibration curve and save figure for uncalibrated and calibrated probabilities
        # print("Plotting calibration curves...")
        # preds, targets, probs = make_preds(dls.valid, isotonic_models, optimal_thresholds)
        # _, _, calibrated_probs = make_preds(dls.valid, isotonic_models, optimal_thresholds_cali, calibrated=True)
        # make_calibration_curve(probs, targets, calibrated_probs, model_class.__name__, optimal_thresholds_cali)

        # # Initialize or update running sum of probabilities
        # if prob_sums is None:
        #     prob_sums = probs
        # else:
        #     prob_sums += probs

        # if pred_sums is None:
        #     pred_sums = preds
        # else:
        #     pred_sums += preds




        # Evaluate using test set

        # Evaluate with no thresholds
        print("Evaluating test set without thresholds...")
        evaluate(test_dl, isotonic_models, [0.5] * len(optimal_thresholds))

        # Evaluate with calibration and thresholds
        print("Evaluating test set with calibration and calibrated thresholds...")
        evaluate(test_dl, isotonic_models, optimal_thresholds_cali, calibrated=True)

        # Plot calibration curve and save figure for uncalibrated and calibrated probabilities
        print("Plotting calibration curves...")
        preds, targets, probs = make_preds(test_dl, isotonic_models, optimal_thresholds)
        _, _, calibrated_probs = make_preds(test_dl, isotonic_models, optimal_thresholds_cali, calibrated=True)
        make_calibration_curve(probs, targets, calibrated_probs, model_class.__name__, optimal_thresholds_cali)    

        # Initialize or update running sum of probabilities
        if prob_sums is None:
            prob_sums = probs
        else:
            prob_sums += probs

        if pred_sums is None:
            pred_sums = preds
        else:
            pred_sums += preds

        print("Done")


print("Ensemble model:")
i = 1
model_class = 'ensemble'


# Calibrate probabilities
calibrated_probs, targets, isotonic_models = calibrate_probs(dls.train)

# Find optimal thresholds with calibration
optimal_thresholds_cali = optimize_thresholds(dls.train, calibrated_probs=calibrated_probs, targets=targets)

print(f"Optimal calibrated thresholds: {optimal_thresholds_cali}")



# Calculate ensemble probabilities and predictions
probs = prob_sums / len(models)
_, targets = learn.get_preds(dl = test_dl)

# Calculate preds by dividing by len(preds) and rounding
preds = pred_sums / len(models)
preds = torch.round(preds)

preds = torch.tensor(preds, device=device)
targets = targets.to(device)

metrics_list = []

model_metrics = {
    "Model": "Full Model",
    "Accuracy": accuracy_multi(preds, targets).tolist(),
    "Subset Accuracy": subset_accuracy(preds, targets).tolist(),
    "ROC AUC Score": RocAucMulti(average='micro')(probs, targets),
    "F1 Score": F1ScoreMulti(average='micro')(preds, targets),
    "Recall": RecallMulti(average='micro')(preds, targets),
    "Precision": PrecisionMulti(average='micro')(preds, targets)
}
metrics_list.append(model_metrics)

print("Metrics:")
for name, value in model_metrics.items():
    print(f"{name}: {value}")

print("Per class metrics:")
for i in range(targets.shape[1]):
    targets_i = targets[:, i]
    preds_i = preds[:, i]
    probs_i = probs[:, i]

    class_metrics = {
        "Model": f"Class {index_to_classname[i]}",
        "Accuracy": binary_accuracy(preds_i, targets_i).tolist(),
        "ROC AUC Score": RocAuc()(probs_i, targets_i),
        "F1 Score": F1Score()(preds_i, targets_i),
        "Recall": Recall()(preds_i, targets_i),
        "Precision": Precision()(preds_i, targets_i)
    }

    for name, value in class_metrics.items():
        print(f"  {name}: {value}")

    metrics_list.append(class_metrics)

write_metrics_to_csv(metrics_list, model_class, calibrated= False)




# # calibrate probabilities
# probs = apply_calibration(probs, isotonic_models)
probs_np = probs.cpu().numpy()

# Convert the optimal thresholds to numpy array
optimal_thresholds_np = np.array(optimal_thresholds_cali)

# Initialize predictions with zeros
preds = np.zeros_like(probs_np)

# Apply thresholds to get the predictions
for i in range(probs_np.shape[1]):
    preds[:, i] = (probs_np[:, i] >= optimal_thresholds_np[i]).astype(int)

preds = torch.tensor(preds, device=device)
targets = targets.to(device)


metrics_list = []

model_metrics = {
    "Model": "Full Model",
    "Accuracy": accuracy_multi(preds, targets).tolist(),
    "Subset Accuracy": subset_accuracy(preds, targets).tolist(),
    "ROC AUC Score": RocAucMulti(average='micro')(probs, targets),
    "F1 Score": F1ScoreMulti(average='micro')(preds, targets),
    "Recall": RecallMulti(average='micro')(preds, targets),
    "Precision": PrecisionMulti(average='micro')(preds, targets)
}
metrics_list.append(model_metrics)

print("Metrics:")
for name, value in model_metrics.items():
    print(f"{name}: {value}")

print("Per class metrics:")
for i in range(targets.shape[1]):
    targets_i = targets[:, i]
    preds_i = preds[:, i]
    probs_i = probs[:, i]

    class_metrics = {
        "Model": f"Class {index_to_classname[i]}",
        "Accuracy": binary_accuracy(preds_i, targets_i).tolist(),
        "ROC AUC Score": RocAuc()(probs_i, targets_i),
        "F1 Score": F1Score()(preds_i, targets_i),
        "Recall": Recall()(preds_i, targets_i),
        "Precision": Precision()(preds_i, targets_i)
    }

    for name, value in class_metrics.items():
        print(f"  {name}: {value}")

    metrics_list.append(class_metrics)

write_metrics_to_csv(metrics_list, model_class, calibrated= True)

# Plot calibration curve and save figure for uncalibrated and calibrated probabilities
print("Plotting calibration curves...")
preds, targets, probs = make_preds(test_dl, isotonic_models, optimal_thresholds)
_, _, calibrated_probs = make_preds(test_dl, isotonic_models, optimal_thresholds_cali, calibrated=True)
make_calibration_curve(probs, targets, calibrated_probs, model_class, optimal_thresholds_cali)    
