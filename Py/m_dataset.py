import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tsai.data.all import TSDatasets

class mDataset(Dataset):
    def __init__(self, data_input, data_labels):
        self.data_input = data_input
        self.data_labels = data_labels

        # Group by patient ID and sort by time
        self.patients = self.data_input.groupby('pid').apply(lambda x: x.sort_values('dtime'))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        # Get the patient's data
        patient_id = self.patients.pid.unique()[idx]
        patient_data = self.patients[self.patients.pid == patient_id]

        # The inputs are the diagnosis codes and the times
        X = patient_data.iloc[:, 1:].values.astype(float) # pid is first column

        # The outputs are fetched from the output data using the patient ID
        y = self.data_labels.loc[self.data_labels['pid'] == patient_id, self.data_labels.columns[2:9]].values

        if len(y) > 1:
            print(f"Found multiple labels for pid: {patient.pid}", flush=True)
            print(y, flush=True)
        return X, y
