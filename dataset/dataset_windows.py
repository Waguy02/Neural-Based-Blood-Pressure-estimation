import os
from enum import Enum

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from constants import WINDOWS_DATA_DIR

WINDOWS_PER_FILE=6

class DatasetType(Enum):
    TRAIN="train"
    VAL= "val"
    TEST="test"
class DatasetWindows(Dataset):
    def __init__(self, type):
        self.type=type
        self.load_data()
        pass
    def init_transforms(self):
        """
        Initialize transforms.Might be different for each dataset type
        """
        # if self.type==DatasetType.TRAIN or self.type==DatasetType.VALID:
        #     self.transforms = transforms.Compose([
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])])
        # else :
        #     self.transforms = transforms.Compose([
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406])])


    def load_data(self):
        """
        Load data from the data items if necessary
        Returns:
        """
        self.data_file=os.path.join(WINDOWS_DATA_DIR, self.type.value,"all_subjects.csv")
        self.datas=pd.read_csv(self.data_file,dtype=np.float32).dropna().to_numpy()
        pass
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        """
        How to retrieve item from the dataset
        Args:
            idx:
        Returns:
        """
        current_row=self.datas[idx]

        features=torch.unsqueeze(torch.as_tensor(current_row[:-2]),1)
        features=torch.transpose(features,0,1)
        sbp,dbp=torch.as_tensor(current_row[-2:-1]),torch.as_tensor(current_row[-1:])
        return features,sbp,dbp







