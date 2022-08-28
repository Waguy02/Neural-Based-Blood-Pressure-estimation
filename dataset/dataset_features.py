import csv
import glob
import json
import os
import random

from unicodedata import category
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from constants import FEATURES_DATA_DIR
from enum import Enum
WINDOWS_PER_FILE=6

class DatasetType(Enum):
    TRAIN="train"
    VAL= "val"
    TEST="test"
class DatasetFeatures(Dataset):
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
        self.data_file=os.path.join(FEATURES_DATA_DIR, self.type.value,"all_subjects.csv")
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
        features,sbp,dbp=torch.as_tensor(current_row[:-2]),torch.as_tensor(current_row[-2:-1]),torch.as_tensor(current_row[-1:])
        return features,sbp,dbp








