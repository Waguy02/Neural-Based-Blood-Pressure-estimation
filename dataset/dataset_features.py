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
class DatasetType(Enum):
    TRAIN="train"
    VAL= "val"
    TEST="test"
class DatasetFeatures(Dataset):
    def __init__(self,features, type):
        self.features=features
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
        self.datas=pd.read_csv(self.data_file,dtype=np.float32).dropna()
        self.datas["SUT_DT_ADD"]=self.datas["SUT"]+self.datas["DT"]
        self.features=self.datas[self.features].to_numpy()
        self.targets=self.datas[["SBP","DBP"]].to_numpy()

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

        features=torch.as_tensor(self.features[idx])

        sbp,dbp=torch.as_tensor(self.targets[idx][:1]),torch.as_tensor(self.targets[idx][1:])
        return features,sbp,dbp








