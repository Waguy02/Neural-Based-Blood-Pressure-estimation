import json
import os
from unicodedata import category
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from constants import DATA_DIR


from enum import Enum
class DatasetType(Enum):
    TRAIN="train"
    VALID="valid"
    TEST="test"
class CustomDataset(Dataset):
    def __init__(self, type):
        self.type=type
        self.init_transforms()
        pass
    def init_transforms(self):
        """
        Initialize transforms.Might be different for each dataset type
        """
        if self.type==DatasetType.TRAIN or self.type==DatasetType.VALID:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        else :
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406])])
            pass
    def load_data(self):
        """
        Load data from the data items if necessary
        Returns:

        """
        self.data = []
        pass
    def __len__(self):
        return len(self.data)
        pass
    def __getitem__(self, idx):
        """
        How to retrieve item from the dataset
        Args:
            idx:
        Returns:
        """
        pass

def collate_fn(batch):
    """
    Collate function to bad batchs
    """
    return batch
def create_dataloader(type,batch_size=1,shuffle=False,num_workers=0):
    """
    Create dataloader for the dataset
    """
    dataset = CustomDataset(type)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader





