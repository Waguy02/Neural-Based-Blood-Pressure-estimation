import os

import torch

ROOT_DIR=os.path.dirname(os.path.realpath(__file__))
FEATURES_DATA_DIR= os.path.join(ROOT_DIR, "data/features_data_mmic_baseline")
WINDOWS_DATA_DIR= os.path.join(ROOT_DIR,"data/windows_data_mmic")
#


RAW_DATA_DIR=os.path.join(ROOT_DIR,"Rec_mimic")

N_FEATURES=21
EXPERIMENTS_DIR=os.path.join(ROOT_DIR, "logs/experiments")
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")