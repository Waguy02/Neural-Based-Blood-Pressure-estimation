import os

import torch

ROOT_DIR=os.path.dirname(os.path.realpath(__file__))
# FEATURES_DATA_DIR= r"F:\Projets\Gaby project\NeuralnetworkBPestimation\data\features_data_mat_21"
FEATURES_DATA_DIR= r"F:\Projets\Gaby project\NeuralnetworkBPestimation\data\features_data_mmic"
WINDOWS_DATA_DIR= r"F:\Projets\Gaby project\NeuralnetworkBPestimation\data\windows_data_mmic"
RAW_DATA_FILE=r"F:\Projets\Gaby project\NeuralnetworkBPestimation\win2_ppg.h5"

N_FEATURES=21
EXPERIMENTS_DIR=os.path.join(ROOT_DIR, "logs/experiments")
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")