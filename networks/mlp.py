import logging
import os
import torch
import torchvision.models
from torch import nn
from constants import ROOT_DIR, N_FEATURES, DEVICE


class MLP(nn.Module):
    def __init__(self, experiment_dir,n_features,reset=False):
        super(MLP, self).__init__()
        self.experiment_dir=experiment_dir
        self.model_name=os.path.basename(self.experiment_dir)
        self.save_experiment_dir=experiment_dir
        self.reset = reset
        self.n_features=n_features
        self.setup_dirs()
        self.setup_network()
        if not reset:self.load_state()
    ##1. Defining network architecture
    def setup_network(self):
        """
        Initialize the network  architecture here
        @return:
        """
        self.base_mlp=torch.nn.Sequential(
            nn.Linear(self.n_features,35),
            nn.Hardsigmoid(),
            nn.BatchNorm1d(35),
            nn.Linear(35, 20),
            nn.Hardsigmoid(),
            nn.BatchNorm1d(20),
        )
        self.sbp_dense=nn.Sequential(
            nn.Linear(20,1),
        )
        self.dbp_dense=nn.Sequential(
            nn.Linear(20,1)

        )


    ##2. Model Saving/Loading
    def load_state(self,best=False):
        """
        Load model
        :param self:
        :return:
        """
        if  best and os.path.exists (self.save_best_file):
            logging.info(f"Loading best model state : {self.save_file}")
            self.load_state_dict(torch.load(self.save_file, map_location=DEVICE))
            return 
            
        if os.path.exists(self.save_file):
            logging.info(f"Loading model state : {self.save_file}")
            self.load_state_dict(torch.load(self.save_file, map_location=DEVICE))
    def save_state(self,best=False):
        if best:
            logging.info("Saving best model")
            torch.save(self.state_dict(), self.save_best_file)
        torch.save(self.state_dict(), self.save_file)

    ##3. Setupping directories for weights /logs ... etc
    def setup_dirs(self):
        """
        Checking and creating directories for weights storage
        @return:
        """
        self.save_file = os.path.join(self.experiment_dir, f"{self.model_name}.pt")
        self.save_best_file = os.path.join(self.experiment_dir, f"{self.model_name}_best.pt")
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    #4. Forward call
    def forward(self, features):
        base_output=self.base_mlp(features)
        sbp,dbp=self.sbp_dense(base_output),self.dbp_dense(base_output)
        return sbp,dbp







