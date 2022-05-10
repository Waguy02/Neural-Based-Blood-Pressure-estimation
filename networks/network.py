import logging
import os
import torch
import torchvision.models
from torch import nn
from constants import ROOT_DIR
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
class CustomNetwork(nn.Module):
    def __init__(self, model_name="my_model",reset=False,load_best=True):
        super(CustomNetwork, self).__init__()
        self.model_name = model_name
        self.reset = reset
        self.load_best = load_best
        self.setup_dirs()
        self.setup_networks()
    ##1. Defining network architecture
    def setup_network(self):
        """
        Initialize the network  architecture here
        @return:
        """
        pass


    ##2. Model Saving/Loading
    def load_state(self):
        """
        Load model
        :param self:
        :return:
        """
        if  self.load_best and os.path.exists (self.save_best_file):
            logging.info(f"Loading best model state : {self.save_file}")
            self.load_state_dict(torch.load(self.save_file,map_location=device))
            return 
            
        if os.path.exists(self.save_file):
            logging.info(f"Loading model state : {self.save_file}")
            self.load_state_dict(torch.load(self.save_file,map_location=device))
    def save_state(self,best=False):
        if best:
            torch.save(self.state_dict(), self.save_best_file)
        else:
            torch.save(self.state_dict(), self.save_file)

    ##3. Setupping directories for weights /logs ... etc
    def setup_dirs(self):
        """
        Checking and creating directories for weights storage
        @return:
        """
        self.save_path = os.path.join(ROOT_DIR, 'zoos')
        self.model_dir = os.path.join(self.save_path, self.model_name)
        self.save_file = os.path.join(self.model_dir, f"{self.model_name}.pt")
        self.save_best_file = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    #4. Forward call
    def forward(self, input):
        # In this function we pass the 3 images and get the 3 embeddings
        "Forward call here"
        pass






