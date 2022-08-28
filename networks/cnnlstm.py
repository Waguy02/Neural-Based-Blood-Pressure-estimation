import logging
import os
import torch
import torchvision.models
from torch import nn
from constants import ROOT_DIR, N_FEATURES, DEVICE
import torch.nn.functional as F
FS=125
class CnnLSTM(nn.Module):
    def __init__(self, experiment_dir,reset=False,use_derivative=True):
        super(CnnLSTM, self).__init__()
        self.use_derivative=use_derivative
        self.experiment_dir=experiment_dir
        self.model_name=os.path.basename(self.experiment_dir)
        self.save_experiment_dir=experiment_dir
        self.reset = reset

        self.setup_dirs()
        self.setup_network()
        if not reset:self.load_state()

    ##1. Defining network architecture
    def setup_network(self):
        """
        Initialize the network  architecture here
        @return:
        """
        n_input_channels=1 if not self.use_derivative else 3

        self.conv1=torch.nn.Sequential(
            nn.Conv1d(n_input_channels,2,kernel_size=350,padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(2),
            nn.MaxPool1d(175,stride=1,padding=87))

        self.conv2=torch.nn.Sequential(
            nn.Conv1d(2,10, kernel_size=175, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(25, stride=1, padding=12))

        self.   conv3=torch.nn.Sequential(
            nn.Conv1d(10,20, kernel_size=25, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.MaxPool1d(11, stride=1, padding=5))

        self.conv4=torch.nn.Sequential(
            nn.Conv1d(20,40, kernel_size=10, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(40),
            nn.MaxPool1d(5, stride=1, padding=2))

        self.lstm1=nn.LSTM(input_size=40,hidden_size=128,bidirectional=True,batch_first=True)
        self.lstm2=nn.LSTM(input_size=256, hidden_size=350, bidirectional=True,batch_first=True)

        self.sbp_dense=nn.Linear(700,1)
        self.dbp_dense = nn.Linear(700, 1)



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
    def forward(self, x):
        if self.use_derivative:
            dt1 = (x[:,:, 1:] - x[:,:, :-1]) * FS
            dt2 = (dt1[:,:, 1:] - dt1[:,:, :-1]) * FS
            dt1 = F.pad(dt1,(0, 1, 0, 0, 0, 0))
            dt2 = F.pad(dt2, (0,2, 0, 0, 0, 0))
            x = torch.concat([x, dt1, dt2], axis=1)


        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=x.view(x.shape[0],700,40)

        x=self.lstm1(x)[0]
        x=self.lstm2(x)[1][0]
        x=torch.transpose(x,0,1)
        x = x.reshape(x.shape[0],-1)
        sbp=self.sbp_dense(x)
        dbp=self.dbp_dense(x)
        return sbp,dbp









