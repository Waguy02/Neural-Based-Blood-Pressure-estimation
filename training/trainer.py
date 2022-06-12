import json
import logging
import os
import shutil
import subprocess
import torch
from torch.utils.tensorboard import SummaryWriter
from constants import TENSORBOARD_DIR
from tqdm import tqdm
# CUDA for PyTorch
from my_utils import Averager
from networks.network import CustomNetwork
from dataset.dataset import create_dataloader, DatasetType

use_cuda = torch .cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Trainer:
    """
    Class to manage the full training pipeline
    """

    def __init__(self, network:CustomNetwork, loss, optimizer, nb_epochs=10, batch_size=128, num_workers=4, reset=False, autorun_tb=False):
        """
        @param network:
        @param dataset_name:
        @param images_dirs:
        @param loss:
        @param optimizer:
        @param nb_epochs:
        @param nb_workers: Number of worker for the dataloader
        """
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.network=network
        self.optimizer=optimizer
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.loss=loss
        self.nb_epochs=nb_epochs
        self.train_data_loader=create_dataloader(type=DatasetType.TRAIN,batch_size=batch_size,num_workers=num_workers)
        self.val_data_loader=create_dataloader(type=DatasetType.VAL,batch_size=batch_size,num_workers=num_workers)
        self.tb_dir=os.path.join(TENSORBOARD_DIR,self.network.model_name)
        self.epoch_index_file=os.path.join(self.tb_dir,"epoch_index.json")

        if reset:
            if os.path.exists(self.tb_dir):
                shutil.rmtree(self.tb_dir)
        if not os.path.exists(self.tb_dir):
            os.makedirs(self.tb_dir) 
        self.summary_writer = SummaryWriter(log_dir=self.tb_dir)
        self.start_epoch=0
        if not reset and os.path.exists(self.epoch_index_file):
            with open(self.epoch_index_file, "r") as f:
                self.start_epoch=json.load(f)["epoch"]+1
                self.nb_epochs+=self.start_epoch
                logging.info("Resuming from epoch {}".format(self.start_epoch))
        self.autorun_tb=autorun_tb

    def run_tensorboard(self):
        """
        Launch tensorboard
        @return:
        """
        cmd = f"tensorboard --logdir '{self.tb_dir}' --host \"0.0.0.0\" --port 6007"
        _ = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, shell=True)
    def fit(self):
        if self.autorun_tb:self.run_tensorboard()
        logging.info("Launch training on {}".format(device))
        self.network.to(device)
        itr=self.start_epoch*len(self.train_data_loader)*self.batch_size##Global counter for steps
        best_loss=1e20#infinity
        if os.path.exists(os.path.join(self.tb_dir,"best_model_info.json")):
            with open(os.path.join(self.tb_dir,"best_model_info.json"), "r") as f:
                best_model_info=json.load(f)
                best_loss=best_model_info["eval_loss"]
        for epoch in range(self.start_epoch,self.nb_epochs): #Training loop
            self.network.train()

            """"
            0. Initialize loss and other metrics
            """
            running_loss=Averager()

            for _, batch in enumerate(tqdm(self.train_data_loader,desc=f"Epoch {epoch+1}/{self.nb_epochs}")):
                """
                Training lopp
                """
                itr+=self.batch_size
                

                """
                1.Forward pass
                """
                output =self.network(batch['image'].to(device))

                """
                2.Loss computation and other metrics
                """
                loss=self.loss(output,batch['label'].to(device))
                loss_value=loss.cpu().item()
                running_loss.send(loss_value)



                """
                3.Optimizing
                """
                #loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


                """
                4.Writing logs and tensorboard data, loss and other metrics
                """
                self.summary_writer.add_scalar("Train/loss",loss_value,itr)


            epoch_loss=running_loss.value
            self.summary_writer.add_scalar("Train/epoch_loss",epoch_loss,epoch)

            # Saving the model at the end of the epoch is better than the preious best one
            self.network.save_state()
            
            epoch_loss_val=self.eval(epoch)
            self.scheduler.step(epoch_loss_val)
            if epoch_loss_val<best_loss:
                logging.info("Saving the best model")
                best_loss=epoch_loss_val
                self.network.save_state(best=True)
                with open(os.path.join(self.tb_dir,"best_model_info.json"), "w") as f:
                    f.write(json.dumps({"train_loss":float(epoch_loss),
                     "eval_loss":float(epoch_loss_val),"epoch":epoch
                     
                    },indent=4))
    def eval(self,epoch):
        """
        Compute loss and metrics on a validation dataloader
        @return:
        """
        with torch.no_grad():
            self.network.eval()
            running_loss = Averager()
            for _, batch in enumerate(tqdm(self.val_data_loader, desc=f"Eval Epoch {epoch + 1}/{self.nb_epochs}")):
                """
                Training lopp
                """
                """
                1.Forward pass
                """
                output = self.network(batch['image'].to(device))



                """
                2.Loss computation and other metrics
                """
                loss = self.loss(output, batch['label'].to(device))
                loss_value = loss.cpu().item()
                running_loss.send(loss_value)


            epoch_loss = running_loss.value
            self.summary_writer.add_scalar("Validation/epoch_loss", epoch_loss, epoch)

            return epoch_loss